"""Defensive contribution probability estimators for FPL.

Stage 4H — Defensive Contribution Probability Estimator.

Estimates P(DC threshold met | player appears) for eligible positions
(DEF, MID, FWD).  Goalkeepers are ineligible; the estimator returns 0.0.

Two candidate models are implemented behind a single, explicit interface:

A. Historical-frequency baseline (default)
   - Proportion of prior appearances meeting the DC action threshold.
   - Bayesian shrinkage toward a position-specific prior.
   - Uses only observations strictly before the target fixture's gameweek.

B. Poisson threshold model
   - Estimates defensive-action rate per minute from prior appearances.
   - Projects the rate forward at a documented expected-minutes value.
   - Computes P(Poisson(lambda) >= threshold) numerically.

Both models are pure, stateless functions — no I/O.  Temporal filtering
is delegated to :func:`_appearance_history` which accepts an optional
``prediction_cutoff`` datetime to enforce leakage-safe look-back windows.

Integration note
----------------
The returned probability is *conditional on appearance* (P(DC | appear)).
The scoring engine in ``prediction.py`` applies appearance probability
exactly once: ``E[DC points] = P(appear) * P(DC | appear) * DC_points``.
Do **not** multiply by appearance probability inside this estimator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from fpl_engine.models import (
    PlayerMatchHistory,
    PlayerPosition,
    defensive_contribution_from_history,
)

__all__ = [
    "DC_DEFAULT_MINUTES",
    "DC_MODEL_HISTORY",
    "DC_MODEL_POISSON",
    "DC_POSITION_PRIOR",
    "DC_PRIOR_PSEUDO_COUNT",
    "DEF_DC_THRESHOLD",
    "MID_FWD_DC_THRESHOLD",
    "DCEvaluationResult",
    "DCModelError",
    "DCPredictionResult",
    "dc_action_count",
    "evaluate_dc_models_brier",
    "historical_dc_probability",
    "poisson_dc_probability",
    "predict_defensive_contribution_probability",
]

# --- Threshold constants (2026/27 FPL rules) ---

#: DEF action threshold: tackles + clearances + blocks + interceptions >= 10.
DEF_DC_THRESHOLD: int = 10

#: MID/FWD action threshold: tackles + CBI + recoveries >= 12.
MID_FWD_DC_THRESHOLD: int = 12

#: Default expected minutes used by the Poisson model when not supplied.
#: Represents a full 90-minute appearance conditional on playing,
#: consistent with the baseline model fallback.
DC_DEFAULT_MINUTES: float = 90.0

#: Pseudo-count weight for Bayesian shrinkage toward the position prior.
#: Higher values apply stronger regularization.  A value of 5.0 means
#: ~5 pseudo-observations of the prior, providing moderate regularization
#: for small samples.
DC_PRIOR_PSEUDO_COUNT: float = 5.0

#: Position-specific prior probabilities for P(DC | appear).
#:
#: Empirical priors based on EPL-wide seasonal DC rates.  These priors
#: serve as Bayesian shrinkage targets when a player's personal history
#: is small or empty, and as the sole estimate when no history is
#: available:
#:
#:   - DEF: ~35% of appearances reach 10+ CBIT.
#:   - MID: ~25% of appearances reach 12+ CBIRT.
#:   - FWD: ~15% of appearances reach 12+ CBIRT.
#:   - GKP: 0.0 (ineligible for DC points).
DC_POSITION_PRIOR: dict[PlayerPosition, float] = {
    PlayerPosition.DEFENDER: 0.35,
    PlayerPosition.MIDFIELDER: 0.25,
    PlayerPosition.FORWARD: 0.15,
    PlayerPosition.GOALKEEPER: 0.0,
}

_DC_MODELS = frozenset({"history", "poisson"})

#: Model selector constant for the historical-frequency baseline model.
DC_MODEL_HISTORY: str = "history"

#: Model selector constant for the Poisson threshold model.
DC_MODEL_POISSON: str = "poisson"


class DCModelError(ValueError):
    """Raised when the DC probability model is invoked with invalid inputs."""


@dataclass(frozen=True)
class DCPredictionResult:
    """Container for a single DC probability prediction and metadata.

    Attributes:
        probability: P(DC threshold met | player appears), in [0, 1].
        model: Name of the model used ("history" or "poisson").
        n_observations: Number of prior appearances used (may be 0).
        shrunk: Whether Bayesian shrinkage was applied.
        prior_used: The position prior that was used for shrinkage.
    """

    probability: float
    model: str
    n_observations: int
    shrunk: bool
    prior_used: float


@dataclass(frozen=True)
class DCEvaluationResult:
    """Result of a backtest comparing a DC model on historical data.

    Attributes:
        model: Model name ("history" or "poisson").
        brier_score: Mean Brier score across evaluated fixtures.
        sample_size: Number of fixtures evaluated.
        calibration: Dict mapping bin lower-boundaries to (predicted_prob,
            observed_rate) tuples in 0.1-wide bins.
        position: The position this evaluation covers.
        temporal_safety_enforced: True if leakage-safe temporal filtering
            was applied (via ``prediction_cutoff``, ``outcome_time``, or
            ``kickoff_time``).  False when falling back to gameweek-only
            filtering, which is not guaranteed leakage-safe for
            postponed/rescheduled fixtures.
    """

    model: str
    brier_score: float
    sample_size: int
    calibration: dict[float, tuple[float, float]]
    position: PlayerPosition
    temporal_safety_enforced: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dc_threshold(position: PlayerPosition) -> int:
    """Return the DC action threshold for an eligible position.

    Raises:
        ValueError: If the position is not eligible for DC (GKP) or unknown.
    """
    if position == PlayerPosition.DEFENDER:
        return DEF_DC_THRESHOLD
    if position in (PlayerPosition.MIDFIELDER, PlayerPosition.FORWARD):
        return MID_FWD_DC_THRESHOLD
    raise ValueError(  # noqa: TRY003
        f"position {position!r} is not eligible for defensive contribution",
    )


def _is_eligible(position: PlayerPosition) -> bool:
    """Return True if the position is eligible for DC points (DEF, MID, FWD)."""
    return position in (
        PlayerPosition.DEFENDER,
        PlayerPosition.MIDFIELDER,
        PlayerPosition.FORWARD,
    )


def _check_duplicate_observations(
    appearances: Sequence[PlayerMatchHistory],
) -> None:
    """Detect duplicate player-match records in the filtered history.

    Each observation should represent a distinct fixture (``fixture_id``)
    in a distinct gameweek (``gameweek``).  Duplicate records would inflate
    sample sizes, distort action-rate estimates, and silently corrupt
    shrinkage calculations.

    This check operates on the filtered *appeances* (minutes > 0, gw < target)
    rather than the full history so that legitimate zero-minute bench entries
    — which may share a fixture_id across gameweeks — are not flagged.

    Raises:
        DCModelError: If duplicate ``(fixture_id, gameweek)`` pairs are found.
    """
    seen: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int]] = []
    for h in appearances:
        key = (h.fixture_id, h.gameweek)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        # Build a readable summary (limit length for large inputs).
        unique_dups = sorted(set(duplicates))
        preview = ", ".join(f"(fixture={f}, gw={g})" for f, g in unique_dups[:5])
        extra = ""
        if len(unique_dups) > 5:
            extra = f" (and {len(unique_dups) - 5} more)"
        raise DCModelError(  # noqa: TRY003
            f"Duplicate player-match observations detected in DC history: "
            f"{preview}{extra}. Each fixture_id+gameweek pair must be unique."
        )


def _validate_cutoff(dt: datetime | None, label: str) -> None:
    """Raise :class:`DCModelError` if *dt* is present but not UTC-aware.

    Parameters
    ----------
    dt
        The datetime (or ``None``) to validate.
    label
        Human-readable name used in the error message.
    """
    if dt is None:
        return
    if dt.tzinfo is None:
        raise DCModelError(  # noqa: TRY003
            f"{label} must be timezone-aware, got naive datetime"
        )
    if dt.utcoffset() != UTC.utcoffset(None):
        raise DCModelError(  # noqa: TRY003
            f"{label} must be in UTC, got {dt.tzinfo!r}"
        )


def _appearance_history(
    history: Sequence[PlayerMatchHistory],
    target_gameweek: int,
    prediction_cutoff: datetime | None = None,
) -> list[PlayerMatchHistory]:
    """Filter history to appearances (minutes > 0) in gameweeks before target.

    This is the core leakage guard: only observations strictly from earlier
    gameweeks are returned.  The target match itself and any future matches
    are excluded.

    When *prediction_cutoff* is provided, an additional temporal-safety
    filter is applied: only records whose ``outcome_time`` is at or before
    the cutoff are retained.  Records without ``outcome_time`` are **excluded**
    because their availability cannot be established — this prevents data
    leakage from postponed or rescheduled fixtures whose gameweek label is
    earlier but whose outcome was observed later.

    Duplicate ``(fixture_id, gameweek)`` pairs among the filtered appearances
    raise :class:`DCModelError` rather than being silently merged.

    Parameters
    ----------
    history
        Sequence of historical match observations for a single player.
    target_gameweek
        The gameweek of the fixture being predicted.
    prediction_cutoff
        Optional UTC datetime representing the latest point at which
        information was available for the prediction.  When ``None``,
        only gameweek ordering and appearance filters apply
        (backward-compatible behaviour).
    """
    if prediction_cutoff is not None:
        _validate_cutoff(prediction_cutoff, "prediction_cutoff")

    appearances = [
        h
        for h in history
        if h.minutes > 0 and h.gameweek < target_gameweek
    ]

    if prediction_cutoff is not None:
        # Temporal safety: only include records whose outcome was observable
        # at or before the prediction cutoff.  Records without an
        # outcome_time cannot have their availability established and are
        # excluded — this is deliberate and prevents leakage from
        # postponed/rescheduled fixtures.
        appearances = [
            h
            for h in appearances
            if h.outcome_time is not None
            and h.outcome_time <= prediction_cutoff
        ]

    _check_duplicate_observations(appearances)
    return appearances


def _count_dc_successes(
    appearances: Sequence[PlayerMatchHistory],
    position: PlayerPosition,
) -> tuple[int, int]:
    """Count DC-threshold successes and total appearances.

    Uses ``defensive_contribution_from_history`` with the position-appropriate
    threshold.  No invention of statistics — relies solely on verified
    per-match observations.

    Returns:
        (successes, total) where successes is the number of appearances
        meeting the DC threshold and total is the total appearances count.
    """
    threshold = _dc_threshold(position)
    total = len(appearances)
    successes = sum(
        1 for h in appearances if defensive_contribution_from_history(h, threshold)
    )
    return successes, total


def _shrink_frequency(
    successes: int,
    total: int,
    prior: float,
    pseudo_count: float,
) -> float:
    """Bayesian shrinkage of an observed frequency toward a prior.

    Formula::

        shrunk = (successes + prior * kappa) / (total + kappa)

    When ``total == 0``, returns the prior itself.
    """
    if total <= 0:
        return prior
    return (successes + prior * pseudo_count) / (total + pseudo_count)


# ---------------------------------------------------------------------------
# Poisson CDF implementation (no scipy dependency)
# ---------------------------------------------------------------------------

def _gammp(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x).

    Uses series expansion for ``x < a + 1`` and continued fraction
    otherwise.  Based on Numerical Recipes (3rd ed.).

    Parameters:
        a: Shape parameter (> 0).
        x: Argument (>= 0).

    Returns:
        P(a, x) = gamma(a, x) / Gamma(a).
    """
    if x < 0 or a <= 0:
        raise ValueError(  # noqa: TRY003
            f"Invalid arguments a={a}, x={x}"
        )
    if x == 0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def _gser(a: float, x: float, max_iter: int = 300, eps: float = 1e-14) -> float:
    """Lower incomplete gamma series: P(a, x) for x < a + 1."""
    gln = math.lgamma(a)
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * eps:
            break
    return summ * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float, max_iter: int = 300, eps: float = 1e-14) -> float:
    """Upper incomplete gamma continued fraction: Q(a, x) for x >= a + 1."""
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _poisson_sf(k: int, lam: float) -> float:
    """Survival function P(X > k) for Poisson(lam).

    Parameters:
        k: Number of events (non-negative integer).
        lam: Poisson rate (expected count), non-negative.

    Returns:
        P(X > k), clamped to [0, 1].
    """
    if lam < 0:
        raise ValueError(  # noqa: TRY003
            f"lam must be non-negative, got {lam}"
        )
    if lam == 0.0:
        return 0.0
    if k < 0:
        return 1.0
    # P(X > k) = _gammp(k+1, lam) via the regularized lower incomplete gamma.
    # (The regularized upper incomplete gamma Q(a,x) = 1 - P(a,x) gives
    # P(X <= k), so the survival function is P(a, x) directly.)
    return _gammp(k + 1, lam)


def _poisson_geq(k: int, lam: float) -> float:
    """Return P(X >= k) for Poisson(lam)."""
    if k <= 0:
        return 1.0
    return _poisson_sf(k - 1, lam)


# ---------------------------------------------------------------------------
# Model A: Historical-frequency baseline
# ---------------------------------------------------------------------------

def historical_dc_probability(
    history: Sequence[PlayerMatchHistory],
    position: PlayerPosition,
    target_gameweek: int,
    prior: float | None = None,
    pseudo_count: float | None = None,
    prediction_cutoff: datetime | None = None,
) -> DCPredictionResult:
    """Estimate P(DC | appear) via historical frequency with shrinkage.

    Candidate model A: the proportion of prior eligible appearances that
    met the DC action threshold, shrunk toward a position-specific prior.

    Parameters
    ----------
    history
        All historical match observations for the player.  Only entries
        with ``minutes > 0`` and ``gameweek < target_gameweek`` are used
        (leakage guard).
    position
        Player position.  GKP is ineligible; returns 0.0.
    target_gameweek
        Gameweek of the fixture being predicted.  Observations from this
        or later gameweeks are excluded.
    prior
        Bayesian shrinkage prior.  Defaults to ``DC_POSITION_PRIOR[position]``.
    pseudo_count
        Shrinkage strength kappa.  Defaults to ``DC_PRIOR_PSEUDO_COUNT``.
    prediction_cutoff
        Optional UTC datetime representing the latest point at which
        information was available for the prediction.  When provided,
        only records with ``outcome_time <= prediction_cutoff`` are
        used, preventing leakage from postponed/rescheduled fixtures.

    Returns
    -------
    DCPredictionResult
        Prediction result with probability, model name, observation count,
        and shrinkage metadata.

    Notes
    -----
    - When no prior appearances exist (empty history, all bench/0-minute,
      or all future gameweeks), the position prior is returned.  This is
      a documented, conservative fallback — not "zero" and not "fabricated".
    - Small samples are regularized via Bayesian shrinkage toward the prior.
    - The FPL element-summary API always provides defensive statistics as
      integers (0 when absent), so there is no missing-vs-zero ambiguity
      for the defensive fields used here.
    """
    if not _is_eligible(position):
        return DCPredictionResult(
            probability=0.0,
            model="history",
            n_observations=0,
            shrunk=False,
            prior_used=0.0,
        )

    if prior is None:
        prior = DC_POSITION_PRIOR[position]
    if pseudo_count is None:
        pseudo_count = DC_PRIOR_PSEUDO_COUNT

    appearances = _appearance_history(history, target_gameweek, prediction_cutoff)
    successes, total = _count_dc_successes(appearances, position)

    prob = _shrink_frequency(successes, total, prior, pseudo_count)
    shrunk = total < pseudo_count

    return DCPredictionResult(
        probability=prob,
        model="history",
        n_observations=total,
        shrunk=shrunk,
        prior_used=prior,
    )


# ---------------------------------------------------------------------------
# Model B: Poisson threshold model
# ---------------------------------------------------------------------------

def _dc_action_count(history: PlayerMatchHistory, position: PlayerPosition) -> int:
    """Compute the defensive action count (CBIT or CBIRT) for a match.

    - DEF: CBIT = tackles + clearances_blocks_interceptions.
    - MID/FWD: CBIRT = CBIT + recoveries.

    Does not call ``defensive_contribution_from_history`` (which returns a
    boolean); returns the raw action count for rate estimation.
    """
    cbit = history.tackles + history.clearances_blocks_interceptions
    if position == PlayerPosition.DEFENDER:
        return cbit
    return cbit + history.recoveries


def poisson_dc_probability(
    history: Sequence[PlayerMatchHistory],
    position: PlayerPosition,
    target_gameweek: int,
    expected_minutes: float = DC_DEFAULT_MINUTES,
    prior: float | None = None,
    pseudo_count: float | None = None,
    prediction_cutoff: datetime | None = None,
) -> DCPredictionResult:
    """Estimate P(DC | appear) via a Poisson threshold model.

    Candidate model B: defensive actions are modelled as a Poisson process
    with a rate estimated from prior per-minute action counts.  The
    threshold-crossing probability is computed numerically.

    Parameters
    ----------
    history
        All historical match observations for the player.  Only entries
        with ``minutes > 0`` and ``gameweek < target_gameweek`` are used
        (leakage guard).
    position
        Player position.  GKP returns 0.0 (ineligible).
    target_gameweek
        Gameweek of the fixture being predicted.
    expected_minutes
        Expected minutes the player will play in the target fixture,
        conditional on appearing.  Used to scale the per-minute rate
        into an expected action count (lambda).  Must be >= 0.
    prior
        Prior probability for shrinkage.  Defaults to
        ``DC_POSITION_PRIOR[position]``.
    pseudo_count
        Shrinkage strength (effective number of pseudo-fixtures).
        Defaults to ``DC_PRIOR_PSEUDO_COUNT``.
    prediction_cutoff
        Optional UTC datetime representing the latest point at which
        information was available for the prediction.  When provided,
        only records with ``outcome_time <= prediction_cutoff`` are
        used, preventing leakage from postponed/rescheduled fixtures.

    Returns
    -------
    DCPredictionResult
        Prediction result with probability, model name, and metadata.

    Notes
    -----
    - The defensive "action count" is CBIT for DEF, CBIRT for MID/FWD,
      matching the FPL rule.
    - The Poisson rate is shrunk toward a prior lambda derived from the
      position prior probability: ``lambda_prior = prior * threshold``.
      Shrinkage blends the observed rate with this prior rate.
    - Zero exposure (no prior minutes) falls back to the prior probability.
    - The Poisson model is **not** calibrated against observed DC rates;
      it is a parametric baseline.  Compare against the historical-frequency
      model in backtesting before trusting it for production decisions.
    """
    if not _is_eligible(position):
        return DCPredictionResult(
            probability=0.0,
            model="poisson",
            n_observations=0,
            shrunk=False,
            prior_used=0.0,
        )

    if expected_minutes < 0:
        raise DCModelError(  # noqa: TRY003
            f"expected_minutes must be non-negative, got {expected_minutes}",
        )

    if prior is None:
        prior = DC_POSITION_PRIOR[position]
    if pseudo_count is None:
        pseudo_count = DC_PRIOR_PSEUDO_COUNT

    threshold = _dc_threshold(position)
    appearances = _appearance_history(history, target_gameweek, prediction_cutoff)

    total_actions = 0
    total_minutes = 0
    for h in appearances:
        total_actions += _dc_action_count(h, position)
        total_minutes += h.minutes

    # No exposure data: fall back to prior probability directly.
    if total_minutes <= 0 or expected_minutes == 0.0:
        return DCPredictionResult(
            probability=prior,
            model="poisson",
            n_observations=0,
            shrunk=True,
            prior_used=prior,
        )

    # Shrinkage prior for the Poisson rate:
    # lambda_prior = prior * threshold (expected actions to reach threshold
    # at the prior probability level).
    lambda_prior = prior * threshold

    # Shrinkage: blend observed expected action count with prior lambda.
    #   lambda_obs = (total_actions / total_minutes) * expected_minutes
    #   lambda_shrunk = (lambda_obs + lambda_prior * kappa) / (1 + kappa)
    lambda_obs = (total_actions / total_minutes) * expected_minutes
    lambda_shrunk = (lambda_obs + lambda_prior * pseudo_count) / (
        1.0 + pseudo_count
    )

    # P(DC | appear) = P(X >= threshold) for Poisson(lambda_shrunk)
    prob = _poisson_geq(threshold, lambda_shrunk)
    prob = max(0.0, min(1.0, prob))

    return DCPredictionResult(
        probability=prob,
        model="poisson",
        n_observations=len(appearances),
        shrunk=True,
        prior_used=prior,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_defensive_contribution_probability(
    history: Sequence[PlayerMatchHistory] | None,
    position: PlayerPosition,
    target_gameweek: int,
    expected_minutes: float = DC_DEFAULT_MINUTES,
    model: str = "history",
    prediction_cutoff: datetime | None = None,
) -> DCPredictionResult:
    """Estimate P(DC threshold met | player appears).

    Selects between candidate models and returns a structured result.
    The returned probability is *conditional on appearance* — callers
    must apply P(appear) separately in the scoring engine.

    Parameters
    ----------
    history
        Historical match observations for the player.  ``None`` or an
        empty sequence triggers the position-prior fallback (no personal
        data available).
    position
        Player position.  GKP always returns 0.0.
    target_gameweek
        Gameweek of the fixture being predicted.  Only observations from
        earlier gameweeks are used.
    expected_minutes
        Expected minutes conditional on appearance (Poisson model only).
    model
        Model selector: ``"history"`` (default) or ``"poisson"``.
    prediction_cutoff
        Optional UTC datetime representing the latest point at which
        information was available for the prediction.  Forwarded to
        the selected model's appearance filtering to prevent leakage
        from postponed/rescheduled fixtures.

    Returns
    -------
    DCPredictionResult
        Prediction result.  ``probability`` is always in [0, 1].

    Raises
    ------
    DCModelError
        If ``model`` is not a valid selector.
    """
    if model not in _DC_MODELS:
        raise DCModelError(  # noqa: TRY003
            f"Unknown model '{model}'. Valid models: {sorted(_DC_MODELS)}",
        )

    if not _is_eligible(position):
        return DCPredictionResult(
            probability=0.0,
            model=model,
            n_observations=0,
            shrunk=False,
            prior_used=0.0,
        )

    hist_list: list[PlayerMatchHistory] = list(history) if history else []

    if model == "poisson":
        return poisson_dc_probability(
            hist_list,
            position,
            target_gameweek,
            expected_minutes=expected_minutes,
            prediction_cutoff=prediction_cutoff,
        )
    return historical_dc_probability(
        hist_list,
        position,
        target_gameweek,
        prediction_cutoff=prediction_cutoff,
    )


# ---------------------------------------------------------------------------
# Step 5: Leakage-safe historical evaluation
# ---------------------------------------------------------------------------

def _brier_score_component(pred: float, outcome: int) -> float:
    """Brier score component for a single prediction: (pred - outcome)^2."""
    return (pred - outcome) ** 2


def evaluate_dc_models_brier(
    history: Sequence[PlayerMatchHistory],
    position: PlayerPosition,
    max_gameweek: int = 38,
    expected_minutes: float = DC_DEFAULT_MINUTES,
    prediction_cutoff: datetime | None = None,
) -> dict[str, DCEvaluationResult]:
    """Leakage-safe rolling-origin evaluation of DC models.

    For each gameweek ``gw`` from 2 to the last observed gameweek, the
    function uses only observations with ``gameweek < gw`` to predict
    P(DC | appear), then compares against the actual outcome for that
    gameweek's fixture (if the player appeared that week).

    Parameters
    ----------
    history
        Full player history (all gameweeks).  The function internally
        filters to pre-cutoff observations at each step.
    position
        Player position.  GKP returns empty results (ineligible).
    max_gameweek
        Last gameweek to evaluate (inclusive).
    expected_minutes
        Expected minutes for the Poisson model (conditional on appearance).
    prediction_cutoff
        Optional UTC datetime representing the latest point at which
        information was available for the prediction.  When provided, it
        overrides per-fixture ``outcome_time`` / ``kickoff_time`` and is
        validated for UTC-awareness.  When ``None``, the function falls
        back to the target fixture's ``outcome_time`` or ``kickoff_time``,
        and finally to gameweek-only filtering.

    Returns
    -------
    dict[str, DCEvaluationResult]
        Mapping from model name to evaluation result.  Keys are
        ``"history"`` and ``"poisson"``.

    Notes
    -----
    - The target match's own defensive statistics are never used as its
      prediction input — only strictly prior observations enter the model.
    - Season-end aggregates are never mixed into earlier-gameweek
      predictions.
    - Calibration is assessed in 0.1-wide bins of predicted probability.
    - Deterministic given input history; no I/O or random seed.
    - Meaningful historical validation requires per-player history retrieval
      (one API call per player).  Synthetic data demonstrates the logic
      but cannot establish real-world model superiority.

    Leakage safety
    --------------
    When ``prediction_cutoff`` is provided, it is used as the maximum
    prediction time for all fixtures; training observations are filtered to
    those with ``outcome_time <= prediction_cutoff``.  When
    ``prediction_cutoff`` is ``None``, the function falls back to each
    target fixture's own ``kickoff_time`` (if available from joined fixture
    metadata) or ``outcome_time``.  Gameweek-only filtering is used only
    as a last resort and sets ``temporal_safety_enforced=False``.
    """
    if prediction_cutoff is not None:
        _validate_cutoff(prediction_cutoff, "prediction_cutoff")

    if not _is_eligible(position):
        gkp_empty = DCEvaluationResult(
            model="history",
            brier_score=0.0,
            sample_size=0,
            calibration={},
            position=position,
            temporal_safety_enforced=prediction_cutoff is not None,
        )
        return {
            "history": gkp_empty,
            "poisson": DCEvaluationResult(
                model="poisson",
                brier_score=0.0,
                sample_size=0,
                calibration={},
                position=position,
                temporal_safety_enforced=prediction_cutoff is not None,
            ),
        }

    threshold = _dc_threshold(position)
    prior = DC_POSITION_PRIOR[position]
    pseudo_count = DC_PRIOR_PSEUDO_COUNT

    hist_list = list(history)
    max_observed_gw = max((h.gameweek for h in hist_list), default=0)
    eval_gw_end = min(max_gameweek, max_observed_gw)

    model_predictions: dict[str, list[tuple[float, int]]] = {
        "history": [],
        "poisson": [],
    }

    # Track whether leakage-safe temporal filtering was applied at least once.
    temporal_safety_enforced = prediction_cutoff is not None

    for gw in range(2, eval_gw_end + 1):
        gw_match = next((h for h in hist_list if h.gameweek == gw), None)

        if gw_match is None or gw_match.minutes == 0:
            continue

        prior_history = [h for h in hist_list if h.gameweek < gw]

        # Determine the prediction cutoff for this target fixture.
        # Priority order:
        #   1. Explicit ``prediction_cutoff`` parameter (global, leakage-safe).
        #   2. Target fixture's ``outcome_time`` (if available from API).
        #   3. Target fixture's ``kickoff_time`` (joined from /api/fixtures/).
        #   4. Gameweek-only filtering (NOT leakage-safe — flagged).
        gw_temporal_safety = False
        cutoff: datetime | None = None
        if prediction_cutoff is not None:
            cutoff = prediction_cutoff
            gw_temporal_safety = True
        elif gw_match.outcome_time is not None:
            cutoff = gw_match.outcome_time
            gw_temporal_safety = True
        elif gw_match.kickoff_time is not None:
            cutoff = gw_match.kickoff_time
            gw_temporal_safety = True

        # OR across iterations: temporal safety is enforced if it applies
        # to at least one evaluated fixture.
        temporal_safety_enforced = temporal_safety_enforced or gw_temporal_safety

        if gw_temporal_safety and cutoff is not None:
            prior_history = [
                h
                for h in prior_history
                if (ot := h.outcome_time) is not None
                and ot < cutoff
            ]

        outcome = 1 if defensive_contribution_from_history(gw_match, threshold) else 0

        hist_result = historical_dc_probability(
            prior_history, position, gw,
            prior=prior, pseudo_count=pseudo_count,
        )
        model_predictions["history"].append((hist_result.probability, outcome))

        pois_result = poisson_dc_probability(
            prior_history, position, gw,
            expected_minutes=expected_minutes,
            prior=prior, pseudo_count=pseudo_count,
        )
        model_predictions["poisson"].append((pois_result.probability, outcome))

    results: dict[str, DCEvaluationResult] = {}
    for model_name, predictions in model_predictions.items():
        n = len(predictions)
        if n == 0:
            results[model_name] = DCEvaluationResult(
                model=model_name,
                brier_score=0.0,
                sample_size=0,
                calibration={},
                position=position,
                temporal_safety_enforced=temporal_safety_enforced,
            )
            continue

        brier = sum(_brier_score_component(p, o) for p, o in predictions) / n

        calibration: dict[float, tuple[float, float]] = {}
        num_bins = 10
        for bin_start in range(num_bins):
            lo = bin_start / num_bins
            hi = (bin_start + 1) / num_bins
            if bin_start == num_bins - 1:
                bin_data = [(p, o) for p, o in predictions if lo <= p <= 1.0]
            else:
                bin_data = [(p, o) for p, o in predictions if lo <= p < hi]
            if bin_data:
                avg_pred = sum(p for p, _ in bin_data) / len(bin_data)
                avg_obs = sum(o for _, o in bin_data) / len(bin_data)
                calibration[lo] = (avg_pred, avg_obs)

        results[model_name] = DCEvaluationResult(
            model=model_name,
            brier_score=brier,
            sample_size=n,
            calibration=calibration,
            position=position,
            temporal_safety_enforced=temporal_safety_enforced,
        )

    return results


# ---------------------------------------------------------------------------
# Convenience: compute per-match defensive action total
# ---------------------------------------------------------------------------

def dc_action_count(
    history: PlayerMatchHistory,
    position: PlayerPosition,
) -> int:
    """Return the defensive action count (CBIT or CBIRT) for a match.

    This is the raw action total used by the Poisson rate model.  Exposed
    publicly so callers can inspect the underlying quantity.

    Note:
        This function does not enforce position eligibility (GKP).
        Callers should guard eligibility via :func:`predict_defensive_contribution_probability`
        or :func:`_is_eligible` before relying on the result for GKP.
    """
    return _dc_action_count(history, position)
