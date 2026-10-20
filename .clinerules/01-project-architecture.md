# Project Architecture and Documentation

## Source of Truth

* Read `README.md`, `docs/PROJECT_SPEC.md`, `pyproject.toml`, and the relevant source files before making architectural decisions.
* Treat `docs/PROJECT_SPEC.md` as the intended product specification, not proof that a feature already exists.
* Treat the actual code and tests as evidence of current implementation.
* If the specification, README, rules, and implementation disagree, report the discrepancy instead of silently choosing one.
* Never claim a feature is implemented merely because it appears in documentation or the roadmap.

## Architecture

* Keep the core FPL intelligence engine independent of the user interface.
* Separate data ingestion, validation, feature engineering, prediction, simulation, optimization, and presentation into appropriate modules.
* Avoid circular imports, duplicated business logic, oversized modules, and unnecessary abstractions.
* Keep configuration centralized and secrets out of source control.
* Prefer simple, maintainable designs over premature complexity.
* Preserve existing public APIs unless a change is necessary and approved.

## Product Direction

The project aims to provide reliable Fantasy Premier League decision support, including:

* Historical and current FPL data ingestion.
* Data validation, normalization, and feature engineering.
* Player and fixture analysis.
* Expected-points prediction.
* Probabilistic simulation and uncertainty estimates.
* Squad selection, transfer planning, and chip decisions.
* Historical backtesting.
* A Streamlit interface.
* Future screenshot-based squad recognition and visual squad generation.

These are intended capabilities, not assumptions about current implementation.

## Decision Principles

* Recommendations must consider uncertainty, opportunity cost, and the option to make no transfer.
* Separate raw data, derived features, predictions, and final recommendations.
* Avoid coupling core calculations to Streamlit.
* Do not build future roadmap features prematurely when foundational work remains incomplete.

## Before Architectural Changes

1. Inspect the relevant code and tests.
2. Explain the problem and the proposed approach.
3. Identify affected modules and compatibility risks.
4. Make the smallest coherent change.
5. Update relevant documentation and tests.
6. Report verified results and remaining limitations.
