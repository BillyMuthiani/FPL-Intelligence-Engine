# Prediction, Simulation, and FPL Decisions

## Prediction Principles

* Establish a simple, measurable baseline before introducing complex models.
* Keep predictions separate from observed outcomes.
* Clearly define the prediction target, prediction horizon, and available input features.
* Treat expected points as an estimate, not a guarantee.
* Where feasible, communicate uncertainty as well as the central estimate.
* Compare complex models against appropriate baselines.

## Prevent Data Leakage

* Never use information that would not have been available at the historical prediction date.
* Use chronological or rolling-origin evaluation for time-dependent predictions.
* Fit preprocessing, feature selection, and model parameters using training data only.
* Do not randomly split time-dependent observations when that creates unrealistic evaluation.
* Audit every feature for possible future-information leakage.

## Model Evaluation

* Use metrics appropriate to the target and decision.
* Compare predictions with simple baselines.
* Inspect errors across players, positions, gameweeks, and relevant subgroups.
* Keep a reproducible record of model configuration, data period, and evaluation results.
* Do not report improved performance without measured evidence.

## Simulation

* Model uncertainty explicitly where the data supports it.
* Keep assumptions and probability distributions documented.
* Validate that probabilities are sensible and outputs are reproducible when a random seed is supplied.
* Do not present simulated outcomes as certain forecasts.

## Squad and Transfer Optimization

* Respect actual FPL squad constraints, budgets, positions, club limits, and applicable rules.
* Consider transfer costs, free transfers, points hits, and chip availability where represented by the data.
* Compare the proposed action against keeping the current squad.
* Consider multiple gameweeks when the prediction horizon supports it.
* Do not recommend an action solely because one player has a higher expected-points estimate.
* Distinguish model outputs from the final decision recommendation.

## Responsible Interpretation

* State important assumptions and limitations.
* Do not fabricate injury status, player availability, prices, fixtures, or model performance.
* Prefer robust decisions over false precision.
* Keep model code independent of presentation logic.
