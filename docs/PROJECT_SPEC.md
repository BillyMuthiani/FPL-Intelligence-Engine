# FPL Intelligence Engine

## Master Project Specification

You are the primary software engineering agent responsible for building a production-quality Fantasy Premier League (FPL) analytics and decision-support platform.

The project is called **FPL Intelligence Engine**.

The goal is NOT to build another basic FPL statistics dashboard.

The goal is to build an intelligent, data-driven decision-support system that helps an FPL manager answer:

> **"Given my exact squad, the current state of the league, upcoming fixtures, player performance, uncertainty, available transfers, budget and chips, what are my best options and what are the likely consequences of each decision?"**

The system must be modular, testable, explainable and capable of evolving from a local research project into a deployed Streamlit application.

---

# 1. CORE PRODUCT VISION

The platform should combine:

* Official FPL data
* Historical Premier League data
* Advanced football statistics
* Fixture analysis
* Player availability information
* Machine-learning predictions
* Probabilistic simulation
* Squad optimization
* Transfer optimization
* Captaincy analysis
* Chip strategy
* Screenshot-based squad recognition
* Personalized squad analysis
* Explainable recommendations
* Visualized recommended squads

The system should NEVER simply output:

> "Buy Player X."

It should explain:

* Why Player X is recommended
* What assumptions produced the recommendation
* How much improvement is expected
* Over what time horizon
* What risks exist
* What alternative decisions are available
* What happens if the manager does nothing

---

# 2. IMPORTANT ARCHITECTURAL PRINCIPLE

Separate the system into three conceptual layers:

## A. Intelligence Engine

This is the core of the project.

It contains:

* Data ingestion
* Data validation
* Feature engineering
* Prediction models
* Probabilistic models
* Monte Carlo simulation
* Squad analysis
* Optimization
* Decision evaluation
* Backtesting

The Intelligence Engine must NOT depend on Streamlit.

It should be usable independently through Python APIs.

Example:

```python
from fpl_engine import DecisionEngine

engine = DecisionEngine(...)

recommendations = engine.analyze_squad(...)
```

## B. Application Layer

Initially:

* Streamlit

Later this could become:

* FastAPI backend
* Web frontend
* Mobile client

The application layer consumes the Intelligence Engine.

## C. Presentation Layer

Responsible for:

* Tables
* Charts
* Squad visualizations
* Pitch layouts
* Recommendation cards
* Explanations
* Generated squad images

Do not mix UI logic with prediction/optimization logic.

---

# 3. HIGH-LEVEL ARCHITECTURE

Implement toward this architecture:

```text
DATA SOURCES
    |
    v
DATA INGESTION
    |
    v
RAW DATA
    |
    v
VALIDATION / NORMALIZATION
    |
    v
PROCESSED DATA
    |
    v
FEATURE ENGINEERING
    |
    +-----------------------------+
    |                             |
    v                             v
PLAYER FEATURES              FIXTURE FEATURES
    |                             |
    +-------------+---------------+
                  |
                  v
           PREDICTION ENGINE
                  |
       +----------+----------+
       |          |          |
       v          v          v
    Minutes     Goals      Assists
       |          |          |
       +----------+----------+
                  |
                  v
        FPL POINTS MODEL
                  |
                  v
        MONTE CARLO ENGINE
                  |
                  v
        SQUAD / TRANSFER
           OPTIMIZATION
                  |
                  v
        DECISION ENGINE
                  |
       +----------+----------+
       |          |          |
       v          v          v
   Transfers   Captain      Chips
       |
       v
  SQUAD ANALYZER
       |
       v
STREAMLIT / API
       |
       v
USER
```

---

# 4. DATA SOURCES

Design the ingestion system so sources are replaceable.

Do NOT tightly couple models to scraping code.

Primary sources should include:

## Official FPL data

Use official FPL endpoints wherever practical for:

* Players
* Teams
* Positions
* Prices
* Ownership
* FPL points
* Gameweek history
* Fixtures
* Transfers
* Minutes
* Goals
* Assists
* Bonus
* BPS
* FPL-specific statistics

The exact endpoint implementation must be isolated inside the data ingestion layer.

## Historical football data

Potential sources:

* Football-Data.co.uk
* Other legally usable historical datasets

Use for:

* Historical match results
* Goals
* Home/away
* Team performance
* Long-term team strength

## Advanced football data

Potential sources include:

* Understat-derived datasets
* Other legally usable football datasets/APIs

Potential features:

* xG
* xA
* shots
* shots on target
* big chances
* shot locations
* chance creation
* set pieces

Do not assume every source will always be available.

Create provider interfaces/adapters.

Example:

```python
class PlayerDataProvider(Protocol):
    def get_players(self) -> pd.DataFrame:
        ...

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        ...
```

This allows providers to be replaced without changing downstream models.

---

# 5. DATA MODEL

Create canonical internal entities.

At minimum:

```text
Player
Team
Fixture
Gameweek
PlayerGameweekStats
PlayerSeasonStats
TeamStrength
PlayerAvailability
Prediction
SimulationResult
Squad
Transfer
Recommendation
```

Use stable IDs whenever possible.

Do not use player names as primary identifiers.

Names can change.

---

# 6. FEATURE ENGINEERING

The feature system must be modular.

Features should include multiple time horizons.

Examples:

## Form

* Last 1 GW
* Last 3 GW
* Last 5 GW
* Last 10 GW
* Season average

## Attacking

* Goals
* xG
* xG/90
* Assists
* xA
* xA/90
* Shots
* Shots/90
* Shots on target
* Big chances
* Big chances/90
* Key passes
* Chance creation

## Defensive

* Clean sheets
* Goals conceded
* Defensive contributions
* Tackles
* Interceptions
* Clearances
* Blocks
* Defensive actions

## Role

* Position
* Starts
* Expected minutes
* Start probability
* Rotation probability
* Set-piece involvement
* Penalty involvement
* Corner involvement
* Free-kick involvement

## Context

* Home/away
* Opponent strength
* Team attacking strength
* Team defensive strength
* Fixture difficulty
* Fixture congestion

## FPL economics

* Current price
* Price change
* Ownership
* Transfers in
* Transfers out
* Points per £m
* Projected value

---

# 7. MINUTES MODEL

Minutes are extremely important.

Build a dedicated minutes/start probability model.

Output should include:

```text
start_probability
expected_minutes
probability_60_plus
probability_90
bench_probability
```

Potential inputs:

* Recent starts
* Recent minutes
* Substitution patterns
* Injury status
* Suspension
* Team rotation
* Fixture congestion
* European matches
* International duty
* Manager selection patterns

Do not assume a player will play 90 minutes simply because they are a good player.

---

# 8. PLAYER PREDICTION MODELS

Do not initially create one black-box model for total FPL points.

Prefer separate prediction components.

At minimum:

```text
Minutes model
Goals model
Assists model
Clean-sheet model
Card model
Bonus model
Defensive contribution model
```

These feed into an expected FPL points model.

Models should initially have strong statistical baselines.

Then compare:

* Baseline
* Linear models
* Random Forest
* XGBoost
* LightGBM

Do not use complex ML merely because it is available.

Choose models based on backtesting.

---

# 9. EXPECTED FPL POINTS

Create a transparent points calculation layer.

The system should convert predicted football outcomes into expected FPL points using the current FPL scoring rules.

Do not hardcode scoring assumptions throughout the project.

Create one centralized scoring configuration.

Example:

```python
class FPLScoringRules:
    ...
```

This allows scoring rules to be updated without rewriting models.

---

# 10. MONTE CARLO SIMULATION

This is a major component.

Do not only output:

```text
Player X = 7.4 expected points
```

Simulate thousands of possible outcomes.

For example:

```text
10,000 simulations
```

Output:

* Mean points
* Median points
* Standard deviation
* 10th percentile
* 25th percentile
* 75th percentile
* 90th percentile
* Probability of blank
* Probability of 6+
* Probability of 10+
* Probability of haul
* Probability of return

Simulation parameters must come from the prediction models.

Make the simulator deterministic when a random seed is provided.

---

# 11. SQUAD ANALYZER

This is a central feature.

The system must analyze an EXISTING FPL squad.

Input:

```text
Current squad
Bank
Free transfers
Transfer penalty
Available chips
Current Gameweek
```

Analyze:

## Strengths

Examples:

* Strong captaincy options
* Strong midfield
* Good fixture coverage
* Good bench
* Strong value

## Weaknesses

Examples:

* Poor forward depth
* Fixture concentration
* Rotation risk
* Weak bench
* Excessive money in one position
* Missing premium captaincy option

## Risks

Examples:

* Injury
* Suspension
* Rotation
* Fixture swing
* High variance

Output an overall squad-health score, but do not let that score replace detailed analysis.

---

# 12. SCREENSHOT SQUAD IMPORT

Users should be able to upload an FPL screenshot.

Pipeline:

```text
Screenshot
    |
    v
Image preprocessing
    |
    v
Position / player-region detection
    |
    v
OCR / visual recognition
    |
    v
Player-name matching
    |
    v
FPL player database
    |
    v
Confidence validation
```

Use fuzzy matching where appropriate.

Never silently accept low-confidence matches.

If confidence is low, expose a correction mechanism.

Example:

```text
Detected:
"Isak"

Confidence: 98%

Detected:
"Gabriel"

Confidence: 62%

Please confirm:
Gabriel Magalhaes
Gabriel Martinelli
```

The screenshot importer must be isolated from the core prediction engine.

---

# 13. TEAM ID IMPORT

Where supported, allow users to enter their FPL Team ID.

This can complement screenshot recognition.

Ideal workflow:

```text
Screenshot
+
Optional FPL Team ID
```

The Team ID can validate the screenshot-derived squad.

---

# 14. TRANSFER OPTIMIZER

The optimizer must consider:

* Current squad
* Bank
* Free transfers
* Transfer penalties
* Player prices
* Position requirements
* Club limits
* Expected points
* Future fixtures
* Minutes
* Risk
* Multi-gameweek horizon

Do NOT optimize only for the next Gameweek.

Support horizons such as:

```text
1 GW
3 GW
5 GW
8 GW
```

The objective function should be configurable.

---

# 15. "DO NOTHING" MUST BE AN OPTION

Always compare:

```text
Save transfer
```

against:

```text
Transfer A → B
Transfer A → C
...
```

The system must be capable of telling the manager:

> Save the transfer.

Do not force a recommendation merely to produce an action.

---

# 16. TRANSFER COST ANALYSIS

For every proposed transfer calculate:

```text
Expected gain
- transfer penalty
= net expected gain
```

Also show:

```text
1 GW gain
3 GW gain
5 GW gain
8 GW gain
```

This allows proper -4 decision analysis.

---

# 17. CAPTAINCY ENGINE

Create a dedicated captain model.

Rank candidates using:

* Expected points
* Haul probability
* Blank probability
* Fixture
* Minutes
* Penalty involvement
* Set pieces
* Ownership
* Variance

Support profiles:

```text
Safe
Balanced
Aggressive
```

Captain recommendations must include reasoning.

---

# 18. CHIP OPTIMIZER

Support:

* Wildcard
* Free Hit
* Bench Boost
* Triple Captain

Analyze future Gameweeks and identify potentially optimal windows.

Do not simply recommend a chip based on one Gameweek.

Simulate alternative chip timing.

---

# 19. "WHAT IF?" ENGINE

Allow managers to simulate decisions.

Examples:

```text
What if I sell Player A?

What if I captain Player B?

What if I take a -4?

What if I wildcard?

What if I save my transfer?

What if I replace Defender A with Defender B?
```

Compare scenarios using the same simulation framework.

---

# 20. FIXTURE INTELLIGENCE

Build our own fixture strength model.

Do not rely exclusively on FPL's FDR.

Calculate:

* Opponent attacking strength
* Opponent defensive strength
* Home advantage
* Team strength
* Recent performance
* Fixture congestion
* Expected goals environment

Detect:

* Fixture swings
* Target runs
* Avoid runs
* Blank Gameweeks
* Double Gameweeks

---

# 21. NEWS / AVAILABILITY LAYER

Keep news ingestion separate from the core model.

Potential information:

* Injuries
* Suspensions
* Press conferences
* Training updates
* Manager comments
* Expected lineups

Convert news into model variables where possible.

Example:

```text
Injury concern
    ↓
Start probability
    ↓
Expected minutes
    ↓
Expected points
    ↓
Transfer recommendation
```

Do not allow unverified news to directly overwrite model data.

Track source and confidence.

---

# 22. DIFFERENTIAL ENGINE

Identify players with:

* Low ownership
* High projected points
* Strong fixture runs
* Strong underlying statistics

But distinguish:

```text
Low-risk differential
High-upside differential
High-variance differential
```

---

# 23. VALUE ENGINE

Calculate:

```text
Expected points / £m
```

and other useful value metrics.

Identify:

* Underpriced players
* Poor-value players
* Price-efficient squads
* Value opportunities

---

# 24. BENCH OPTIMIZER

Determine:

* Starting XI
* Bench order
* Formation
* Expected bench points
* Bench rotation risk

Do not simply sort players by expected points because FPL formation constraints matter.

---

# 25. MINI-LEAGUE ANALYSIS

If manager league data is available, support:

* Squad comparison
* Ownership differences
* Captain differences
* Differential identification
* Rival strategy
* Rank protection
* Rank chasing

Do not simply tell users to copy rivals.

Use rival information as strategic context.

---

# 26. PERSONALIZED MANAGER PROFILE

Allow users to choose:

```text
Conservative
Balanced
Aggressive
Differential hunter
```

This should influence recommendation ranking and risk tolerance.

It must NOT distort the underlying predictions.

Separate:

```text
Prediction
```

from:

```text
Preference
```

---

# 27. RECOMMENDATION EXPLAINABILITY

Every recommendation must answer:

1. What should I do?
2. Why?
3. How much improvement is expected?
4. Over what period?
5. What are the risks?
6. What is the best alternative?
7. What happens if I do nothing?

Example:

```text
SELL PLAYER A → BUY PLAYER B

Why:
- Better 5-GW fixture run
- Higher expected minutes
- Higher xGI
- Better value

Projected impact:
GW1: +1.2
GW3: +4.7
GW5: +8.4

Risk:
Player B has moderate rotation uncertainty.

Alternative:
Player C provides lower upside but greater minutes security.
```

---

# 28. SQUAD VISUALIZATION / GENERATED IMAGE

After transfer optimization, generate a visual representation of the recommended squad.

Input:

```text
Current squad
Recommended transfers
Starting XI
Formation
Player positions
```

Output:

```text
FPL-style pitch image
```

Show:

* Player names
* Positions
* Starting XI
* Bench
* Captain
* Vice captain
* Changed players

The visualization layer must NOT make football decisions.

The Intelligence Engine decides the squad.

The visualization simply renders the result.

Where possible, support:

```text
Current squad image
Recommended squad image
```

and visually distinguish changed players.

---

# 29. STREAMLIT APPLICATION

The first UI should contain:

## Dashboard

* Current squad
* Squad health
* Projected points
* Main problems
* Recommended actions

## My Squad

* Pitch
* Starting XI
* Bench
* Player cards
* Fixture outlook

## Transfers

* Recommended transfers
* Multi-GW impact
* Transfer cost
* Alternatives
* Save-transfer option

## Captain

* Captain ranking
* Probability distributions
* Safe/balanced/aggressive choices

## Fixtures

* Fixture difficulty
* Fixture swings
* Target teams

## Players

* Search
* Filters
* Predictions
* Value
* Form
* Underlying stats

## Chips

* Chip planning
* Future Gameweek simulations

## What If?

* Scenario simulator

## Upload Squad

* Screenshot upload
* Recognition
* Validation
* Analysis
* Refined squad image

---

# 30. DATA STORAGE

Start simple but design for growth.

Possible initial stack:

```text
CSV / Parquet
+
SQLite
```

Later:

```text
PostgreSQL
```

Do not introduce a complicated database architecture prematurely.

Raw data and processed data should be separated.

---

# 31. MODEL EVALUATION

Backtesting is mandatory.

Use chronological validation.

Example:

```text
Train:
GW1 → GW10

Predict:
GW11

Train:
GW1 → GW11

Predict:
GW12
```

Never allow future information into historical predictions.

Evaluate:

* MAE
* RMSE
* Brier score
* Log loss where appropriate
* Calibration
* Rank correlation
* Expected-point error
* Captain accuracy
* Transfer recommendation performance

Compare against simple baselines.

Example:

```text
Naive average
FPL form
FPL points/game
Simple xG model
ML model
```

A complicated model is only justified if it beats simpler baselines.

---

# 32. DATA LEAKAGE

This is a critical requirement.

Features used to predict Gameweek N must only use information available BEFORE Gameweek N.

Examples of prohibited leakage:

* Future fixtures/results
* End-of-season totals
* Future player prices
* Future minutes
* Future injuries
* Future xG
* Future FPL points

Implement tests or validation utilities where practical to detect temporal leakage.

---

# 33. TESTING

Use pytest.

At minimum test:

* Data parsing
* Player matching
* Feature calculations
* FPL scoring
* Prediction outputs
* Simulation reproducibility
* Squad constraints
* Transfer constraints
* Club limits
* Position limits
* Budget constraints
* Historical split correctness

The optimizer should have explicit constraint tests.

---

# 34. CODE QUALITY

Use:

* Python type hints
* Ruff
* Pytest
* Clear docstrings where useful
* Small modules
* Dependency injection for data providers
* Configuration files rather than hardcoded values
* Logging
* Reproducible pipelines

Avoid:

* Giant Python files
* Giant Streamlit scripts
* Hardcoded player names
* Hardcoded team IDs
* Hardcoded current prices
* Hidden global state
* Scraping logic inside UI code

---

# 35. DEVELOPMENT METHODOLOGY

Do NOT attempt to build every feature immediately.

Build vertically in milestones.

Recommended order:

### Phase 1 — Foundation

* Repository
* Package structure
* Configuration
* Logging
* Tests
* Data models
* FPL provider

### Phase 2 — Data Pipeline

* Official FPL ingestion
* Historical data ingestion
* Normalization
* Storage
* Data validation

### Phase 3 — Baseline Analytics

* Player statistics
* Fixture strength
* Form
* Value
* Basic expected points

### Phase 4 — Prediction Engine

* Minutes
* Goals
* Assists
* Clean sheets
* FPL points

### Phase 5 — Monte Carlo

* Simulation
* Probability distributions
* Uncertainty

### Phase 6 — Squad Intelligence

* Squad analyzer
* Transfer optimizer
* Captain engine
* Bench optimizer

### Phase 7 — Backtesting

* Historical simulations
* Model comparison
* Calibration
* Evaluation reports

### Phase 8 — Streamlit

* Dashboard
* Players
* Transfers
* Captain
* Fixtures
* What-if

### Phase 9 — Screenshot Intelligence

* Screenshot upload
* OCR
* Player matching
* Validation
* Squad analysis

### Phase 10 — Visual Squad Generator

* Pitch renderer
* Recommended XI
* Transfer visualization
* Export image

### Phase 11 — Advanced Intelligence

* Chips
* News
* Mini-leagues
* Manager profiles
* Natural-language assistant

Do not skip foundational phases to build flashy UI.

---

# 36. AGENT BEHAVIOR

You are an engineering agent, not merely a code generator.

Before implementing a significant feature:

1. Inspect the existing repository.
2. Understand current architecture.
3. Identify affected modules.
4. Propose the implementation approach.
5. Implement incrementally.
6. Run tests.
7. Run lint/type checks where configured.
8. Fix failures.
9. Update documentation.
10. Summarize exactly what changed.

Do not overwrite working systems unnecessarily.

Do not introduce dependencies without justification.

If a requirement is ambiguous, inspect the existing architecture and choose the least destructive interpretation.

If an external API/source is unavailable, create a provider abstraction/mock rather than hardcoding a workaround throughout the application.

---

# 37. IMPORTANT PRODUCT PHILOSOPHY

The system is a decision-support platform.

It must not pretend predictions are certainties.

Prefer:

> "Player A has a 68% probability of starting."

over:

> "Player A will start."

Prefer:

> "Transfer A → B has an estimated +7.2 points over five Gameweeks."

over:

> "Transfer A → B is definitely better."

Expose uncertainty.

Expose assumptions.

Expose alternatives.

Allow the manager to make the final decision.

---

# 38. CURRENT TASK

Do NOT build the entire platform in one pass.

Begin with **Phase 1: Foundation**.

First inspect the repository and existing files.

Then produce a concise implementation assessment containing:

1. Current repository structure
2. Existing code that can be reused
3. Missing architecture
4. Dependencies required
5. Proposed Phase 1 changes
6. Potential risks
7. Exact files you intend to create/modify

Do not modify files until this assessment is complete.

After the assessment, wait for approval before implementing Phase 1.

The project must remain clean, modular and extensible throughout development.

---

# 39. SUCCESS CRITERIA

The eventual finished platform should be capable of:

```text
Upload FPL screenshot
        ↓
Recognize squad
        ↓
Validate players
        ↓
Analyze squad
        ↓
Predict player performance
        ↓
Evaluate fixtures
        ↓
Simulate future outcomes
        ↓
Optimize transfers
        ↓
Evaluate captain
        ↓
Evaluate chips
        ↓
Compare alternatives
        ↓
Explain recommendations
        ↓
Generate refined squad
        ↓
Render recommended FPL-style squad image
```

The ultimate objective is:

> **Build an FPL intelligence system that helps managers make better decisions, not merely a website that displays football statistics.**
