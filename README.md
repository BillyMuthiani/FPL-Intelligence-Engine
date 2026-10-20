# FPL Intelligence Engine

A production-quality Fantasy Premier League (FPL) analytics and decision-support platform.

## Overview

The FPL Intelligence Engine is a modular, testable, and explainable decision-support system for FPL managers. It combines official FPL data, historical Premier League data, advanced football statistics, fixture analysis, player availability information, machine-learning predictions, probabilistic simulation, squad optimization, transfer optimization, captaincy analysis, and chip strategy.

The system is designed with a three-layer architecture:

1. **Intelligence Engine** - Core prediction, simulation, and optimization logic (no UI dependencies)
2. **Application Layer** - Streamlit (initially), later FastAPI backend / web frontend / mobile client
3. **Presentation Layer** - Tables, charts, squad visualizations, pitch layouts, recommendation cards

## Installation

### Prerequisites

- Python 3.11+
- pip

### Install from source

```bash
git clone https://github.com/BillyMuthiani/fpl-intelligence-engine.git
cd fpl-intelligence-engine
pip install -e ".[dev]"
```

### Install production dependencies only

```bash
pip install -e .
```

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Runtime environment (`development`, `production`, `testing`) |
| `DEBUG` | `false` | Enable debug mode |
| `FPL_BASE_URL` | `https://fantasy.premierleague.com/api` | Base URL for official FPL API |
| `FPL_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for FPL API requests |
| `FPL_MAX_RETRIES` | `3` | Maximum retry attempts for failed requests |
| `FPL_RETRY_BACKOFF_FACTOR` | `0.5` | Backoff factor for retry delays |
| `UNDERSTAT_BASE_URL` | (empty) | Base URL for Understat API (optional) |
| `RANDOM_SEED` | `42` | Random seed for reproducible simulations |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | `json` | Log format: `json` or `console` |

## Usage

### Quick Start

```python
from fpl_engine import get_settings, configure_logging, get_logger
from fpl_engine.providers import FPLClient

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Load settings
settings = get_settings()

# Create FPL client
with FPLClient(settings) as client:
    # Get all players
    players = client.get_players()
    logger.info("Loaded players", count=len(players))
    
    # Get current gameweek
    current_gw = client.get_current_gameweek()
    logger.info("Current gameweek", gameweek=current_gw.id if current_gw else None)
    
    # Get fixtures for next 3 gameweeks
    fixtures = client.get_fixtures()
    upcoming = [f for f in fixtures if f.event <= (current_gw.id + 2) if current_gw]
```

### Working with Data Models

```python
from fpl_engine.models import Player, PlayerPosition, PlayerStatus, Team, Squad, SquadPlayer

# Create a player
player = Player(
    id=1,
    web_name="Salah",
    first_name="Mohamed",
    second_name="Salah",
    team_id=14,
    position=PlayerPosition.MIDFIELDER,
    status=PlayerStatus.AVAILABLE,
    now_cost=130,  # £13.0m
    code=123456,
    element_type=3,
    team_code=14,
)

# Create a team
team = Team(
    id=14,
    code=3,
    name="Liverpool",
    short_name="LIV",
    strength=5,
)

# Create a squad
squad_players = [
    SquadPlayer(player=player, position=1, is_captain=True),
    # ... add 14 more players
]
squad = Squad(players=squad_players, gameweek=1, bank=5.0)

# Validate formation
valid, errors = squad.validate_formation()
if not valid:
    print("Formation errors:", errors)

# Validate club constraints
valid, errors = squad.validate_club_constraints()

# Get formation string
print(f"Formation: {squad.get_formation_string()}")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fpl_engine

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Run linter
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/
```

## Project Structure

```
fpl-intelligence-engine/
├── src/
│   └── fpl_engine/
│       ├── __init__.py
│       ├── config.py          # Configuration management (Pydantic Settings)
│       ├── logging.py         # Structured logging (structlog)
│       ├── models/            # Canonical data models
│       │   ├── __init__.py
│       │   ├── player.py
│       │   ├── team.py
│       │   ├── fixture.py
│       │   ├── gameweek.py
│       │   └── squad.py
│       └── providers/         # Data provider abstractions
│           ├── __init__.py
│           ├── protocols.py   # Provider protocols (runtime_checkable Protocols)
│           └── fpl_client.py  # Official FPL API client
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_config.py
│   ├── test_models.py
│   └── test_providers.py
├── pyproject.toml            # Project metadata, dependencies, tool config
├── .env.example              # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

## Architecture

### Data Flow

```
Official FPL API
    │
    ▼
FPLClient (implements CompositeDataProvider)
    │
    ├─► get_bootstrap_data() → Player, Team, Gameweek, ElementType
    ├─► get_fixtures() → Fixture
    ├─► get_player_history() → Gameweek stats
    └─► get_gameweeks() → Gameweek
    │
    ▼
Canonical Models (Player, Team, Fixture, Gameweek, Squad)
    │
    ▼
Feature Engineering → Prediction Models → Monte Carlo → Optimization → Decision Engine
```

### Provider Protocols

The system uses Python `Protocol` classes (with `@runtime_checkable`) to define provider interfaces:

- `PlayerDataProvider` - Player data access
- `TeamDataProvider` - Team data access
- `FixtureDataProvider` - Fixture data access
- `GameweekDataProvider` - Gameweek data access
- `BootstrapDataProvider` - Combined bootstrap data
- `CompositeDataProvider` - All of the above

This allows swapping data sources (e.g., mock data for testing, different APIs) without changing downstream logic.

## Development

### Adding New Features

1. Follow the three-layer architecture
2. Add models to `src/fpl_engine/models/`
3. Add provider protocols to `src/fpl_engine/providers/protocols.py`
4. Implement providers in `src/fpl_engine/providers/`
5. Write tests in `tests/`
6. Run `ruff check`, `mypy`, and `pytest` before committing

### Phase 1 Scope (Foundation)

This phase delivers:
- ✅ Package structure and configuration
- ✅ Canonical data models (Player, Team, Fixture, Gameweek, Squad)
- ✅ Provider protocols and official FPL API client
- ✅ Logging and environment-based configuration
- ✅ Test infrastructure with pytest, fixtures, mocking
- ✅ Type checking (mypy), linting (ruff)

## License

MIT License - see [LICENSE](LICENSE) for details.