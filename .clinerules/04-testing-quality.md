# Testing and Code Quality

## Before Changes

* Inspect the relevant implementation, tests, project configuration, and existing conventions.
* Reproduce the reported problem when practical.
* Identify the intended behavior before modifying code.

## Tests

* Add or update tests for every meaningful behavior change.
* Include relevant edge cases, invalid inputs, and regression scenarios.
* Use deterministic tests wherever practical.
* Avoid tests that pass merely because they catch any exception.
* Do not weaken, delete, skip, or rewrite valid tests just to obtain a passing test suite.
* Mock external services when appropriate; ordinary unit tests should not require live API access.

## Quality Tools

Use the tools configured by the project, where installed:

* `python -m pytest`
* `ruff check .`
* `mypy src/fpl_engine`

Follow the actual configuration in `pyproject.toml`. Do not assume these commands have passed without running them.

## Dependencies and Compatibility

* Avoid unnecessary dependencies.
* Check Python version requirements before adopting newer language features.
* Do not replace enums, typing patterns, or APIs with newer alternatives without checking compatibility.
* Keep development and runtime dependencies appropriately separated.
* Do not install packages or change dependency constraints without authorization.

## Failure Reporting

* Report the exact checks run and their results.
* Distinguish pre-existing failures from regressions introduced by your changes.
* If a command fails, investigate the actual cause instead of hiding it.
* Never claim tests, linting, or type checking passed when they were not executed successfully.

## Completion Criteria

A task is complete only when:

1. The requested behavior is implemented.
2. Relevant tests are added or updated.
3. Applicable quality checks have been run.
4. Documentation is updated when necessary.
5. The final report accurately describes changes, verification, and known limitations.
