# Data Engineering and Integrity

## Data Sources

* Inspect the existing data pipeline and supported sources before introducing another source.
* Prefer reliable, documented, accessible sources and free data access where practical.
* Do not assume an API endpoint, field, dataset, or provider is available without verification.
* Keep source-specific extraction separate from internal data normalization.

## Validation

* Validate required columns, data types, identifiers, ranges, missing values, duplicates, and relationships between records.
* Handle missing or malformed data explicitly. Do not silently replace unknown values with misleading defaults.
* Distinguish a genuine zero from missing data.
* Report rejected records and validation failures clearly.
* Make ingestion repeatable and safe to rerun.

## Historical Integrity

* Preserve the distinction between historical observations and information available at prediction time.
* Record relevant timestamps, gameweeks, seasons, and source provenance.
* Avoid overwriting historical records with current values when that would corrupt historical analysis.
* Prevent duplicate ingestion from creating duplicate records.
* Make transformations deterministic wherever practical.

## Feature Engineering

* Keep feature creation separate from raw extraction.
* Document feature meanings, units, time windows, and assumptions.
* Avoid using future information to construct historical prediction features.
* Handle newly promoted teams, new players, incomplete histories, and missing fixtures explicitly.

## Reliability

* Add tests for valid records, invalid records, missing fields, duplicates, empty inputs, and edge cases.
* Use clear exceptions and useful error messages.
* Do not hide ingestion failures behind broad exception handlers.
* Do not introduce large datasets or generated artifacts into Git without a clear reason.
* Never fabricate data to make a pipeline appear successful.
