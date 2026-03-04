# Contributing

## Development setup

```bash
git clone https://github.com/FatimaPillosu/ecpoint-python.git
cd ecpoint-python
pip install -e ".[dev]"
```

## Running tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# With coverage report
pytest --cov=ecpoint

# Specific test file
pytest tests/test_config.py

# Specific test
pytest tests/test_config.py::TestConfigDefaults::test_default_config_is_valid
```

The test suite runs without GRIB data — all GRIB I/O is either mocked or tested only for error handling. Tests marked with `skip` require the full ecPoint calibration dataset.

## Project layout

- `src/ecpoint/ecpoint.py` — Single-module implementation containing all logic.
- `tests/conftest.py` — Shared fixtures providing minimal configs, paths, and calibration data.
- `tests/test_*.py` — Test modules organised by functional area.
- `data/mapping_functions/` — Default calibration tables.

## Code style

- Type annotations are used throughout (`from __future__ import annotations`).
- Configuration is validated with Pydantic models.
- Numerical operations use vectorised NumPy — avoid Python-level loops over grid points.

## Adding a new variable

To support a variable beyond rainfall:

1. Add an entry to `VARIABLE_REGISTRY` in `ecpoint.py` with the required param IDs, level types, and valid accumulations.
2. Define the corresponding predictor specifications (similar to `RAINFALL_PREDICTORS`).
3. Implement the predictor computation logic for the new variable.
4. Prepare calibration data (breakpoints and FERs) from a training period.
5. Add tests covering the new predictor formulas and weather type classification.
