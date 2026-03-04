# ecPoint

Post-processing system for NWP ensemble forecasts. Converts raw ensemble model output into bias-corrected point rainfall forecasts using weather-type-dependent calibration.

## Overview

ecPoint improves the skill of numerical weather prediction (NWP) ensemble forecasts by applying location-dependent bias corrections. The method:

1. **Classifies weather types (WTs)** at each grid point based on atmospheric predictors (convective precipitation ratio, wind speed at 700 hPa, CAPE, solar radiation).
2. **Applies Forecast Error Ratios (FERs)** — empirically derived multiplicative correction factors — to transform the raw forecast into a bias-corrected Cumulative Distribution Function (CDF).
3. **Computes percentiles** across all ensemble members and FER quantiles, producing a full probabilistic point-rainfall forecast.

## Installation

### Requirements

- Python >= 3.10
- [earthkit-data](https://github.com/ecmwf/earthkit-data) >= 0.10
- NumPy >= 1.24
- pandas >= 2.0
- Pydantic >= 2.0
- Click >= 8.0

### Install from source

```bash
git clone https://github.com/FatimaPillosu/ecpoint-python.git
cd ecpoint-python
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
ecpoint \
  --var rainfall \
  --acc 12 \
  --date-start 2020-02-19 \
  --date-end 2020-02-19 \
  --step-start 0 \
  --step-final 114 \
  --step-disc 6 \
  --ens-start 0 \
  --ens-end 50 \
  --main-dir /path/to/project \
  --cal-version 1.0.0 \
  -v
```

Use `--config` to load parameters from a JSON file:

```bash
ecpoint --config config.json -v
```

### Python API

```python
from ecpoint import EcPointConfig, run_ecpoint

cfg = EcPointConfig(
    var_to_postprocess="rainfall",
    accumulation_hours=12,
    base_date_start=datetime.date(2020, 2, 19),
    base_date_end=datetime.date(2020, 2, 19),
    step_start=0,
    step_final=114,
    step_disc=6,
    main_dir=Path("/path/to/project"),
)

run_ecpoint(cfg)
```

## Project Structure

```
ecpoint-python/
├── src/ecpoint/
│   ├── __init__.py          # Public API: EcPointConfig, run_ecpoint
│   └── ecpoint.py           # Full pipeline implementation
├── tests/
│   ├── conftest.py          # Shared fixtures (configs, calibration data)
│   ├── test_calibration.py  # Calibration CSV loading
│   ├── test_config.py       # Configuration validation
│   ├── test_filesystem.py   # Path building and directory creation
│   ├── test_grib_utils.py   # GRIB I/O and helper functions
│   ├── test_percentiles.py  # Percentile computation
│   ├── test_postprocess.py  # WT classification and FER application
│   ├── test_predict.py      # Predictor computation formulas
│   └── test_validation.py   # Environment validation
├── data/mapping_functions/
│   ├── breakpoints_wt.txt   # WT classification thresholds
│   └── fers.txt             # Forecast Error Ratios
└── pyproject.toml
```

## Pipeline

The processing pipeline runs in two phases:

### Phase 1 — Environment Setup

- Build directory paths from configuration
- Validate that calibration files and sample GRIB files exist
- Create the output directory tree
- Load calibration tables (breakpoints and FERs)

### Phase 2 — Post-Processing

For each (base date, base time, forecast step) combination:

1. **Compute predictors**: derive 6 fields per ensemble member from raw ECMWF GRIB data:
   - Total Precipitation (TP, m to mm)
   - Convective Precipitation Ratio (CPR)
   - TP (repeated as predictor)
   - Wind Speed at 700 hPa (weighted time average of u/v components)
   - CAPE (weighted time average)
   - 24h Solar Radiation (J/m^2 to W/m^2)

2. **Post-process each ensemble member**:
   - Classify each grid point into a weather type using breakpoint thresholds
   - Apply FERs to generate a bias-corrected CDF
   - Save grid-scale corrected rainfall and weather type codes

3. **Compute percentiles** across all ensemble members and FER quantiles

4. **Move outputs** to the final output directory

### Directory Layout

```
main_dir/
├── input_db/                          # Raw ECMWF GRIB inputs
├── scripts/comp_files/
│   ├── map_func/{mode}/ecpoint_{var}/{acc}/{ver}/
│   │   ├── breakpoints_wt.txt         # WT breakpoint thresholds
│   │   └── fers.txt                   # Forecast Error Ratios
│   └── samples/ecmwf_ens_18km/global/
│       └── global.grib                # Spatial template for output
└── {mode}/ecpoint_{var}/{acc}/{ver}/
    ├── work/                          # Temporary intermediate files
    │   ├── predict/                   # Predictand + predictors per EM
    │   ├── pt_rain_cdf/               # Bias-corrected CDFs
    │   ├── grid_rain/                 # Grid-scale corrected rainfall
    │   ├── weather_types/             # WT codes per EM
    │   └── percentiles/               # Computed percentile fields
    └── forecasts/                     # Final output products
        ├── pt_bias_corr_perc/         # Point percentile forecasts
        ├── grid_bias_corr_vals/       # Grid-scale corrected values
        └── weather_types/             # WT codes (all EMs concatenated)
```

## Calibration Data

The calibration files are derived from a training period comparing forecasts against observations:

- **breakpoints_wt.txt**: defines weather types as hyper-rectangular regions in predictor space. Each row specifies `[low, high)` threshold pairs for every predictor.
- **fers.txt**: for each weather type, provides 100 Forecast Error Ratios representing quantiles of the empirical error distribution. FER = (obs - fcst) / fcst, so the correction factor is (FER + 1).

## Testing

```bash
pytest
```

The test suite covers configuration validation, calibration loading, predictor formulas, weather type classification, FER application, percentile computation, and filesystem operations — all without requiring real GRIB data.

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:

- [Installation](docs/installation.md) — requirements, setup, and input data
- [Usage](docs/usage.md) — CLI options, Python API, and JSON config format
- [Configuration Reference](docs/configuration.md) — all `EcPointConfig` parameters and validation rules
- [Pipeline Architecture](docs/pipeline.md) — detailed processing steps and directory layout
- [Calibration Data](docs/calibration.md) — file formats and how to create custom calibration tables
- [Contributing](docs/contributing.md) — development setup, testing, and adding new variables

## License

Apache License 2.0. Copyright 2020 ECMWF.
