# Installation

## Requirements

- Python >= 3.10
- [earthkit-data](https://github.com/ecmwf/earthkit-data) >= 0.10
- NumPy >= 1.24
- pandas >= 2.0
- Pydantic >= 2.0
- Click >= 8.0

## Install from source

```bash
git clone https://github.com/FatimaPillosu/ecpoint-python.git
cd ecpoint-python
pip install -e .
```

## Development install

Includes testing dependencies (pytest, pytest-cov, pytest-tmp-files):

```bash
pip install -e ".[dev]"
```

## Verify installation

After installing, verify the CLI is available:

```bash
ecpoint --help
```

Or import in Python:

```python
from ecpoint import EcPointConfig, run_ecpoint
```

## Calibration data

ecPoint requires calibration files (breakpoints and FERs) derived from a training period. Default calibration data for rainfall is included in the repository under `data/mapping_functions/`. To use custom calibration data, place your files in the expected directory structure (see [Calibration](calibration.md)) and point `scripts_dir` to their parent directory.

## Input data

ecPoint processes ECMWF ensemble GRIB files. You must provide:

1. **Ensemble GRIB files** — raw ECMWF ensemble forecast output, organised in the directory tree under `main_dir/input_db/`.
2. **Sample GRIB file** — a single GRIB field used as a spatial template for output metadata. Place it at `main_dir/scripts/comp_files/samples/ecmwf_ens_18km/global/global.grib`.
