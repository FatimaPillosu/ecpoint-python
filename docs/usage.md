# Usage

ecPoint can be used as a command-line tool or as a Python library.

## Command line

### Basic invocation

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

### Using a JSON configuration file

```bash
ecpoint --config config.json -v
```

CLI arguments override values from the config file, so you can use a base config and adjust individual parameters:

```bash
ecpoint --config config.json --date-start 2021-01-01 --date-end 2021-01-31
```

### CLI options

| Option | Description | Default |
|---|---|---|
| `--config` | Path to JSON configuration file | None |
| `--var` | Variable to post-process | `rainfall` |
| `--acc` | Accumulation period in hours | `12` |
| `--cal-version` | Calibration data version string | `1.0.0` |
| `--run-mode` | Run mode label for output paths | `dev` |
| `--date-start` | First base date (YYYY-MM-DD) | `2020-02-19` |
| `--date-end` | Last base date (YYYY-MM-DD) | `2020-02-19` |
| `--time-start` | First base time (UTC hour, 0–23) | `0` |
| `--time-end` | Last base time (UTC hour, 0–23) | `0` |
| `--time-disc` | Base time step size in hours | `12` |
| `--step-start` | First forecast lead-time step | `0` |
| `--step-final` | Last forecast lead-time step | `114` |
| `--step-disc` | Step increment | `6` |
| `--ens-start` | First ensemble member number | `0` |
| `--ens-end` | Last ensemble member number | `50` |
| `--main-dir` | Project root directory | User home |
| `--lat` | Latitude for point mode (-90 to 90) | None |
| `--lon` | Longitude for point mode (-180 to 360) | None |
| `--data-source` | Data source: `local` or `polytope` | `local` |
| `-v`, `--verbose` | Increase verbosity (repeat for debug) | `0` (WARNING) |

### Verbosity levels

| Flag | Level | Description |
|---|---|---|
| (none) | WARNING | Only warnings and errors |
| `-v` | INFO | Progress messages for each processing step |
| `-vv` | DEBUG | Detailed diagnostic output |

## Python API

### Minimal example

```python
import datetime
from pathlib import Path
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

### Loading configuration from JSON

```python
from ecpoint.ecpoint import load_config

# From file only
cfg = load_config(Path("config.json"))

# From file with overrides
cfg = load_config(
    Path("config.json"),
    base_date_start=datetime.date(2021, 6, 1),
    base_date_end=datetime.date(2021, 6, 30),
)
```

### JSON configuration format

```json
{
  "var_to_postprocess": "rainfall",
  "accumulation_hours": 12,
  "calibration_version": "1.0.0",
  "run_mode": "dev",
  "base_date_start": "2020-02-19",
  "base_date_end": "2020-02-19",
  "base_time_start": 0,
  "base_time_end": 0,
  "base_time_disc": 12,
  "step_start": 0,
  "step_final": 114,
  "step_disc": 6,
  "ensemble_member_start": 0,
  "ensemble_member_end": 50,
  "main_dir": "/path/to/project",
  "percentiles": [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
}
```

### Selecting specific percentiles

By default all 99 percentiles (1–99) are computed. To compute only a subset:

```python
cfg = EcPointConfig(
    percentiles=[10, 25, 50, 75, 90],
    # ... other parameters
)
```

## Point mode

Point mode processes a single lat/lon location instead of the full global grid. This is much faster and outputs CSV instead of GRIB.

### CLI — point mode with local GRIB files

```bash
ecpoint --lat 51.5 --lon -0.1 \
  --var rainfall --acc 12 \
  --date-start 2020-02-19 --date-end 2020-02-19 \
  --main-dir /path/to/project -v
```

### CLI — point mode with ECMWF Polytope service

Retrieves only the needed point data from ECMWF servers, avoiding full field downloads:

```bash
ecpoint --lat 51.5 --lon -0.1 --data-source polytope \
  --var rainfall --acc 12 \
  --date-start 2020-02-19 --date-end 2020-02-19 \
  --main-dir /path/to/project -v
```

Requires the `polytope` extra: `pip install ecpoint[polytope]`

### Python API — point mode

```python
from ecpoint import EcPointConfig, run_ecpoint

cfg = EcPointConfig(
    point_lat=51.5,
    point_lon=-0.1,
    var_to_postprocess="rainfall",
    accumulation_hours=12,
    base_date_start=datetime.date(2020, 2, 19),
    base_date_end=datetime.date(2020, 2, 19),
    main_dir=Path("/path/to/project"),
)

run_ecpoint(cfg)  # automatically uses point-mode pipeline, outputs CSV
```

### CSV output format

Point mode writes a CSV file to the output directory with one row per (date, time, step) combination:

```csv
date,time,step_start,step_end,lat,lon,wt_code,grid_bc,p1,p2,...,p99
20200219,00,0,12,51.5,-0.1,11234,2.4500,0.1000,0.3000,...,6.8000
```

Columns:
- `date`, `time`: model initialisation date and UTC hour
- `step_start`, `step_end`: forecast lead-time window
- `lat`, `lon`: point coordinates
- `wt_code`: dominant weather type code from the last ensemble member
- `grid_bc`: grid-scale bias-corrected rainfall (mean of CDF)
- `pN`: one column per requested percentile
