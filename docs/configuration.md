# Configuration Reference

`EcPointConfig` is a [Pydantic](https://docs.pydantic.dev/) model that validates all parameters at instantiation time. Invalid combinations raise `ValueError` with descriptive messages.

## Parameters

### Variable and accumulation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `var_to_postprocess` | `Literal["rainfall"]` | `"rainfall"` | Variable to post-process. Currently only rainfall is supported. |
| `accumulation_hours` | `int` | `12` | Accumulation period in hours. Must be in the variable's `valid_accumulations` list (rainfall: `[12]`). |

### Calibration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calibration_version` | `str` | `"1.0.0"` | Version string for the calibration data directory. |
| `run_mode` | `str` | `"dev"` | Run mode label, used in output directory paths. |

### Date range

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_date_start` | `datetime.date` | `2020-02-19` | First model initialisation date (inclusive). |
| `base_date_end` | `datetime.date` | `2020-02-19` | Last model initialisation date (inclusive). Must be >= `base_date_start`. |

### Time range

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_time_start` | `int` | `0` | First base time in UTC hours (0–23). |
| `base_time_end` | `int` | `0` | Last base time in UTC hours (0–23). Must be >= `base_time_start`. |
| `base_time_disc` | `int` | `12` | Step size for iterating over base times. |

### Forecast steps

| Parameter | Type | Default | Description |
|---|---|---|---|
| `step_start` | `int` | `0` | First forecast lead-time step (non-negative). |
| `step_final` | `int` | `114` | Last forecast lead-time step. Must be >= `step_start`. |
| `step_disc` | `int` | `6` | Increment between forecast steps. |

### Ensemble members

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ensemble_member_start` | `int` | `0` | First ensemble member number (non-negative). |
| `ensemble_member_end` | `int` | `50` | Last ensemble member number. Must be >= `ensemble_member_start`. |

### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `percentiles` | `list[int]` | `[1, 2, ..., 99]` | Percentiles to compute. Each value must be 1–99, no duplicates. The list is automatically sorted. |
| `float_precision` | `Literal["float32", "float64"]` | `"float32"` | NumPy dtype for numerical arrays. `float32` is faster; `float64` offers more precision. |

### Filesystem paths

| Parameter | Type | Default | Description |
|---|---|---|---|
| `main_dir` | `Path` | User home | Project root directory containing input data and output products. |
| `scripts_dir` | `Path \| None` | `main_dir / "scripts"` | Directory containing calibration data and sample GRIB files. |

### Point mode

| Parameter | Type | Default | Description |
|---|---|---|---|
| `point_lat` | `float \| None` | `None` | Latitude for point mode (-90 to 90). When set with `point_lon`, enables point-mode processing. |
| `point_lon` | `float \| None` | `None` | Longitude for point mode (-180 to 360). Must be set together with `point_lat`. |
| `data_source` | `Literal["local", "polytope"]` | `"local"` | Data source: `"local"` reads GRIB files from disk, `"polytope"` requests point data from ECMWF's Polytope service. |

### Formatting

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_digits_base_time` | `int` | `2` | Zero-padding width for base time in filenames. |
| `num_digits_step` | `int` | `3` | Zero-padding width for forecast step in filenames. |
| `num_digits_acc` | `int` | `3` | Zero-padding width for accumulation hours in filenames. |
| `num_digits_ensemble_member` | `int` | `3` | Zero-padding width for ensemble member number in filenames. |

## Derived properties

These are computed from the configuration and available as read-only properties:

| Property | Type | Description |
|---|---|---|
| `num_ensemble_members` | `int` | `ensemble_member_end - ensemble_member_start + 1` |
| `accumulation_str` | `str` | Zero-padded accumulation string (e.g. `"012"`) |
| `variable_info` | `dict` | Metadata from the variable registry (param IDs, levels, thresholds) |
| `min_predictand_value` | `float` | Minimum threshold below which precipitation is treated as zero |
| `numpy_dtype` | `np.dtype` | NumPy dtype object corresponding to `float_precision` |
| `is_point_mode` | `bool` | `True` when both `point_lat` and `point_lon` are set |

## Validation rules

The following cross-field constraints are enforced:

- `base_date_end >= base_date_start`
- `base_time_end >= base_time_start`
- `base_time_start` and `base_time_end` must be 0–23
- `ensemble_member_end >= ensemble_member_start`
- `ensemble_member_start >= 0`
- `step_final >= step_start`
- `step_start >= 0`
- `accumulation_hours` must be in the variable's `valid_accumulations` list
- `percentiles` must be non-empty, all values 1–99, no duplicates
- `point_lat` and `point_lon` must both be set or both be `None`
- `data_source="polytope"` requires point mode (both `point_lat` and `point_lon` set)
