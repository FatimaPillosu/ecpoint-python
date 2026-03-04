# Calibration Data

ecPoint uses empirically derived calibration tables to correct ensemble forecasts. These tables are produced by comparing historical forecasts against observations during a training period.

## Files

Two CSV files are required, located under `scripts/comp_files/map_func/{mode}/ecpoint_{var}/{acc}/{ver}/`:

### breakpoints_wt.txt

Defines weather types as hyper-rectangular regions in predictor space.

**Format:** CSV with columns:

```
WTcode, pred_1_thrL, pred_1_thrH, pred_2_thrL, pred_2_thrH, ...
```

| Column | Description |
|---|---|
| `WTcode` | Integer weather type identifier |
| `pred_N_thrL` | Lower threshold for predictor N (inclusive) |
| `pred_N_thrH` | Upper threshold for predictor N (exclusive) |

A grid point is assigned to a weather type if **all** its predictor values fall within the corresponding `[thrL, thrH)` ranges. A value of `-9999` means "no lower bound" (i.e., negative infinity).

**Example:**

```csv
11111,-9999,0.25,-9999,2,-9999,5,-9999,50,-9999,70
11112,-9999,0.25,-9999,2,-9999,5,-9999,50,70,9999
```

Weather type `11111` matches grid points where predictor 1 < 0.25, predictor 2 < 2, predictor 3 < 5, predictor 4 < 50, and predictor 5 in [−∞, 70).

### fers.txt

Contains Forecast Error Ratios for each weather type.

**Format:** CSV with columns:

```
Wtcode, FER1, FER2, ..., FER100
```

Each row corresponds to a weather type (matching `WTcode` in breakpoints). The 100 FER columns represent quantiles of the empirical error distribution for that weather type.

**FER definition:**

```
FER = (observation − forecast) / forecast
```

The bias correction applies as:

```
corrected_value = raw_forecast × (FER + 1)
```

This transforms the single raw forecast value into a 100-member CDF representing the range of likely point values given the weather type.

## How weather types are matched

The classification algorithm iterates through weather types in order. For each grid point:

1. Check all predictor values against the breakpoint thresholds.
2. The **first** matching weather type (where all predictors fall within bounds) is assigned.
3. If no weather type matches, the grid point receives a code of `0` and its FER-corrected values default to the raw forecast.

The implementation uses vectorised NumPy broadcasting: all grid points are classified simultaneously against each weather type, avoiding Python-level loops over grid points.

## Creating custom calibration data

To use custom calibration tables:

1. Prepare `breakpoints_wt.txt` and `fers.txt` in the format described above.
2. Ensure both files have the **same number of rows** and matching weather type codes.
3. The number of predictor threshold pairs in breakpoints must match the number of predictors defined in the variable registry (6 for rainfall).
4. FER files should have exactly 100 FER columns.
5. Place both files in the appropriate directory and set `calibration_version` in the configuration to match.
