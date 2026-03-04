# Pipeline Architecture

ecPoint processes ensemble weather forecasts in a two-phase pipeline: environment setup followed by iterative post-processing over all (date, time, step) combinations.

## Phase 1 — Environment setup

1. **Build paths** — `build_paths(cfg)` resolves all directory and file paths from the configuration into an `EcPointPaths` dataclass.
2. **Validate environment** — `validate_environment(cfg, paths)` checks that all required calibration files and sample GRIB files exist. All errors are collected and reported together.
3. **Create filesystem** — `create_filesystem(cfg, paths)` pre-creates the full directory tree (work and output directories, including per-ensemble-member subdirectories).
4. **Load calibration** — `load_calibration(paths)` reads breakpoint thresholds and FER tables from CSV into NumPy arrays stored in a `CalibrationData` dataclass.

## Phase 2 — Post-processing

For each combination of (base date, base time, forecast step):

### Step 1: Compute predictors

`compute_predictors()` derives 6 predictor fields per ensemble member from raw ECMWF GRIB data:

| # | Predictor | Source | Transformation |
|---|---|---|---|
| 1 | Total Precipitation (TP) | paramId 228.128, sfc | m to mm conversion |
| 2 | Convective Precipitation Ratio (CPR) | paramId 143.128 / 228.128, sfc | cp / tp (guarded against division by zero) |
| 3 | TP (repeat) | same as #1 | Used as a second predictor dimension |
| 4 | Wind Speed at 700 hPa | paramId 10.128, 700 hPa | sqrt(u² + v²), weighted time-average |
| 5 | CAPE | paramId 59.128, sfc | Weighted time-average |
| 6 | 24h Solar Radiation | paramId 228022, sfc | J/m² to W/m² conversion |

Weighted time-averaging combines the field at the start and end of the accumulation window using the formula:

```
average = weight_start * field_start + weight_end * field_end
```

where the weights depend on the accumulation period relative to the forecast step.

### Step 2: Post-process ensemble

For each ensemble member:

1. **Classify weather types** — Each grid point is compared against the breakpoint thresholds for every weather type. A grid point matches a weather type if all its predictor values fall within the `[low, high)` ranges. The implementation uses vectorised NumPy broadcasting for performance.

2. **Apply FERs** — For matched grid points, the raw forecast (predictand) is multiplied by each of the 100 FER values: `corrected = predictand × (FER + 1)`. This produces a 100-member CDF per grid point.

3. **Save outputs** — Three products per ensemble member:
   - Grid-scale bias-corrected rainfall
   - Weather type codes
   - Full bias-corrected CDF (100 fields)

### Step 3: Compute percentiles

Combines all CDFs across ensemble members into a global CDF matrix of size `(num_ensemble_members × num_FERs, num_grid_points)`, then computes the requested percentiles using `np.percentile` along the ensemble/FER axis.

### Step 4: Move outputs

Moves files from the temporary work directories to the permanent output directories.

### Cleanup

After all (date, time, step) combinations are processed, temporary work and input directories are removed.

## Directory layout

```
main_dir/
├── input_db/                                # Raw ECMWF GRIB inputs
├── scripts/comp_files/
│   ├── map_func/{mode}/ecpoint_{var}/{acc}/{ver}/
│   │   ├── breakpoints_wt.txt               # WT breakpoint thresholds
│   │   └── fers.txt                         # Forecast Error Ratios
│   └── samples/ecmwf_ens_18km/global/
│       └── global.grib                      # Spatial template
└── {mode}/ecpoint_{var}/{acc}/{ver}/
    ├── work/                                # Temporary (cleaned up)
    │   ├── predict/                         # Predictand + predictors per EM
    │   ├── pt_rain_cdf/                     # Bias-corrected CDFs
    │   ├── grid_rain/                       # Grid-scale corrected rainfall
    │   ├── weather_types/                   # WT codes per EM
    │   └── percentiles/                     # Computed percentile fields
    └── forecasts/                           # Final output products
        ├── pt_bias_corr_perc/               # Point percentile forecasts
        ├── grid_bias_corr_vals/             # Grid-scale corrected values
        └── weather_types/                   # WT codes (all EMs concatenated)
```
