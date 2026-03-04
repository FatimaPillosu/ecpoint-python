"""
ecPoint — Post-processing system for NWP ensemble forecasts.

Converts raw ensemble model output into bias-corrected point rainfall
forecasts using weather-type-dependent calibration (Forecast Error Ratios).

Uses earthkit-data for GRIB I/O and numpy for numerical operations.

Copyright 2020 ECMWF. Licensed under the Apache License version 2.0.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import click
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

import earthkit.data as ekd

logger = logging.getLogger("ecpoint")


# =============================================================================
# Configuration
# =============================================================================

# Registry of supported post-processing variables.
# Each entry maps a variable name to its ECMWF GRIB parameter codes and
# processing constraints:
#   - predictand_code: GRIB paramId for the raw forecast field
#   - pp_code: GRIB paramId for the post-processed output field
#   - min_predictand_value: values below this threshold (in mm for rainfall)
#     are treated as zero to filter out GRIB packing-precision artifacts
#   - valid_accumulations: allowed accumulation periods (hours)
VARIABLE_REGISTRY: dict[str, dict] = {
    "rainfall": {
        "predictand_code": 228.128,       # Total precipitation (paramId 228)
        "level_type": "sfc",
        "level": 0,
        "min_predictand_value": 0.04,     # 0.04 mm noise floor
        "pp_code": 82.128,               # Large-scale precipitation (output)
        "pp_level_type": "sfc",
        "pp_level": 0,
        "valid_accumulations": [12],      # Only 12h accumulation supported
    },
}

# Predictor fields used for weather type classification of rainfall.
# These predictors partition the atmospheric state into weather types (WTs),
# each representing a distinct precipitation regime (e.g., convective vs
# stratiform, strong vs weak synoptic forcing). The order must match the
# column order in the breakpoints calibration file.
# Field 0 (tp) is the predictand; fields 1-5 are the classification predictors.
RAINFALL_PREDICTORS = [
    {"name": "tp", "param_id": 228.128, "level_type": "sfc", "level": 0},         # Total precipitation (predictand)
    {"name": "cpr", "param_id": 143.128, "level_type": "sfc", "level": 0},        # Convective precipitation ratio (cp/tp)
    {"name": "tp_repeat", "param_id": 228.128, "level_type": "sfc", "level": 0},  # TP repeated as a predictor
    {"name": "wspd700", "param_id": 10.128, "level_type": "pl", "level": 700},    # Wind speed at 700 hPa (synoptic forcing)
    {"name": "cape", "param_id": 59.128, "level_type": "sfc", "level": 0},        # Convective Available Potential Energy
    {"name": "sr24h", "param_id": 228022, "level_type": "sfc", "level": 0},       # 24h solar radiation (diurnal cycle proxy)
]


class EcPointConfig(BaseModel):
    """All input parameters for an ecPoint run.

    Controls which variable is post-processed, the forecast date/time/step
    ranges to iterate over, ensemble member range, output percentiles, and
    file system layout. Pydantic validates all parameters at construction time.
    """

    # Which meteorological variable to post-process (currently only rainfall)
    var_to_postprocess: Literal["rainfall"] = "rainfall"
    # Accumulation period in hours (must match VARIABLE_REGISTRY valid values)
    accumulation_hours: int = Field(default=12, ge=0)
    # Calibration table version (selects the breakpoints/FERs directory)
    calibration_version: str = "1.0.0"
    # Run mode label used in output directory paths (e.g., "dev", "prod")
    run_mode: str = "dev"

    # Forecast base date range: the model initialization dates to process.
    # The pipeline iterates over all dates from base_date_start to base_date_end.
    base_date_start: datetime.date = datetime.date(2020, 2, 19)
    base_date_end: datetime.date = datetime.date(2020, 2, 19)
    # Forecast base time range (UTC hours): the model initialization times.
    # Iterates from base_time_start to base_time_end in base_time_disc steps.
    base_time_start: int = Field(default=0, ge=0, le=23)
    base_time_end: int = Field(default=0, ge=0, le=23)
    base_time_disc: int = Field(default=12, gt=0)
    # Forecast step range (hours from initialization): the lead times to process.
    # step_start is the beginning of the first accumulation window; step_final
    # is the beginning of the last. Each window spans [step, step+accumulation].
    step_start: int = Field(default=0, ge=0)
    step_final: int = Field(default=114, ge=0)
    step_disc: int = Field(default=6, gt=0)

    # Ensemble member range (inclusive). Member 0 is the control forecast;
    # members 1-50 are the perturbed forecasts in the ECMWF EPS.
    ensemble_member_start: int = Field(default=0, ge=0)
    ensemble_member_end: int = Field(default=50, ge=0)

    # Which percentiles (1-99) to compute from the bias-corrected CDF.
    # Default is all 99 percentiles for a full probabilistic description.
    percentiles: list[int] = Field(
        default_factory=lambda: list(range(1, 100))
    )

    # Root directory for inputs/outputs and optional override for scripts location
    main_dir: Path = Path.home()
    scripts_dir: Path | None = None  # defaults to main_dir / "scripts"

    # Zero-padding widths for file/directory naming consistency
    num_digits_base_time: int = 2
    num_digits_step: int = 3
    num_digits_acc: int = 3
    num_digits_ensemble_member: int = 2

    # Floating-point precision for numerical computations
    float_precision: Literal["float32", "float64"] = "float32"

    # --- Validators ---

    @field_validator("percentiles")
    @classmethod
    def _check_percentiles(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("percentiles must not be empty")
        for p in v:
            if p < 1 or p > 99:
                raise ValueError(
                    f"Percentile values must be 1-99, got {p}"
                )
        if len(v) != len(set(v)):
            raise ValueError("percentiles must not contain duplicates")
        return sorted(v)

    @model_validator(mode="after")
    def _cross_validate(self) -> "EcPointConfig":
        if self.base_date_start > self.base_date_end:
            raise ValueError(
                f"base_date_start ({self.base_date_start}) must be "
                f"<= base_date_end ({self.base_date_end})"
            )
        if self.base_time_start > self.base_time_end:
            raise ValueError(
                f"base_time_start ({self.base_time_start}) must be "
                f"<= base_time_end ({self.base_time_end})"
            )
        if self.ensemble_member_start > self.ensemble_member_end:
            raise ValueError(
                f"ensemble_member_start ({self.ensemble_member_start}) must "
                f"be <= ensemble_member_end ({self.ensemble_member_end})"
            )
        if self.step_start > self.step_final:
            raise ValueError(
                f"step_start ({self.step_start}) must be "
                f"<= step_final ({self.step_final})"
            )
        var_info = VARIABLE_REGISTRY.get(self.var_to_postprocess)
        if var_info and self.accumulation_hours not in var_info[
            "valid_accumulations"
        ]:
            raise ValueError(
                f"accumulation_hours={self.accumulation_hours} is not valid "
                f"for {self.var_to_postprocess}. "
                f"Valid: {var_info['valid_accumulations']}"
            )
        return self

    # --- Derived properties ---

    @property
    def num_ensemble_members(self) -> int:
        return self.ensemble_member_end - self.ensemble_member_start + 1

    @property
    def accumulation_str(self) -> str:
        return str(self.accumulation_hours).zfill(self.num_digits_acc)

    @property
    def variable_info(self) -> dict:
        return VARIABLE_REGISTRY[self.var_to_postprocess]

    @property
    def min_predictand_value(self) -> float:
        return self.variable_info["min_predictand_value"]

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.float_precision)


def load_config(config_path: Path | None = None, **overrides) -> EcPointConfig:
    """Load configuration from a YAML/JSON file, with optional overrides."""
    if config_path is not None:
        import json

        text = config_path.read_text()
        data = json.loads(text)
        data.update(overrides)
        return EcPointConfig(**data)
    return EcPointConfig(**overrides)


# =============================================================================
# Validation
# =============================================================================


def validate_environment(cfg: EcPointConfig, paths: "EcPointPaths") -> None:
    """Validate that the runtime environment is ready.

    Checks that required calibration files and sample GRIB files exist.
    Pydantic handles parameter-level validation; this function checks the
    filesystem.
    """
    errors: list[str] = []

    if not cfg.main_dir.is_dir():
        errors.append(f"main_dir does not exist: {cfg.main_dir}")

    bp_file = paths.breakpoints_file
    if not bp_file.is_file():
        errors.append(f"Breakpoints file not found: {bp_file}")

    fer_file = paths.fers_file
    if not fer_file.is_file():
        errors.append(f"FERs file not found: {fer_file}")

    gl_file = paths.global_sample_file
    if not gl_file.is_file():
        errors.append(f"Global sample GRIB not found: {gl_file}")

    if errors:
        raise FileNotFoundError(
            "Environment validation failed:\n  - " + "\n  - ".join(errors)
        )


# =============================================================================
# Path Management
# =============================================================================


@dataclass
class EcPointPaths:
    """All resolved directory and file paths for an ecPoint run.

    Paths are organized into:
      - Root directories: top-level project structure
      - Working sub-directories: intermediate files during processing
      - Output sub-directories: final products delivered to the user
      - Calibration files: breakpoint thresholds and FERs tables
      - Sample GRIB files: spatial templates for output GRIB generation
    """

    # Root directories
    main_dir: Path          # Top-level project directory
    scripts_dir: Path       # Scripts and calibration data
    database_dir: Path      # Raw ECMWF forecast input data (GRIB files)
    temp_dir: Path          # Run-specific temporary root
    work_dir: Path          # Working directory for intermediate files
    out_dir: Path           # Final output directory

    # Working sub-directories (temporary, cleaned up after processing)
    wdir_predict: Path      # Pre-computed predictand + predictors per EM
    wdir_pt_rain_cdf: Path  # Bias-corrected CDF per ensemble member (global)
    wdir_grid_rain: Path    # Grid-scale bias-corrected rainfall per EM
    wdir_wt: Path           # Weather type codes per EM
    wdir_percentiles: Path  # Computed percentile fields

    # Output sub-directories (permanent)
    out_pt_perc: Path       # Point bias-corrected percentiles
    out_grid_vals: Path     # Grid-scale bias-corrected values (all EMs concatenated)
    out_wt: Path            # Weather type codes (all EMs concatenated)

    # Calibration files (read-only inputs)
    map_func_dir: Path       # Directory containing mapping function tables
    breakpoints_file: Path   # WT classification breakpoint thresholds CSV
    fers_file: Path          # Forecast Error Ratios CSV

    # Sample GRIB files (spatial templates for output generation)
    sample_dir: Path
    global_sample_file: Path  # Global grid template for percentile GRIB output


def build_paths(cfg: EcPointConfig) -> EcPointPaths:
    """Build all paths from the configuration.

    The directory layout follows the convention:
      main_dir/
        input_db/          — raw ECMWF GRIB inputs
        scripts/comp_files/ — calibration tables and sample GRIB files
        {run_mode}/ecpoint_{var}/{acc}/{version}/
          work/            — temporary intermediate files
          forecasts/       — final output products
    """
    scripts_dir = cfg.scripts_dir or (cfg.main_dir / "scripts")
    database_dir = cfg.main_dir / "input_db"

    # Path component encoding variable, accumulation period, and version
    var_acc_ver = (
        f"ecpoint_{cfg.var_to_postprocess}"
        f"/{cfg.accumulation_str}"
        f"/v{cfg.calibration_version}"
    )
    temp_dir = cfg.main_dir / cfg.run_mode / var_acc_ver
    work_dir = temp_dir / "work"
    out_dir = temp_dir / "forecasts"

    map_func_dir = (
        scripts_dir / "comp_files" / "map_func" / cfg.run_mode / var_acc_ver
    )
    sample_dir = scripts_dir / "comp_files" / "samples" / "ecmwf_ens_18km"

    return EcPointPaths(
        main_dir=cfg.main_dir,
        scripts_dir=scripts_dir,
        database_dir=database_dir,
        temp_dir=temp_dir,
        work_dir=work_dir,
        out_dir=out_dir,
        wdir_predict=work_dir / "predict",
        wdir_pt_rain_cdf=work_dir / "pt_rain_cdf",
        wdir_grid_rain=work_dir / "grid_rain",
        wdir_wt=work_dir / "weather_types",
        wdir_percentiles=work_dir / "percentiles",
        out_pt_perc=out_dir / "pt_bias_corr_perc",
        out_grid_vals=out_dir / "grid_bias_corr_vals",
        out_wt=out_dir / "weather_types",
        map_func_dir=map_func_dir,
        breakpoints_file=map_func_dir / "breakpoints_wt.txt",
        fers_file=map_func_dir / "fers.txt",
        sample_dir=sample_dir,
        global_sample_file=sample_dir / "global" / "global.grib",
    )


def create_filesystem(
    cfg: EcPointConfig, paths: EcPointPaths
) -> None:
    """Create the full directory tree for working and output files.

    Pre-creates all directories organized by date/time/step so that the
    processing loop can write files without checking directory existence.
    Predictor directories additionally include per-ensemble-member subdirs.
    """
    for base_date in _date_range(cfg.base_date_start, cfg.base_date_end):
        bd_str = base_date.strftime("%Y%m%d")

        for base_time in _time_range(
            cfg.base_time_start, cfg.base_time_end, cfg.base_time_disc
        ):
            bt_str = str(base_time).zfill(cfg.num_digits_base_time)
            date_time_key = f"{bd_str}{bt_str}"

            for step_s in range(
                cfg.step_start, cfg.step_final + 1, cfg.step_disc
            ):
                step_f = step_s + cfg.accumulation_hours
                sf_str = str(step_f).zfill(cfg.num_digits_step)
                dt_step = f"{date_time_key}/{sf_str}"

                # Directories that are per date/time/step
                for d in [
                    paths.wdir_pt_rain_cdf,
                    paths.wdir_grid_rain,
                    paths.wdir_wt,
                    paths.wdir_percentiles,
                    paths.out_pt_perc,
                    paths.out_grid_vals,
                    paths.out_wt,
                ]:
                    (d / dt_step).mkdir(parents=True, exist_ok=True)

                # Directories that also split by ensemble member
                for ens_mem in range(
                    cfg.ensemble_member_start,
                    cfg.ensemble_member_end + 1,
                ):
                    em_str = str(ens_mem).zfill(
                        cfg.num_digits_ensemble_member
                    )
                    (paths.wdir_predict / dt_step / em_str).mkdir(
                        parents=True, exist_ok=True
                    )


# =============================================================================
# GRIB Utilities
# =============================================================================


def read_grib(path: Path) -> ekd.FieldList:
    """Read a GRIB file, raising a clear error if it does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"GRIB file not found: {path}")
    return ekd.from_source("file", str(path))


def get_field_values(fieldlist: ekd.FieldList, index: int) -> np.ndarray:
    """Get the values of a single field as a 1-D numpy array."""
    return fieldlist[index].values


def get_all_values(fieldlist: ekd.FieldList) -> list[np.ndarray]:
    """Get values from every field in a fieldlist."""
    return [f.values for f in fieldlist]


def get_metadata(field, key: str):
    """Get a single metadata value from a GRIB field."""
    return field.metadata(key)


def create_fieldlist_from_arrays(
    values_list: list[np.ndarray],
    template_field,
    metadata_overrides: dict | None = None,
) -> ekd.FieldList:
    """Create a new FieldList from numpy arrays using a template field's
    metadata.

    Each array in values_list becomes one GRIB message, inheriting the
    template's metadata with optional overrides applied.
    """
    md = template_field.metadata().override(**(metadata_overrides or {}))
    metadata_list = [md] * len(values_list)
    return ekd.FieldList.from_array(values_list, metadata_list)


def write_grib(fieldlist: ekd.FieldList, path: Path) -> None:
    """Write a fieldlist to a GRIB file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldlist.save(str(path))


def concat_grib_files(input_paths: list[Path], output_path: Path) -> None:
    """Concatenate multiple GRIB files into one."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldlists = [read_grib(p) for p in input_paths]
    combined = fieldlists[0]
    for fl in fieldlists[1:]:
        combined = combined + fl
    write_grib(combined, output_path)


# =============================================================================
# Calibration Data Loading
# =============================================================================


@dataclass
class CalibrationData:
    """Breakpoints and Forecast Error Ratios (FERs) for weather type classification.

    The calibration data comes from two CSV files:
    - breakpoints_wt.txt: defines the predictor threshold ranges that partition
      the atmospheric state into weather types (WTs). Each WT is a row with
      [low, high) bounds for each predictor.
    - fers.txt: for each WT, provides a set of multiplicative correction factors
      (FERs) that transform the raw forecast into a bias-corrected CDF.
      Typically 100 FER columns represent the empirical error distribution.
    """

    weather_type_codes: np.ndarray   # shape (num_wt,) — integer WT identifiers
    breakpoints_low: np.ndarray      # shape (num_wt, num_predictors) — inclusive lower bounds
    breakpoints_high: np.ndarray     # shape (num_wt, num_predictors) — exclusive upper bounds
    fers: np.ndarray                 # shape (num_wt, num_fers) — error ratio values
    num_predictors: int
    num_weather_types: int
    num_fers: int


def load_calibration(paths: EcPointPaths) -> CalibrationData:
    """Load calibration tables from CSV files.

    The breakpoints file has columns: WTcode, pred1_thrL, pred1_thrH, pred2_thrL, ...
    The FERs file has columns: Wtcode, FER1, FER2, ..., FER100
    Both files must have the same number of weather type rows.
    """
    # --- Breakpoints ---
    bp_df = pd.read_csv(paths.breakpoints_file)
    wt_codes = bp_df.iloc[:, 0].values.astype(np.int64)
    num_wt = len(wt_codes)

    # Columns after WTcode come in pairs: (pred_thrL, pred_thrH) for each predictor.
    # Extract them into separate low/high arrays for efficient vectorized comparison.
    threshold_cols = bp_df.columns[1:]
    num_predictors = len(threshold_cols) // 2

    bp_low = np.empty((num_wt, num_predictors), dtype=np.float64)
    bp_high = np.empty((num_wt, num_predictors), dtype=np.float64)
    for p in range(num_predictors):
        bp_low[:, p] = bp_df.iloc[:, 1 + 2 * p].values.astype(np.float64)
        bp_high[:, p] = bp_df.iloc[:, 2 + 2 * p].values.astype(np.float64)

    # --- FERs ---
    fer_df = pd.read_csv(paths.fers_file)
    # First column is Wtcode (matching breakpoints), rest are FER columns.
    # Each FER column represents one quantile of the empirical error distribution.
    num_fers = len(fer_df.columns) - 1
    fers = fer_df.iloc[:, 1:].values.astype(np.float64)

    if fers.shape[0] != num_wt:
        raise ValueError(
            f"FERs table has {fers.shape[0]} rows but breakpoints "
            f"table has {num_wt} weather types"
        )

    return CalibrationData(
        weather_type_codes=wt_codes,
        breakpoints_low=bp_low,
        breakpoints_high=bp_high,
        fers=fers,
        num_predictors=num_predictors,
        num_weather_types=num_wt,
        num_fers=num_fers,
    )


# =============================================================================
# Predictand & Predictor Computation
# =============================================================================


def _weighted_time_average(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray
) -> np.ndarray:
    """Weighted time average over 3 timesteps.

    Formula: (0.5*f1 + f2 + 0.5*f3) / 2
    Used for wind speed at 700 hPa and CAPE.
    """
    return (0.5 * f1 + f2 + 0.5 * f3) / 2.0


def _compute_steps(
    step_start: int, accumulation: int, step_final_max: int
) -> dict:
    """Compute the various forecast step indices needed for predictor calculation.

    For the accumulation window [step_start, step_start + accumulation]:
      - step1, step2, step3 are the start, midpoint, and end of the window,
        used for time-averaging wind speed and CAPE via a weighted average.
    For 24h solar radiation:
      - step1_sr, step2_sr define a 24h window ending at or before step_f.
        If step_f <= 24, the window is [0, 24]; otherwise [step_f-24, step_f].
    """
    step_f = step_start + accumulation
    step1 = step_start
    step2 = step1 + accumulation // 2
    step3 = step_f

    # Solar radiation always uses a 24h accumulation window
    if step_f <= 24:
        step1_sr = 0
        step2_sr = 24
    else:
        step1_sr = step_f - 24
        step2_sr = step_f

    return {
        "step_f": step_f,
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step1_sr": step1_sr,
        "step2_sr": step2_sr,
    }


def _build_input_file_path(
    db_dir: Path, date_time_str: str, step: int, num_digits_step: int,
    param_code: int, extension: str = "grib",
) -> Path:
    """Build the expected input file path for a given parameter and step."""
    step_str = str(step).zfill(num_digits_step)
    return db_dir / date_time_str / step_str / f"{param_code}_{step_str}.{extension}"


def _check_input_files_exist(file_paths: dict[str, Path]) -> None:
    """Check that all required input files exist, with a clear error."""
    missing = [
        f"  {name}: {path}" for name, path in file_paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing input data files:\n" + "\n".join(missing)
        )


def compute_predictors(
    cfg: EcPointConfig,
    paths: EcPointPaths,
    base_date: datetime.date,
    base_time: int,
    step_start: int,
) -> None:
    """Compute predictand + predictors and save per-ensemble-member GRIB files.

    For each ensemble member, derives 6 fields from the raw ECMWF forecast:
      0. Total Precipitation (TP): accumulated TP difference converted m -> mm
      1. Convective Precipitation Ratio (CPR): fraction of TP that is convective
      2. TP (repeated): same as field 0, used as a predictor for WT classification
      3. Wind Speed at 700 hPa: time-averaged from u/v components (synoptic forcing)
      4. CAPE: time-averaged Convective Available Potential Energy (instability)
      5. 24h Solar Radiation: accumulated over 24h, converted J/m^2 -> W/m^2
    These are saved as a single multi-field GRIB per ensemble member.
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    acc = cfg.accumulation_hours

    steps = _compute_steps(step_start, acc, cfg.step_final)
    step_f = steps["step_f"]
    sf_str = str(step_f).zfill(cfg.num_digits_step)

    nds = cfg.num_digits_step
    db_dir = paths.database_dir

    # Build paths for all required input files
    required_files = {
        "tp_step1": _build_input_file_path(db_dir, date_time_str, steps["step1"], nds, 228),
        "tp_step3": _build_input_file_path(db_dir, date_time_str, steps["step3"], nds, 228),
        "cp_step1": _build_input_file_path(db_dir, date_time_str, steps["step1"], nds, 143),
        "cp_step3": _build_input_file_path(db_dir, date_time_str, steps["step3"], nds, 143),
        "u700_step1": _build_input_file_path(db_dir, date_time_str, steps["step1"], nds, 131),
        "u700_step2": _build_input_file_path(db_dir, date_time_str, steps["step2"], nds, 131),
        "u700_step3": _build_input_file_path(db_dir, date_time_str, steps["step3"], nds, 131),
        "v700_step1": _build_input_file_path(db_dir, date_time_str, steps["step1"], nds, 132),
        "v700_step2": _build_input_file_path(db_dir, date_time_str, steps["step2"], nds, 132),
        "v700_step3": _build_input_file_path(db_dir, date_time_str, steps["step3"], nds, 132),
        "cape_step1": _build_input_file_path(db_dir, date_time_str, steps["step1"], nds, 59),
        "cape_step2": _build_input_file_path(db_dir, date_time_str, steps["step2"], nds, 59),
        "cape_step3": _build_input_file_path(db_dir, date_time_str, steps["step3"], nds, 59),
        "sr_step1": _build_input_file_path(db_dir, date_time_str, steps["step1_sr"], nds, 22),
        "sr_step2": _build_input_file_path(db_dir, date_time_str, steps["step2_sr"], nds, 22),
    }
    _check_input_files_exist(required_files)

    logger.info("  Predictand and predictors computation")

    # Read all input GRIB files
    tp_1 = read_grib(required_files["tp_step1"])
    tp_3 = read_grib(required_files["tp_step3"])
    cp_1 = read_grib(required_files["cp_step1"])
    cp_3 = read_grib(required_files["cp_step3"])
    u700_1 = read_grib(required_files["u700_step1"])
    u700_2 = read_grib(required_files["u700_step2"])
    u700_3 = read_grib(required_files["u700_step3"])
    v700_1 = read_grib(required_files["v700_step1"])
    v700_2 = read_grib(required_files["v700_step2"])
    v700_3 = read_grib(required_files["v700_step3"])
    cape_1 = read_grib(required_files["cape_step1"])
    cape_2 = read_grib(required_files["cape_step2"])
    cape_3 = read_grib(required_files["cape_step3"])
    sr_1 = read_grib(required_files["sr_step1"])
    sr_2 = read_grib(required_files["sr_step2"])

    num_em = cfg.num_ensemble_members

    # Process each ensemble member
    logger.info("  Processing %d ensemble members...", num_em)

    for em_idx in range(cfg.ensemble_member_start, cfg.ensemble_member_end + 1):
        em_str = str(em_idx).zfill(cfg.num_digits_ensemble_member)

        # --- Predictand: Total Precipitation ---
        # Accumulated TP difference over the accumulation window, converted m -> mm
        tp_vals = (tp_3[em_idx].values - tp_1[em_idx].values) * 1000.0

        # --- Predictor 1: Convective Precipitation Ratio (CPR) ---
        # Ratio of convective to total precipitation; indicates the convective
        # contribution. Set to 0 where TP is zero to avoid division by zero.
        cp_vals = (cp_3[em_idx].values - cp_1[em_idx].values) * 1000.0
        cpr_vals = np.where(tp_vals > 0, cp_vals / tp_vals, 0.0)

        # --- Predictor 2: Total Precipitation (same as predictand) ---
        # TP is used both as the predictand and as a predictor for WT classification
        tp_pred_vals = tp_vals.copy()

        # --- Predictor 3: Wind Speed at 700 hPa ---
        # Time-averaged u and v wind components at 700 hPa over the accumulation
        # period using a weighted average, then combined into wind speed magnitude.
        # Indicates the strength of synoptic-scale forcing.
        u_avg = _weighted_time_average(
            u700_1[em_idx].values,
            u700_2[em_idx].values,
            u700_3[em_idx].values,
        )
        v_avg = _weighted_time_average(
            v700_1[em_idx].values,
            v700_2[em_idx].values,
            v700_3[em_idx].values,
        )
        wspd700_vals = np.sqrt(u_avg**2 + v_avg**2)

        # --- Predictor 4: CAPE ---
        # Time-averaged Convective Available Potential Energy; measures
        # atmospheric instability and potential for convective precipitation.
        cape_vals = _weighted_time_average(
            cape_1[em_idx].values,
            cape_2[em_idx].values,
            cape_3[em_idx].values,
        )

        # --- Predictor 5: 24h Solar Radiation ---
        # Accumulated solar radiation over 24h, converted J/m^2 -> W/m^2.
        # Acts as a proxy for the diurnal cycle and surface heating.
        sr24h_vals = (sr_2[em_idx].values - sr_1[em_idx].values) / 86400.0

        # Build output: 6 fields per ensemble member
        all_values = [
            tp_vals, cpr_vals, tp_pred_vals,
            wspd700_vals, cape_vals, sr24h_vals,
        ]

        template = tp_1[em_idx]
        metadata_overrides = {
            "type": "pf",
            "perturbationNumber": em_idx,
            "step": step_f,
        }

        fields = []
        for i, vals in enumerate(all_values):
            pred_info = RAINFALL_PREDICTORS[i]
            overrides = {
                **metadata_overrides,
                "paramId": int(pred_info["param_id"]),
            }
            md = template.metadata().override(**overrides)
            fields.append((vals, md))

        values_arr = [f[0] for f in fields]
        metadata_arr = [f[1] for f in fields]
        result = ekd.FieldList.from_array(values_arr, metadata_arr)

        out_path = (
            paths.wdir_predict / date_time_str / sf_str / em_str
            / f"predict_{em_str}.grib"
        )
        write_grib(result, out_path)

    logger.info("  Predictors saved for %s step %s", date_time_str, sf_str)


# =============================================================================
# Post-Processing
# =============================================================================


def classify_weather_types(
    predictors: list[np.ndarray],
    calibration: CalibrationData,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each grid point into a weather type based on predictor values.

    The weather type (WT) classification uses a decision-tree-like approach:
    each WT defines a hyper-rectangular region in predictor space via
    [low, high) breakpoint thresholds. A grid point is assigned the first WT
    whose thresholds are satisfied by all predictors simultaneously.

    The implementation is fully vectorized using numpy broadcasting to avoid
    per-grid-point Python loops, which is critical for performance on global
    grids (~2M points).

    Returns:
        wt_codes: 1-D array of WT code per grid point (int).
        wt_indices: 1-D array of WT index (0-based) per grid point, or -1 if
                    no weather type matched.
    """
    n_grid = len(predictors[0])
    n_wt = calibration.num_weather_types
    n_pred = calibration.num_predictors

    # Stack predictors into a 2-D array: shape (n_pred, n_grid)
    pred_stack = np.stack(predictors[:n_pred], axis=0)

    # Use broadcasting to compare all grid points against all WTs at once.
    # Expand dimensions so that numpy broadcasts across (n_wt, n_pred, n_grid):
    #   breakpoints: (n_wt, n_pred) -> (n_wt, n_pred, 1)
    #   predictors:  (n_pred, n_grid) -> (1, n_pred, n_grid)
    low = calibration.breakpoints_low[:, :, np.newaxis]   # (n_wt, n_pred, 1)
    high = calibration.breakpoints_high[:, :, np.newaxis]  # (n_wt, n_pred, 1)
    pred_exp = pred_stack[np.newaxis, :, :]                # (1, n_pred, n_grid)

    # A grid point matches a WT when ALL predictors are within [low, high)
    in_range = (pred_exp >= low) & (pred_exp < high)  # (n_wt, n_pred, n_grid)
    all_match = np.all(in_range, axis=1)  # (n_wt, n_grid) — True where all predictors match

    # For each grid point, select the first matching WT (lowest index).
    # np.argmax returns the index of the first True along the WT axis.
    any_match = np.any(all_match, axis=0)  # (n_grid,) — True if any WT matched
    wt_idx = np.argmax(all_match, axis=0)  # (n_grid,) — index of first match

    # Unmatched grid points get index -1 and code 0
    wt_indices = np.where(any_match, wt_idx, -1)
    wt_codes = np.where(
        any_match,
        calibration.weather_type_codes[wt_idx],
        0,
    )

    return wt_codes.astype(np.int64), wt_indices


def apply_fers(
    predictand: np.ndarray,
    wt_indices: np.ndarray,
    calibration: CalibrationData,
) -> list[np.ndarray]:
    """Apply Forecast Error Ratios to create the bias-corrected CDF.

    The FER approach corrects systematic model biases by applying empirically
    derived multiplicative factors. For each grid point:
      corrected_value[i] = raw_forecast * (FER[wt, i] + 1)

    The FER values represent fractional errors: FER = (obs - fcst) / fcst,
    so (FER + 1) = obs/fcst is the correction factor. Each FER column
    corresponds to one quantile of the error distribution, producing an
    ensemble of corrected values that forms a local CDF (Cumulative
    Distribution Function) at that grid point.

    Returns a list of n_fers arrays, each of shape (n_grid,), representing
    the n_fers quantiles of the corrected CDF.
    """
    n_grid = len(predictand)
    n_fers = calibration.num_fers

    # Look up the FER row for each grid point based on its weather type.
    # Grid points with no WT match (wt_indices == -1) get FER = 0,
    # meaning factor = 1, so the raw forecast is returned unchanged.
    valid_mask = wt_indices >= 0
    fer_per_point = np.zeros((n_grid, n_fers), dtype=np.float64)
    fer_per_point[valid_mask] = calibration.fers[wt_indices[valid_mask]]

    # Apply correction: raw_forecast * (FER + 1) for each FER quantile
    factors = fer_per_point + 1.0  # (n_grid, n_fers)
    cdf_values = predictand[:, np.newaxis] * factors  # (n_grid, n_fers)

    return [cdf_values[:, i] for i in range(n_fers)]


def postprocess_ensemble(
    cfg: EcPointConfig,
    paths: EcPointPaths,
    calibration: CalibrationData,
    base_date: datetime.date,
    base_time: int,
    step_start: int,
) -> None:
    """Post-process all ensemble members: classify WTs, apply FERs, save outputs.

    For each ensemble member, the pipeline:
      a. Reads the pre-computed predictand and predictors from GRIB
      b. Classifies each grid point into a weather type (WT) using breakpoints
      c. Applies FERs to generate a bias-corrected CDF at each grid point
      d. Saves the grid-scale bias-corrected rainfall (mean of CDF values)
      e. Saves the weather type codes for diagnostic purposes
      f. Saves the full CDF for later percentile computation across all members
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    step_f = step_start + cfg.accumulation_hours
    sf_str = str(step_f).zfill(cfg.num_digits_step)

    logger.info("  Post-processing ensemble members")

    for em_idx in range(cfg.ensemble_member_start, cfg.ensemble_member_end + 1):
        em_str = str(em_idx).zfill(cfg.num_digits_ensemble_member)
        logger.info("    EM n.%s", em_str)

        # a. Read predictand and predictors
        pred_path = (
            paths.wdir_predict / date_time_str / sf_str / em_str
            / f"predict_{em_str}.grib"
        )
        pred_data = read_grib(pred_path)
        template_field = pred_data[0]

        all_values = get_all_values(pred_data)
        predictand = all_values[0]  # first field = predictand (TP)

        # Threshold small values to zero (packing-precision artifacts)
        predictand = np.where(
            predictand >= cfg.min_predictand_value, predictand, 0.0
        )

        # Predictors are fields 2 onward (CPR, TP, wspd700, CAPE, SR24h)
        predictors = all_values[1:]

        # b. Classify weather types
        wt_codes, wt_indices = classify_weather_types(predictors, calibration)

        # c. Apply FERs to create CDF
        cdf_list = apply_fers(predictand, wt_indices, calibration)

        # d. Save grid-bias corrected rainfall (mean of CDF values)
        grid_bc_vals = np.mean(np.stack(cdf_list, axis=1), axis=1)
        grid_bc_fl = create_fieldlist_from_arrays(
            [grid_bc_vals], template_field
        )
        grid_bc_path = (
            paths.wdir_grid_rain / date_time_str / sf_str
            / f"grid_rain_{em_str}.grib"
        )
        write_grib(grid_bc_fl, grid_bc_path)

        # e. Save WT codes (with a special "no-WT" sentinel for sub-threshold values).
        # Grid points with TP below the minimum threshold are assigned a code of
        # all 9s (e.g., 99999 for 5 predictors) to indicate "no classification".
        no_wt_code = int("9" * calibration.num_predictors)
        wt_output = np.where(
            predictand < cfg.min_predictand_value,
            no_wt_code,
            wt_codes,
        ).astype(np.float64)
        wt_fl = create_fieldlist_from_arrays([wt_output], template_field)
        wt_path = (
            paths.wdir_wt / date_time_str / sf_str
            / f"wt_{em_str}.grib"
        )
        write_grib(wt_fl, wt_path)

        # f. Save point rainfall CDF (global field)
        cdf_fl = create_fieldlist_from_arrays(cdf_list, template_field)
        cdf_path = (
            paths.wdir_pt_rain_cdf / date_time_str / sf_str
            / f"pt_rain_cdf_{em_str}.grib"
        )
        write_grib(cdf_fl, cdf_path)

    logger.info("  Post-processing complete for %s step %s", date_time_str, sf_str)


# =============================================================================
# Percentile Computation
# =============================================================================


def compute_percentiles(
    cfg: EcPointConfig,
    paths: EcPointPaths,
    base_date: datetime.date,
    base_time: int,
    step_start: int,
) -> None:
    """Compute percentiles from global CDFs across all ensemble members.

    Collects all CDF fields (num_ensemble_members * num_fers fields total)
    into a single matrix and computes the requested percentiles along the
    ensemble/FER axis at each grid point. This produces the final
    probabilistic point-rainfall forecast: each percentile field represents
    a quantile of the bias-corrected rainfall distribution.
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    step_f = step_start + cfg.accumulation_hours
    sf_str = str(step_f).zfill(cfg.num_digits_step)

    logger.info("  Percentiles computation")

    # Read CDFs for all ensemble members (full global grid)
    all_cdf_values = []
    for em_idx in range(
        cfg.ensemble_member_start, cfg.ensemble_member_end + 1
    ):
        em_str = str(em_idx).zfill(cfg.num_digits_ensemble_member)
        cdf_path = (
            paths.wdir_pt_rain_cdf / date_time_str / sf_str
            / f"pt_rain_cdf_{em_str}.grib"
        )
        cdf_data = read_grib(cdf_path)
        all_cdf_values.extend(get_all_values(cdf_data))

    # Stack all CDF fields into a matrix: shape (n_total_cdf_fields, n_grid_global)
    # where n_total_cdf_fields = num_ensemble_members * num_fers
    cdf_matrix = np.stack(all_cdf_values, axis=0)

    # Compute percentiles along the field axis (axis=0).
    # For each grid point, this produces the requested quantiles of the
    # combined ensemble+FER distribution.
    # Result shape: (num_percentiles, n_grid_global)
    global_percentiles = np.percentile(
        cdf_matrix, cfg.percentiles, axis=0
    )

    # Create output GRIB using a global sample file as the spatial template.
    # Each percentile is stored as a separate GRIB message, encoded as
    # perturbed forecast members (perturbationNumber = percentile index).
    gl_sample = read_grib(paths.global_sample_file)
    gl_template = gl_sample[0]
    num_perc = len(cfg.percentiles)

    pp_info = cfg.variable_info
    values_list = [global_percentiles[i] for i in range(num_perc)]
    metadata_list = []
    for i in range(num_perc):
        md = gl_template.metadata().override(
            **{
                "class": "od",
                "stream": "enfo",
                "type": "pf",
                "expver": "0001",
                "paramId": int(pp_info["pp_code"]),
                "level": pp_info["pp_level"],
                "levtype": pp_info["pp_level_type"],
                "numberOfForecastsInEnsemble": num_perc,
                "perturbationNumber": i + 1,
            }
        )
        metadata_list.append(md)

    result = ekd.FieldList.from_array(values_list, metadata_list)

    out_path = (
        paths.wdir_percentiles / date_time_str / sf_str
        / "percentiles.grib"
    )
    write_grib(result, out_path)
    logger.info("  Percentiles saved to %s", out_path)


# =============================================================================
# Output File Management
# =============================================================================


def move_outputs(
    cfg: EcPointConfig,
    paths: EcPointPaths,
    base_date: datetime.date,
    base_time: int,
    step_start: int,
) -> None:
    """Move output files from working directories to the final output database.

    Three types of output are produced per date/time/step:
      1. Percentile GRIB: the probabilistic point-rainfall forecast (1 file)
      2. Grid-bias-corrected rainfall: all ensemble members concatenated (1 file)
      3. Weather types: all ensemble members concatenated (1 file)
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    step_f = step_start + cfg.accumulation_hours
    sf_str = str(step_f).zfill(cfg.num_digits_step)
    acc_str = cfg.accumulation_str

    logger.info("  Moving output files to output database")

    # 1. Percentiles (single file move)
    src_cdf = (
        paths.wdir_percentiles / date_time_str / sf_str / "percentiles.grib"
    )
    dst_cdf = (
        paths.out_pt_perc / date_time_str / sf_str
        / f"pt_bc_perc_{acc_str}_{bd_str}_{bt_str}_{sf_str}.grib"
    )
    dst_cdf.parent.mkdir(parents=True, exist_ok=True)
    if src_cdf.is_file():
        shutil.move(str(src_cdf), str(dst_cdf))

    # 2. Grid-bias corrected rainfall (concatenate all EM files)
    grid_rain_dir = paths.wdir_grid_rain / date_time_str / sf_str
    grid_files = sorted(grid_rain_dir.glob("grid_rain_*.grib"))
    if grid_files:
        dst_grid = (
            paths.out_grid_vals / date_time_str / sf_str
            / f"grid_bc_vals_{acc_str}_{bd_str}_{bt_str}_{sf_str}.grib"
        )
        concat_grib_files(grid_files, dst_grid)

    # 3. Weather types (concatenate all EM files)
    wt_dir = paths.wdir_wt / date_time_str / sf_str
    wt_files = sorted(wt_dir.glob("wt_*.grib"))
    if wt_files:
        dst_wt = (
            paths.out_wt / date_time_str / sf_str
            / f"wt_{acc_str}_{bd_str}_{bt_str}_{sf_str}.grib"
        )
        concat_grib_files(wt_files, dst_wt)


# =============================================================================
# Helpers
# =============================================================================


def _date_range(
    start: datetime.date, end: datetime.date
) -> list[datetime.date]:
    """Generate a list of dates from start to end (inclusive)."""
    days = (end - start).days + 1
    return [start + datetime.timedelta(days=i) for i in range(days)]


def _time_range(start: int, end: int, disc: int) -> list[int]:
    """Generate a list of base times from start to end with step disc."""
    return list(range(start, end + 1, disc))


def _step_range(start: int, final: int, disc: int) -> list[int]:
    """Generate a list of step-start values."""
    return list(range(start, final + 1, disc))


# =============================================================================
# Main Orchestrator
# =============================================================================


def run_ecpoint(cfg: EcPointConfig) -> None:
    """Run the full ecPoint post-processing pipeline.

    The pipeline has two phases:
      Phase 1 — Environment setup: build paths, validate that all required
        calibration and sample files exist, create the output directory tree,
        and load the calibration tables (breakpoints + FERs).
      Phase 2 — Post-processing: iterate over all (date, time, step)
        combinations. For each combination, compute predictors from raw ECMWF
        fields, classify weather types, apply FERs to generate CDFs, compute
        percentiles across all ensemble members, and move outputs to the
        final database.
    After processing, temporary working directories are cleaned up.
    """
    logger.info("=" * 60)
    logger.info("ecPoint — Forecasts for Point Values")
    logger.info("  Variable: %s", cfg.var_to_postprocess)
    logger.info("  Calibration Version: %s", cfg.calibration_version)
    logger.info("  Run Mode: %s", cfg.run_mode)
    logger.info("=" * 60)

    # Phase 1: Environment setup
    logger.info("")
    logger.info("Phase 1: Setting up environment")
    logger.info("  Building paths...")
    paths = build_paths(cfg)

    logger.info("  Validating environment...")
    validate_environment(cfg, paths)

    logger.info("  Creating file system...")
    create_filesystem(cfg, paths)

    logger.info("  Loading calibration data...")
    calibration = load_calibration(paths)
    logger.info(
        "  Loaded %d weather types, %d FERs",
        calibration.num_weather_types,
        calibration.num_fers,
    )

    # Phase 2: Post-processing
    logger.info("")
    logger.info("Phase 2: Post-processing")

    dates = _date_range(cfg.base_date_start, cfg.base_date_end)
    times = _time_range(
        cfg.base_time_start, cfg.base_time_end, cfg.base_time_disc
    )
    steps = _step_range(cfg.step_start, cfg.step_final, cfg.step_disc)

    total_iterations = len(dates) * len(times) * len(steps)
    current = 0

    for base_date in dates:
        for base_time in times:
            for step_s in steps:
                current += 1
                step_f = step_s + cfg.accumulation_hours
                logger.info("")
                logger.info(
                    "[%d/%d] %s - %02d UTC - (t+%d, t+%d)",
                    current,
                    total_iterations,
                    base_date.strftime("%Y%m%d"),
                    base_time,
                    step_s,
                    step_f,
                )

                logger.info("Step 1: Predictand and predictors computation")
                compute_predictors(cfg, paths, base_date, base_time, step_s)

                logger.info("Step 2: Post-processing ensemble members")
                postprocess_ensemble(
                    cfg, paths, calibration, base_date, base_time, step_s
                )

                logger.info("Step 3: Percentiles computation")
                compute_percentiles(
                    cfg, paths, base_date, base_time, step_s
                )

                logger.info("Step 4: Moving output files")
                move_outputs(cfg, paths, base_date, base_time, step_s)

    # Cleanup
    logger.info("")
    logger.info("Cleaning up temporary directories...")
    if paths.database_dir.exists():
        shutil.rmtree(paths.database_dir, ignore_errors=True)
    if paths.work_dir.exists():
        shutil.rmtree(paths.work_dir, ignore_errors=True)

    logger.info("Post-processing completed!")


# =============================================================================
# CLI Entry Point
# =============================================================================


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a JSON configuration file.",
)
@click.option("--var", "var_to_postprocess", default="rainfall")
@click.option("--acc", "accumulation_hours", type=int, default=12)
@click.option("--cal-version", "calibration_version", default="1.0.0")
@click.option("--run-mode", default="dev")
@click.option(
    "--date-start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
)
@click.option(
    "--date-end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
)
@click.option("--time-start", type=int, default=0)
@click.option("--time-end", type=int, default=0)
@click.option("--time-disc", type=int, default=12)
@click.option("--step-start", type=int, default=0)
@click.option("--step-final", type=int, default=114)
@click.option("--step-disc", type=int, default=6)
@click.option("--ens-start", "ensemble_member_start", type=int, default=0)
@click.option("--ens-end", "ensemble_member_end", type=int, default=50)
@click.option("--main-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "-v", "--verbose", count=True, help="Increase verbosity (-v, -vv)."
)
def main(
    config_path,
    var_to_postprocess,
    accumulation_hours,
    calibration_version,
    run_mode,
    date_start,
    date_end,
    time_start,
    time_end,
    time_disc,
    step_start,
    step_final,
    step_disc,
    ensemble_member_start,
    ensemble_member_end,
    main_dir,
    verbose,
):
    """ecPoint: Post-processing system for NWP ensemble forecasts."""
    # Configure logging
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config from file or CLI options.
    # CLI options override values from the config file. Only non-default
    # values are passed as overrides to avoid masking file-provided values.
    overrides = {}
    if var_to_postprocess != "rainfall":
        overrides["var_to_postprocess"] = var_to_postprocess
    if accumulation_hours != 12:
        overrides["accumulation_hours"] = accumulation_hours
    if calibration_version != "1.0.0":
        overrides["calibration_version"] = calibration_version
    if run_mode != "dev":
        overrides["run_mode"] = run_mode
    if date_start is not None:
        overrides["base_date_start"] = date_start.date()
    if date_end is not None:
        overrides["base_date_end"] = date_end.date()
    if time_start != 0:
        overrides["base_time_start"] = time_start
    if time_end != 0:
        overrides["base_time_end"] = time_end
    if time_disc != 12:
        overrides["base_time_disc"] = time_disc
    if step_start != 0:
        overrides["step_start"] = step_start
    if step_final != 114:
        overrides["step_final"] = step_final
    if step_disc != 6:
        overrides["step_disc"] = step_disc
    if ensemble_member_start != 0:
        overrides["ensemble_member_start"] = ensemble_member_start
    if ensemble_member_end != 50:
        overrides["ensemble_member_end"] = ensemble_member_end
    if main_dir is not None:
        overrides["main_dir"] = main_dir

    cfg = load_config(config_path, **overrides)
    run_ecpoint(cfg)


if __name__ == "__main__":
    main()
