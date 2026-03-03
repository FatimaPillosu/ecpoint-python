"""
ecPoint — Post-processing system for NWP ensemble forecasts.

Converts raw ensemble model output into bias-corrected point rainfall
forecasts using weather-type-dependent calibration (Forecast Error Ratios).

This is a Python port of the original Metview implementation by ECMWF,
using earthkit-data for GRIB I/O and numpy for numerical operations.

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
# Section 1: Configuration (replaces InParam.mv + Check_Set_InParam.mv)
# =============================================================================

# Lookup table: variable name -> (predictand param code, level type, level,
#                                  minimum reliable value)
VARIABLE_REGISTRY: dict[str, dict] = {
    "Rainfall": {
        "predictand_code": 228.128,
        "level_type": "sfc",
        "level": 0,
        "min_predictand_value": 0.04,
        "pp_code": 82.128,
        "pp_level_type": "sfc",
        "pp_level": 0,
        "valid_accumulations": [12],
    },
}

# Predictor definitions for Rainfall (paramId, levelType, level)
RAINFALL_PREDICTORS = [
    {"name": "tp", "param_id": 228.128, "level_type": "sfc", "level": 0},
    {"name": "cpr", "param_id": 143.128, "level_type": "sfc", "level": 0},
    {"name": "tp_repeat", "param_id": 228.128, "level_type": "sfc", "level": 0},
    {"name": "wspd700", "param_id": 10.128, "level_type": "pl", "level": 700},
    {"name": "cape", "param_id": 59.128, "level_type": "sfc", "level": 0},
    {"name": "sr24h", "param_id": 228022, "level_type": "sfc", "level": 0},
]


class EcPointConfig(BaseModel):
    """All input parameters for an ecPoint run.

    Replaces InParam.mv (parameter values) and Check_Set_InParam.mv (derived
    parameters and validation).
    """

    # Variable to post-process
    var_to_postprocess: Literal["Rainfall"] = "Rainfall"
    accumulation_hours: int = Field(default=12, ge=0)
    calibration_version: str = "1.0.0"
    run_mode: str = "Dev"

    # Forecast date/time/step ranges
    base_date_start: datetime.date = datetime.date(2020, 2, 19)
    base_date_end: datetime.date = datetime.date(2020, 2, 19)
    base_time_start: int = Field(default=0, ge=0, le=23)
    base_time_end: int = Field(default=0, ge=0, le=23)
    base_time_disc: int = Field(default=12, gt=0)
    step_start: int = Field(default=0, ge=0)
    step_final: int = Field(default=114, ge=0)
    step_disc: int = Field(default=6, gt=0)

    # Ensemble members
    ensemble_member_start: int = Field(default=0, ge=0)
    ensemble_member_end: int = Field(default=50, ge=0)

    # Sub-areas
    num_sub_areas: int = 10

    # Percentiles (1-99)
    percentiles: list[int] = Field(
        default_factory=lambda: list(range(1, 100))
    )

    # Directory paths
    main_dir: Path = Path.home()
    scripts_dir: Path | None = None  # defaults to main_dir / "Scripts"

    # Formatting digits
    num_digits_base_time: int = 2
    num_digits_step: int = 3
    num_digits_acc: int = 3
    num_digits_ensemble_member: int = 2
    num_digits_sub_area: int = 2

    # Precision
    float_precision: Literal["float32", "float64"] = "float32"

    # --- Validators ---

    @field_validator("num_sub_areas")
    @classmethod
    def _check_num_sub_areas(cls, v: int) -> int:
        if v not in {5, 10, 20}:
            raise ValueError(
                f"num_sub_areas must be 5, 10 or 20, got {v}"
            )
        return v

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
    def sub_area_str(self) -> str:
        return str(self.num_sub_areas).zfill(self.num_digits_sub_area)

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
# Section 2: Validation (replaces the "# to complete" in Check_Set_InParam.mv)
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

    sa_file = paths.sub_area_sample_file
    if not sa_file.is_file():
        errors.append(f"Sub-area sample GRIB not found: {sa_file}")

    if errors:
        raise FileNotFoundError(
            "Environment validation failed:\n  - " + "\n  - ".join(errors)
        )


# =============================================================================
# Section 3: Path Management (replaces FileSystem.mv)
# =============================================================================


@dataclass
class EcPointPaths:
    """All resolved directory and file paths for an ecPoint run.

    Replaces the global path variables scattered across FileSystem.mv and
    other Metview scripts.
    """

    # Root directories
    main_dir: Path
    scripts_dir: Path
    database_dir: Path
    temp_dir: Path
    work_dir: Path
    out_dir: Path

    # Working sub-directories
    wdir_predict: Path       # 10_Predict_AllREM_GL
    wdir_ens_pt_rain: Path   # 21_EnsPtRain_SingleREM_SA
    wdir_grid_rain: Path     # 22_GridRain_ALLREM_GL
    wdir_wt: Path            # 23_WT_AllREM_GL
    wdir_cdf_sa: Path        # 30_PtRainCDF_SA
    wdir_cdf_gl: Path        # 40_PtRainCDF_GL

    # Output sub-directories
    out_pt_perc: Path        # Pt_BiasCorr_RainPERC
    out_grid_vals: Path      # Grid_BiasCorr_RainVALS
    out_wt: Path             # WT

    # Calibration files
    map_func_dir: Path
    breakpoints_file: Path
    fers_file: Path

    # Sample GRIB files
    sample_dir: Path
    global_sample_file: Path
    sub_area_sample_file: Path


def build_paths(cfg: EcPointConfig) -> EcPointPaths:
    """Build all paths from the configuration."""
    scripts_dir = cfg.scripts_dir or (cfg.main_dir / "Scripts")
    database_dir = cfg.main_dir / "InputDB"

    var_acc_ver = (
        f"ecPoint_{cfg.var_to_postprocess}"
        f"/{cfg.accumulation_str}"
        f"/Vers{cfg.calibration_version}"
    )
    temp_dir = cfg.main_dir / cfg.run_mode / var_acc_ver
    work_dir = temp_dir / "WorkDir"
    out_dir = temp_dir / "Forecasts"

    map_func_dir = (
        scripts_dir / "CompFiles" / "MapFunc" / cfg.run_mode / var_acc_ver
    )
    sample_dir = scripts_dir / "CompFiles" / "Samples" / "ECMWF_ENS_18km"

    return EcPointPaths(
        main_dir=cfg.main_dir,
        scripts_dir=scripts_dir,
        database_dir=database_dir,
        temp_dir=temp_dir,
        work_dir=work_dir,
        out_dir=out_dir,
        wdir_predict=work_dir / "10_Predict_AllREM_GL",
        wdir_ens_pt_rain=work_dir / "21_EnsPtRain_SingleREM_SA",
        wdir_grid_rain=work_dir / "22_GridRain_ALLREM_GL",
        wdir_wt=work_dir / "23_WT_AllREM_GL",
        wdir_cdf_sa=work_dir / "30_PtRainCDF_SA",
        wdir_cdf_gl=work_dir / "40_PtRainCDF_GL",
        out_pt_perc=out_dir / "Pt_BiasCorr_RainPERC",
        out_grid_vals=out_dir / "Grid_BiasCorr_RainVALS",
        out_wt=out_dir / "WT",
        map_func_dir=map_func_dir,
        breakpoints_file=map_func_dir / "BreakPointsWT.txt",
        fers_file=map_func_dir / "FERs.txt",
        sample_dir=sample_dir,
        global_sample_file=sample_dir / "Global" / "Global.grib",
        sub_area_sample_file=(
            sample_dir / "SubArea" / cfg.sub_area_str / "SubArea.grib"
        ),
    )


def create_filesystem(
    cfg: EcPointConfig, paths: EcPointPaths
) -> None:
    """Create the full directory tree for working and output files.

    Replaces the nested shell("mkdir -p ...") calls in FileSystem.mv.
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
                    paths.wdir_grid_rain,
                    paths.wdir_wt,
                    paths.wdir_cdf_gl,
                    paths.out_pt_perc,
                    paths.out_grid_vals,
                    paths.out_wt,
                ]:
                    (d / dt_step).mkdir(parents=True, exist_ok=True)

                # Directories that also split by sub-area
                for sub_area in range(1, cfg.num_sub_areas + 1):
                    sa_str = str(sub_area).zfill(cfg.num_digits_sub_area)
                    for d in [paths.wdir_ens_pt_rain, paths.wdir_cdf_sa]:
                        (d / dt_step / sa_str).mkdir(
                            parents=True, exist_ok=True
                        )

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
# Section 4: GRIB Utilities (new — wraps earthkit-data)
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
    """Concatenate multiple GRIB files into one (replaces grib_copy)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldlists = [read_grib(p) for p in input_paths]
    combined = fieldlists[0]
    for fl in fieldlists[1:]:
        combined = combined + fl
    write_grib(combined, output_path)


# =============================================================================
# Section 5: Calibration Data Loading (extracted from EnsPtRain_GridRain_WT.mv)
# =============================================================================


@dataclass
class CalibrationData:
    """Breakpoints and Forecast Error Ratios for weather type classification.

    Loaded from BreakPointsWT.txt and FERs.txt.
    """

    weather_type_codes: np.ndarray   # shape (num_wt,)
    breakpoints_low: np.ndarray      # shape (num_wt, num_predictors)
    breakpoints_high: np.ndarray     # shape (num_wt, num_predictors)
    fers: np.ndarray                 # shape (num_wt, num_fers)
    num_predictors: int
    num_weather_types: int
    num_fers: int


def load_calibration(paths: EcPointPaths) -> CalibrationData:
    """Load calibration tables from CSV files.

    Replaces the read_table() calls in EnsPtRain_GridRain_WT.mv.
    """
    # --- Breakpoints ---
    bp_df = pd.read_csv(paths.breakpoints_file)
    wt_codes = bp_df.iloc[:, 0].values.astype(np.int64)
    num_wt = len(wt_codes)

    # Columns after WTcode are pairs: (pred_thrL, pred_thrH) for each predictor
    threshold_cols = bp_df.columns[1:]
    num_predictors = len(threshold_cols) // 2

    bp_low = np.empty((num_wt, num_predictors), dtype=np.float64)
    bp_high = np.empty((num_wt, num_predictors), dtype=np.float64)
    for p in range(num_predictors):
        bp_low[:, p] = bp_df.iloc[:, 1 + 2 * p].values.astype(np.float64)
        bp_high[:, p] = bp_df.iloc[:, 2 + 2 * p].values.astype(np.float64)

    # --- FERs ---
    fer_df = pd.read_csv(paths.fers_file)
    # First column is Wtcode, rest are FER1..FER100
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
# Section 6: Predictand & Predictor Computation (replaces Predict.mv)
# =============================================================================


def _trapezoidal_time_average(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray
) -> np.ndarray:
    """Trapezoidal-rule time average over 3 timesteps.

    Formula: (0.5*f1 + f2 + 0.5*f3) / 2
    Used for wind speed at 700 hPa and CAPE.
    """
    return (0.5 * f1 + f2 + 0.5 * f3) / 2.0


def _compute_steps(
    step_start: int, accumulation: int, step_final_max: int
) -> dict:
    """Compute the various step indices needed for predictor calculation.

    Returns a dict with step1, step2, step3 (for the accumulation period)
    and step1_sr, step2_sr (for 24h solar radiation).
    """
    step_f = step_start + accumulation
    step1 = step_start
    step2 = step1 + accumulation // 2
    step3 = step_f

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

    Replaces Predict.mv.
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

        # --- Predictand: Total Precipitation (m -> mm) ---
        tp_vals = (tp_3[em_idx].values - tp_1[em_idx].values) * 1000.0

        # --- Predictor 1: Convective Precipitation Ratio ---
        cp_vals = (cp_3[em_idx].values - cp_1[em_idx].values) * 1000.0
        cpr_vals = np.where(tp_vals > 0, cp_vals / tp_vals, 0.0)

        # --- Predictor 2: Total Precipitation (same as predictand) ---
        tp_pred_vals = tp_vals.copy()

        # --- Predictor 3: Wind Speed at 700 hPa ---
        u_avg = _trapezoidal_time_average(
            u700_1[em_idx].values,
            u700_2[em_idx].values,
            u700_3[em_idx].values,
        )
        v_avg = _trapezoidal_time_average(
            v700_1[em_idx].values,
            v700_2[em_idx].values,
            v700_3[em_idx].values,
        )
        wspd700_vals = np.sqrt(u_avg**2 + v_avg**2)

        # --- Predictor 4: CAPE ---
        cape_vals = _trapezoidal_time_average(
            cape_1[em_idx].values,
            cape_2[em_idx].values,
            cape_3[em_idx].values,
        )

        # --- Predictor 5: Daily Solar Radiation ---
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
            / f"Predict_{em_str}.grib"
        )
        write_grib(result, out_path)

    logger.info("  Predictors saved for %s step %s", date_time_str, sf_str)


# =============================================================================
# Section 7: Post-Processing (replaces EnsPtRain_GridRain_WT.mv)
# =============================================================================


def classify_weather_types(
    predictors: list[np.ndarray],
    calibration: CalibrationData,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each grid point into a weather type based on predictor values.

    For each weather type, checks whether all predictor values fall within
    the corresponding breakpoint thresholds [low, high).

    Returns:
        wt_codes: 1-D array of WT code per grid point (int).
        wt_indices: 1-D array of WT index (0-based) per grid point, or -1 if
                    no weather type matched.
    """
    n_grid = len(predictors[0])
    n_wt = calibration.num_weather_types
    n_pred = calibration.num_predictors

    # Stack predictors: shape (n_pred, n_grid)
    pred_stack = np.stack(predictors[:n_pred], axis=0)

    # Broadcast comparison: shape (n_wt, n_pred, n_grid)
    # bp_low/high: (n_wt, n_pred) -> (n_wt, n_pred, 1)
    low = calibration.breakpoints_low[:, :, np.newaxis]   # (n_wt, n_pred, 1)
    high = calibration.breakpoints_high[:, :, np.newaxis]  # (n_wt, n_pred, 1)
    pred_exp = pred_stack[np.newaxis, :, :]                # (1, n_pred, n_grid)

    # Check: low <= predictor < high for all predictors
    in_range = (pred_exp >= low) & (pred_exp < high)  # (n_wt, n_pred, n_grid)
    all_match = np.all(in_range, axis=1)  # (n_wt, n_grid)

    # For each grid point, find the first matching WT
    # Use argmax on the match matrix (first True along WT axis)
    # If no WT matches, all_match will be all-False for that column
    any_match = np.any(all_match, axis=0)  # (n_grid,)
    wt_idx = np.argmax(all_match, axis=0)  # (n_grid,) — index of first match

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

    For each grid point, the predictand is multiplied by (FER[wt, i] + 1)
    for each FER column, producing n_fers new values.

    Returns a list of n_fers arrays, each of shape (n_grid,).
    """
    n_grid = len(predictand)
    n_fers = calibration.num_fers

    # Build the FER matrix for each grid point: shape (n_grid, n_fers)
    # For grid points with no WT match (wt_indices == -1), FER = 0 -> factor = 1
    valid_mask = wt_indices >= 0
    fer_per_point = np.zeros((n_grid, n_fers), dtype=np.float64)
    fer_per_point[valid_mask] = calibration.fers[wt_indices[valid_mask]]

    # CDF values: predictand * (FER + 1)
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

    Replaces EnsPtRain_GridRain_WT.mv.
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    step_f = step_start + cfg.accumulation_hours
    sf_str = str(step_f).zfill(cfg.num_digits_step)

    logger.info("  Post-processing ensemble members")

    # Read sub-area sample for later use
    sa_sample = read_grib(paths.sub_area_sample_file)
    n_grid_sa = len(sa_sample[0].values)

    for em_idx in range(cfg.ensemble_member_start, cfg.ensemble_member_end + 1):
        em_str = str(em_idx).zfill(cfg.num_digits_ensemble_member)
        logger.info("    EM n.%s", em_str)

        # a. Read predictand and predictors
        pred_path = (
            paths.wdir_predict / date_time_str / sf_str / em_str
            / f"Predict_{em_str}.grib"
        )
        pred_data = read_grib(pred_path)
        template_field = pred_data[0]

        # Extract metadata from the first field
        grib_base_date = get_metadata(pred_data[0], "dataDate")
        grib_base_time = get_metadata(pred_data[0], "dataTime")
        grib_step = get_metadata(pred_data[0], "stepRange")
        grib_em_number = get_metadata(pred_data[0], "number")

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
            / f"GridRain_{em_str}.grib"
        )
        write_grib(grid_bc_fl, grid_bc_path)

        # e. Save WT codes (with "no-WT" code for sub-threshold values)
        no_wt_code = int("9" * calibration.num_predictors)
        wt_output = np.where(
            predictand < cfg.min_predictand_value,
            no_wt_code,
            wt_codes,
        ).astype(np.float64)
        wt_fl = create_fieldlist_from_arrays([wt_output], template_field)
        wt_path = (
            paths.wdir_wt / date_time_str / sf_str
            / f"WT_{em_str}.grib"
        )
        write_grib(wt_fl, wt_path)

        # f. Save point rainfall CDF split by sub-areas
        pp_info = cfg.variable_info
        sa_metadata_overrides = {
            "class": "od",
            "stream": "enfo",
            "type": "pf",
            "number": em_idx,
            "expver": "0001",
            "paramId": int(pp_info["pp_code"]),
            "level": pp_info["pp_level"],
            "levtype": pp_info["pp_level_type"],
            "dataDate": grib_base_date,
            "dataTime": grib_base_time,
            "stepRange": str(grib_step),
        }
        sa_template = sa_sample[0]

        for sa_code in range(1, cfg.num_sub_areas + 1):
            sa_str = str(sa_code).zfill(cfg.num_digits_sub_area)

            # Extract the sub-area slice from the global CDF
            grid_sa_start = n_grid_sa * (sa_code - 1)
            grid_sa_end = grid_sa_start + n_grid_sa

            sa_values = [
                cdf[grid_sa_start:grid_sa_end] for cdf in cdf_list
            ]

            sa_metadata_list = [
                sa_template.metadata().override(**sa_metadata_overrides)
            ] * len(sa_values)
            sa_fl = ekd.FieldList.from_array(sa_values, sa_metadata_list)

            sa_path = (
                paths.wdir_ens_pt_rain / date_time_str / sf_str / sa_str
                / f"EnsPtRain_{sa_str}_{em_str}.grib"
            )
            write_grib(sa_fl, sa_path)

    logger.info("  Post-processing complete for %s step %s", date_time_str, sf_str)


# =============================================================================
# Section 8: Percentile Computation & Merging (replaces PtRainPerc_MergeSA.mv)
# =============================================================================


def compute_percentiles_and_merge(
    cfg: EcPointConfig,
    paths: EcPointPaths,
    base_date: datetime.date,
    base_time: int,
    step_start: int,
) -> None:
    """Compute percentiles from CDFs per sub-area, then merge to global field.

    Replaces PtRainPerc_MergeSA.mv.

    NOTE: The original code had a bug on line 33 where the ensemble member
    loop was hardcoded to 'for CodeEM = EnsMemS to 2'. This implementation
    correctly uses cfg.ensemble_member_end.
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    step_f = step_start + cfg.accumulation_hours
    sf_str = str(step_f).zfill(cfg.num_digits_step)

    logger.info("  Percentiles computation & merging")

    percentiles_per_sa: list[np.ndarray] = []
    # Will be list of arrays, each shape (num_percentiles, n_grid_sa)

    for sa_code in range(1, cfg.num_sub_areas + 1):
        sa_str = str(sa_code).zfill(cfg.num_digits_sub_area)
        logger.info("    Reading SA n.%d", sa_code)

        # Read CDFs for all ensemble members for this sub-area
        all_cdf_values = []
        for em_idx in range(
            cfg.ensemble_member_start, cfg.ensemble_member_end + 1
        ):
            em_str = str(em_idx).zfill(cfg.num_digits_ensemble_member)
            cdf_path = (
                paths.wdir_ens_pt_rain / date_time_str / sf_str / sa_str
                / f"EnsPtRain_{sa_str}_{em_str}.grib"
            )
            cdf_data = read_grib(cdf_path)
            cdf_vals = get_all_values(cdf_data)  # list of arrays
            all_cdf_values.extend(cdf_vals)

        # Stack all CDF values: shape (n_total_cdf_fields, n_grid_sa)
        cdf_matrix = np.stack(all_cdf_values, axis=0)

        # Compute percentiles along the field axis (axis=0)
        # Result shape: (num_percentiles, n_grid_sa)
        perc_values = np.percentile(
            cdf_matrix, cfg.percentiles, axis=0
        )
        percentiles_per_sa.append(perc_values)

    # Merge sub-areas into global field
    logger.info("    Merging %d SAs into global field", cfg.num_sub_areas)
    num_perc = len(cfg.percentiles)

    # percentiles_per_sa[sa] has shape (num_perc, n_grid_sa)
    # We need to concatenate along the grid dimension for each percentile
    global_percentiles = np.concatenate(percentiles_per_sa, axis=1)
    # shape: (num_perc, n_grid_global)

    # Create output GRIB using global sample as template
    gl_sample = read_grib(paths.global_sample_file)
    gl_template = gl_sample[0]

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
        paths.wdir_cdf_gl / date_time_str / sf_str / "PtRainCDF_GL.grib"
    )
    write_grib(result, out_path)
    logger.info("  Percentiles saved to %s", out_path)


# =============================================================================
# Section 9: Output File Management (replaces shell mv/grib_copy in ecPoint.mv)
# =============================================================================


def move_outputs(
    cfg: EcPointConfig,
    paths: EcPointPaths,
    base_date: datetime.date,
    base_time: int,
    step_start: int,
) -> None:
    """Move output files from working directories to the final output database.

    Replaces the shell("mv ...") and shell("grib_copy ...") calls in
    ecPoint.mv.
    """
    bd_str = base_date.strftime("%Y%m%d")
    bt_str = str(base_time).zfill(cfg.num_digits_base_time)
    date_time_str = f"{bd_str}{bt_str}"
    step_f = step_start + cfg.accumulation_hours
    sf_str = str(step_f).zfill(cfg.num_digits_step)
    acc_str = cfg.accumulation_str

    logger.info("  Moving output files to output database")

    # 1. Point Rainfall CDFs (single file move)
    src_cdf = (
        paths.wdir_cdf_gl / date_time_str / sf_str / "PtRainCDF_GL.grib"
    )
    dst_cdf = (
        paths.out_pt_perc / date_time_str / sf_str
        / f"Pt_BC_PERC_{acc_str}_{bd_str}_{bt_str}_{sf_str}.grib"
    )
    dst_cdf.parent.mkdir(parents=True, exist_ok=True)
    if src_cdf.is_file():
        shutil.move(str(src_cdf), str(dst_cdf))

    # 2. Grid-bias corrected rainfall (concatenate all EM files)
    grid_rain_dir = paths.wdir_grid_rain / date_time_str / sf_str
    grid_files = sorted(grid_rain_dir.glob("GridRain_*.grib"))
    if grid_files:
        dst_grid = (
            paths.out_grid_vals / date_time_str / sf_str
            / f"Grid_BC_VALS_{acc_str}_{bd_str}_{bt_str}_{sf_str}.grib"
        )
        concat_grib_files(grid_files, dst_grid)

    # 3. Weather types (concatenate all EM files)
    wt_dir = paths.wdir_wt / date_time_str / sf_str
    wt_files = sorted(wt_dir.glob("WT_*.grib"))
    if wt_files:
        dst_wt = (
            paths.out_wt / date_time_str / sf_str
            / f"WT_{acc_str}_{bd_str}_{bt_str}_{sf_str}.grib"
        )
        concat_grib_files(wt_files, dst_wt)


# =============================================================================
# Section 10: Helpers
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
# Section 11: Main Orchestrator (replaces ecPoint.mv)
# =============================================================================


def run_ecpoint(cfg: EcPointConfig) -> None:
    """Run the full ecPoint post-processing pipeline.

    This is the main entry point that orchestrates the entire workflow,
    replacing ecPoint.mv.
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

                logger.info("Step 3: Percentiles computation & merging")
                compute_percentiles_and_merge(
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
# Section 12: CLI Entry Point
# =============================================================================


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a JSON configuration file.",
)
@click.option("--var", "var_to_postprocess", default="Rainfall")
@click.option("--acc", "accumulation_hours", type=int, default=12)
@click.option("--cal-version", "calibration_version", default="1.0.0")
@click.option("--run-mode", default="Dev")
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
@click.option("--num-sa", "num_sub_areas", type=int, default=10)
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
    num_sub_areas,
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

    # Build config from file or CLI options
    overrides = {}
    if var_to_postprocess != "Rainfall":
        overrides["var_to_postprocess"] = var_to_postprocess
    if accumulation_hours != 12:
        overrides["accumulation_hours"] = accumulation_hours
    if calibration_version != "1.0.0":
        overrides["calibration_version"] = calibration_version
    if run_mode != "Dev":
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
    if num_sub_areas != 10:
        overrides["num_sub_areas"] = num_sub_areas
    if main_dir is not None:
        overrides["main_dir"] = main_dir

    cfg = load_config(config_path, **overrides)
    run_ecpoint(cfg)


if __name__ == "__main__":
    main()
