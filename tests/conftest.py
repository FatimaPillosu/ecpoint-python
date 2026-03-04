"""Shared fixtures for ecPoint tests.

Provides reusable test fixtures including:
  - default_config / small_config: EcPointConfig instances for unit tests
  - small_paths: derived EcPointPaths for filesystem-related tests
  - sample_breakpoints_csv / sample_fers_csv: minimal calibration CSV files
  - sample_calibration: pre-built CalibrationData with 3 weather types
"""

import datetime
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ecpoint.ecpoint import (
    CalibrationData,
    EcPointConfig,
    EcPointPaths,
    build_paths,
)


@pytest.fixture
def default_config():
    """An EcPointConfig with default values."""
    return EcPointConfig()


@pytest.fixture
def small_config(tmp_path):
    """A minimal config with small ranges for fast tests."""
    return EcPointConfig(
        base_date_start=datetime.date(2020, 1, 1),
        base_date_end=datetime.date(2020, 1, 1),
        base_time_start=0,
        base_time_end=0,
        base_time_disc=12,
        step_start=0,
        step_final=12,
        step_disc=12,
        ensemble_member_start=0,
        ensemble_member_end=2,
        percentiles=list(range(1, 100)),
        main_dir=tmp_path,
    )


@pytest.fixture
def small_paths(small_config):
    """Paths derived from the small config."""
    return build_paths(small_config)


@pytest.fixture
def sample_breakpoints_csv(tmp_path):
    """Create a small BreakPointsWT.txt with 3 weather types and 2 predictors."""
    content = textwrap.dedent("""\
        WTcode,pred1_thrL,pred1_thrH,pred2_thrL,pred2_thrH
        101,-9999,0.5,-9999,10
        102,0.5,1.0,-9999,10
        103,0.5,1.0,10,9999
    """)
    p = tmp_path / "BreakPointsWT.txt"
    p.write_text(content)
    return p


@pytest.fixture
def sample_fers_csv(tmp_path):
    """Create a small FERs.txt with 3 weather types and 5 FER columns."""
    content = textwrap.dedent("""\
        Wtcode,FER1,FER2,FER3,FER4,FER5
        101,-0.5,-0.2,0.0,0.3,0.8
        102,-0.3,0.0,0.2,0.5,1.0
        103,-0.1,0.1,0.4,0.7,1.5
    """)
    p = tmp_path / "FERs.txt"
    p.write_text(content)
    return p


@pytest.fixture
def sample_calibration():
    """A small CalibrationData with 3 weather types and 2 predictors.

    WT 101: pred1 in [-9999, 0.5),  pred2 in [-9999, 10)  — "light/stratiform"
    WT 102: pred1 in [0.5, 1.0),    pred2 in [-9999, 10)  — "moderate"
    WT 103: pred1 in [0.5, 1.0),    pred2 in [10, 9999)   — "convective"

    FERs have 5 columns representing a small empirical error distribution.
    """
    return CalibrationData(
        weather_type_codes=np.array([101, 102, 103]),
        breakpoints_low=np.array([
            [-9999, -9999],
            [0.5, -9999],
            [0.5, 10],
        ], dtype=np.float64),
        breakpoints_high=np.array([
            [0.5, 10],
            [1.0, 10],
            [1.0, 9999],
        ], dtype=np.float64),
        fers=np.array([
            [-0.5, -0.2, 0.0, 0.3, 0.8],
            [-0.3, 0.0, 0.2, 0.5, 1.0],
            [-0.1, 0.1, 0.4, 0.7, 1.5],
        ], dtype=np.float64),
        num_predictors=2,
        num_weather_types=3,
        num_fers=5,
    )
