# (C) Copyright 2020 European Centre for Medium-Range Weather Forecasts.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for point-mode configuration, processing, and CSV output.

Validates point-mode config fields, the is_point_mode property, core
processing with shape (1,) arrays, and CSV output formatting.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from ecpoint.ecpoint import (
    EcPointConfig,
    PointResult,
    _process_single_member,
    classify_weather_types,
    apply_fers,
    write_point_csv,
    build_paths,
)


# =============================================================================
# Config Validation
# =============================================================================


class TestPointModeConfig:
    """Tests for point-mode configuration fields and validation."""

    def test_default_is_grid_mode(self):
        cfg = EcPointConfig()
        assert cfg.is_point_mode is False
        assert cfg.point_lat is None
        assert cfg.point_lon is None

    def test_point_mode_enabled(self):
        cfg = EcPointConfig(point_lat=51.5, point_lon=-0.1)
        assert cfg.is_point_mode is True

    def test_lat_only_raises(self):
        with pytest.raises(ValueError, match="point_lat and point_lon must both"):
            EcPointConfig(point_lat=51.5)

    def test_lon_only_raises(self):
        with pytest.raises(ValueError, match="point_lat and point_lon must both"):
            EcPointConfig(point_lon=-0.1)

    def test_lat_out_of_range(self):
        with pytest.raises(ValueError):
            EcPointConfig(point_lat=91.0, point_lon=0.0)

    def test_lat_negative_boundary(self):
        cfg = EcPointConfig(point_lat=-90.0, point_lon=0.0)
        assert cfg.point_lat == -90.0

    def test_lon_range_positive(self):
        cfg = EcPointConfig(point_lat=0.0, point_lon=359.0)
        assert cfg.point_lon == 359.0

    def test_lon_range_negative(self):
        cfg = EcPointConfig(point_lat=0.0, point_lon=-180.0)
        assert cfg.point_lon == -180.0

    def test_data_source_default(self):
        cfg = EcPointConfig()
        assert cfg.data_source == "local"

    def test_data_source_polytope_requires_point_mode(self):
        with pytest.raises(ValueError, match="data_source='polytope' requires point mode"):
            EcPointConfig(data_source="polytope")

    def test_data_source_polytope_with_point_mode(self):
        cfg = EcPointConfig(
            point_lat=51.5, point_lon=-0.1, data_source="polytope"
        )
        assert cfg.data_source == "polytope"
        assert cfg.is_point_mode is True


# =============================================================================
# Core Processing with Shape (1,) Arrays
# =============================================================================


class TestProcessSingleMemberPoint:
    """Tests that the core processing function works with single-point arrays."""

    def test_single_point_classification(self, sample_calibration):
        """classify_weather_types works with (1,) arrays."""
        # Values matching WT 101: pred1 in [-9999, 0.5), pred2 in [-9999, 10)
        predictors = [np.array([0.3]), np.array([5.0])]
        wt_codes, wt_indices = classify_weather_types(
            predictors, sample_calibration
        )
        assert wt_codes.shape == (1,)
        assert wt_indices.shape == (1,)
        assert wt_codes[0] == 101  # WT code for first weather type

    def test_single_point_apply_fers(self, sample_calibration):
        """apply_fers works with (1,) arrays."""
        predictand = np.array([5.0])
        wt_indices = np.array([0])  # first WT
        cdf_list = apply_fers(predictand, wt_indices, sample_calibration)
        assert len(cdf_list) == sample_calibration.num_fers
        for arr in cdf_list:
            assert arr.shape == (1,)

    def test_process_single_member_point(self, sample_calibration):
        """_process_single_member works end-to-end with (1,) arrays."""
        predictand = np.array([5.0])
        # Values matching WT 101: pred1 in [-9999, 0.5), pred2 in [-9999, 10)
        predictors = [np.array([0.3]), np.array([5.0])]

        grid_bc, wt_codes, cdf_list = _process_single_member(
            predictand, predictors, sample_calibration,
            min_predictand_value=0.04,
        )

        assert grid_bc.shape == (1,)
        assert wt_codes.shape == (1,)
        assert len(cdf_list) == sample_calibration.num_fers
        assert grid_bc[0] > 0  # bias-corrected value should be positive

    def test_process_single_member_below_threshold(self, sample_calibration):
        """Sub-threshold predictand gets sentinel WT code."""
        predictand = np.array([0.01])  # below 0.04 threshold
        predictors = [np.array([0.3]), np.array([5.0])]

        grid_bc, wt_codes, cdf_list = _process_single_member(
            predictand, predictors, sample_calibration,
            min_predictand_value=0.04,
        )

        # Sentinel code: all 9s, length = num_predictors
        expected_sentinel = int("9" * sample_calibration.num_predictors)
        assert wt_codes[0] == expected_sentinel


# =============================================================================
# Percentile Computation with Single Point
# =============================================================================


class TestPercentilesPoint:
    """Tests that percentile computation works with single-point CDF arrays."""

    def test_percentiles_single_point(self):
        """np.percentile works correctly on (n, 1) shaped CDF matrix."""
        # Simulate 5 CDF values at a single point
        cdf_matrix = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        percentiles = [25, 50, 75]
        result = np.percentile(cdf_matrix, percentiles, axis=0)
        assert result.shape == (3, 1)
        assert result[1, 0] == 3.0  # median


# =============================================================================
# CSV Output
# =============================================================================


class TestWritePointCSV:
    """Tests for CSV output formatting."""

    def test_csv_output(self, tmp_path):
        cfg = EcPointConfig(
            point_lat=51.5,
            point_lon=-0.1,
            percentiles=[10, 50, 90],
            main_dir=tmp_path,
        )
        paths = build_paths(cfg)

        results = [
            PointResult(
                date=datetime.date(2020, 2, 19),
                time=0,
                step_start=0,
                step_end=12,
                lat=51.5,
                lon=-0.1,
                wt_code=11111,
                grid_bc=2.45,
                percentile_values=[0.8, 2.4, 5.1],
            ),
            PointResult(
                date=datetime.date(2020, 2, 19),
                time=0,
                step_start=6,
                step_end=18,
                lat=51.5,
                lon=-0.1,
                wt_code=22222,
                grid_bc=1.23,
                percentile_values=[0.3, 1.2, 3.0],
            ),
        ]

        out_path = write_point_csv(cfg, paths, results)
        assert out_path.is_file()

        df = pd.read_csv(out_path)
        assert len(df) == 2
        assert list(df.columns) == [
            "date", "time", "step_start", "step_end", "lat", "lon",
            "wt_code", "grid_bc", "p10", "p50", "p90",
        ]
        assert df.iloc[0]["wt_code"] == 11111
        assert df.iloc[1]["grid_bc"] == pytest.approx(1.23, abs=0.001)

    def test_csv_filename_includes_coordinates(self, tmp_path):
        cfg = EcPointConfig(
            point_lat=51.5,
            point_lon=-0.1,
            percentiles=[50],
            main_dir=tmp_path,
        )
        paths = build_paths(cfg)
        results = [
            PointResult(
                date=datetime.date(2020, 2, 19),
                time=0,
                step_start=0,
                step_end=12,
                lat=51.5,
                lon=-0.1,
                wt_code=0,
                grid_bc=0.0,
                percentile_values=[0.0],
            ),
        ]

        out_path = write_point_csv(cfg, paths, results)
        assert "51.5" in out_path.name
        assert "-0.1" in out_path.name


# =============================================================================
# Polytope Import Guard
# =============================================================================


class TestPolytopeImportGuard:
    """Tests that polytope import failure gives a clear error."""

    def test_require_polytope_import_error(self, monkeypatch):
        """_require_polytope raises ImportError when polytope is not installed."""
        import ecpoint.ecpoint as mod

        monkeypatch.setattr(mod, "HAS_POLYTOPE", False)
        with pytest.raises(ImportError, match="polytope-python is required"):
            mod._require_polytope()

    def test_require_polytope_succeeds(self, monkeypatch):
        """_require_polytope passes when polytope is available."""
        import ecpoint.ecpoint as mod

        monkeypatch.setattr(mod, "HAS_POLYTOPE", True)
        mod._require_polytope()  # should not raise
