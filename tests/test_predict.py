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

"""Tests for predictor computation.

Tests the pure-numpy components of predictor calculation without requiring
GRIB files: weighted time averaging, forecast step index computation,
input file path construction, and the core precipitation/wind/radiation formulas.
"""

import numpy as np
import pytest

from ecpoint.ecpoint import (
    _build_input_file_path,
    _check_input_files_exist,
    _compute_steps,
    _weighted_time_average,
)


class TestWeightedTimeAverage:
    """Test the (0.5*f1 + f2 + 0.5*f3) / 2 formula."""

    def test_uniform_values(self):
        f = np.array([10.0, 10.0, 10.0])
        result = _weighted_time_average(f, f, f)
        np.testing.assert_allclose(result, [10.0, 10.0, 10.0])

    def test_known_values(self):
        f1 = np.array([2.0])
        f2 = np.array([4.0])
        f3 = np.array([6.0])
        # (0.5*2 + 4 + 0.5*6) / 2 = (1 + 4 + 3) / 2 = 4.0
        result = _weighted_time_average(f1, f2, f3)
        np.testing.assert_allclose(result, [4.0])

    def test_zero_input(self):
        z = np.zeros(5)
        result = _weighted_time_average(z, z, z)
        np.testing.assert_allclose(result, np.zeros(5))

    def test_large_array(self):
        rng = np.random.default_rng(42)
        f1 = rng.random(10000)
        f2 = rng.random(10000)
        f3 = rng.random(10000)
        result = _weighted_time_average(f1, f2, f3)
        expected = (0.5 * f1 + f2 + 0.5 * f3) / 2.0
        np.testing.assert_allclose(result, expected)


class TestComputeSteps:
    """Test step index computation."""

    def test_basic_12h_accumulation(self):
        steps = _compute_steps(step_start=0, accumulation=12, step_final_max=114)
        assert steps["step_f"] == 12
        assert steps["step1"] == 0
        assert steps["step2"] == 6
        assert steps["step3"] == 12

    def test_solar_radiation_steps_early(self):
        """When step_f <= 24, SR uses 0 and 24."""
        steps = _compute_steps(step_start=0, accumulation=12, step_final_max=114)
        assert steps["step1_sr"] == 0
        assert steps["step2_sr"] == 24

    def test_solar_radiation_steps_late(self):
        """When step_f > 24, SR uses step_f-24 and step_f."""
        steps = _compute_steps(step_start=24, accumulation=12, step_final_max=114)
        assert steps["step_f"] == 36
        assert steps["step1_sr"] == 12
        assert steps["step2_sr"] == 36

    def test_boundary_24(self):
        """Exactly step_f == 24 should use the early case."""
        steps = _compute_steps(step_start=12, accumulation=12, step_final_max=114)
        assert steps["step_f"] == 24
        assert steps["step1_sr"] == 0
        assert steps["step2_sr"] == 24


class TestBuildInputFilePath:
    """Test input file path construction."""

    def test_format(self, tmp_path):
        path = _build_input_file_path(
            tmp_path, "2020021900", step=12, num_digits_step=3,
            param_code=228
        )
        assert path.name == "228_012.grib"
        assert "012" in str(path)
        assert "2020021900" in str(path)


class TestCheckInputFilesExist:
    """Test that missing files produce clear errors."""

    def test_all_exist(self, tmp_path):
        files = {}
        for name in ["tp", "cp", "u700"]:
            p = tmp_path / f"{name}.grib"
            p.write_bytes(b"dummy")
            files[name] = p
        # Should not raise
        _check_input_files_exist(files)

    def test_some_missing(self, tmp_path):
        existing = tmp_path / "tp.grib"
        existing.write_bytes(b"dummy")
        files = {
            "tp": existing,
            "cp": tmp_path / "missing_cp.grib",
            "u700": tmp_path / "missing_u700.grib",
        }
        with pytest.raises(FileNotFoundError) as exc_info:
            _check_input_files_exist(files)
        msg = str(exc_info.value)
        assert "cp" in msg
        assert "u700" in msg
        assert "tp" not in msg  # tp exists, shouldn't be in error


class TestPrecipitationComputation:
    """Test the core predictor formulas as pure numpy operations."""

    def test_total_precipitation(self):
        """TP = (tp_step3 - tp_step1) * 1000 (m -> mm)."""
        tp_1 = np.array([0.001, 0.005, 0.010])
        tp_3 = np.array([0.003, 0.012, 0.025])
        tp_mm = (tp_3 - tp_1) * 1000.0
        np.testing.assert_allclose(tp_mm, [2.0, 7.0, 15.0])

    def test_cpr_normal(self):
        """CPR = cp / tp where tp > 0."""
        tp = np.array([10.0, 5.0, 20.0])
        cp = np.array([3.0, 5.0, 8.0])
        cpr = np.where(tp > 0, cp / tp, 0.0)
        np.testing.assert_allclose(cpr, [0.3, 1.0, 0.4])

    def test_cpr_zero_division(self):
        """CPR should be 0 where tp == 0."""
        tp = np.array([0.0, 5.0, 0.0])
        cp = np.array([0.0, 3.0, 0.0])
        with np.errstate(invalid="ignore", divide="ignore"):
            cpr = np.where(tp > 0, cp / tp, 0.0)
        np.testing.assert_allclose(cpr, [0.0, 0.6, 0.0])

    def test_wind_speed(self):
        """wspd = sqrt(u^2 + v^2)."""
        u = np.array([3.0, 0.0, -4.0])
        v = np.array([4.0, 5.0, 3.0])
        wspd = np.sqrt(u**2 + v**2)
        np.testing.assert_allclose(wspd, [5.0, 5.0, 5.0])

    def test_solar_radiation(self):
        """SR = (sr_step2 - sr_step1) / 86400."""
        sr_1 = np.array([0.0, 86400.0])
        sr_2 = np.array([86400.0, 172800.0])
        sr24h = (sr_2 - sr_1) / 86400.0
        np.testing.assert_allclose(sr24h, [1.0, 1.0])
