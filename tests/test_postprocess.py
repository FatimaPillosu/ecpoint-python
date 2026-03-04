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

"""Tests for post-processing (weather type classification and FER application).

Tests use the sample_calibration fixture (3 WTs, 2 predictors, 5 FERs) to
verify correct WT assignment including boundary conditions and vectorized
multi-grid-point classification, plus FER-based bias correction arithmetic.
"""

import numpy as np
import pytest

from ecpoint.ecpoint import (
    CalibrationData,
    apply_fers,
    classify_weather_types,
)


class TestClassifyWeatherTypes:
    """Test weather type classification using breakpoint thresholds."""

    def test_single_matching_wt(self, sample_calibration):
        """A grid point that matches WT 101 (pred1 < 0.5, pred2 < 10)."""
        predictors = [
            np.array([0.2]),   # pred1: within [-9999, 0.5)
            np.array([5.0]),   # pred2: within [-9999, 10)
        ]
        codes, indices = classify_weather_types(
            predictors, sample_calibration
        )
        assert codes[0] == 101
        assert indices[0] == 0

    def test_second_wt_match(self, sample_calibration):
        """A grid point matching WT 102 (pred1 in [0.5, 1.0), pred2 < 10)."""
        predictors = [
            np.array([0.7]),  # pred1: in [0.5, 1.0)
            np.array([5.0]),  # pred2: in [-9999, 10)
        ]
        codes, indices = classify_weather_types(
            predictors, sample_calibration
        )
        assert codes[0] == 102
        assert indices[0] == 1

    def test_third_wt_match(self, sample_calibration):
        """A grid point matching WT 103 (pred1 in [0.5, 1.0), pred2 >= 10)."""
        predictors = [
            np.array([0.8]),   # pred1: in [0.5, 1.0)
            np.array([50.0]),  # pred2: in [10, 9999)
        ]
        codes, indices = classify_weather_types(
            predictors, sample_calibration
        )
        assert codes[0] == 103
        assert indices[0] == 2

    def test_no_match(self, sample_calibration):
        """A grid point that matches no WT should get code 0 and index -1."""
        predictors = [
            np.array([2.0]),   # pred1 >= 1.0 — outside all WTs
            np.array([5.0]),
        ]
        codes, indices = classify_weather_types(
            predictors, sample_calibration
        )
        assert codes[0] == 0
        assert indices[0] == -1

    def test_multiple_grid_points(self, sample_calibration):
        """Test vectorized classification across multiple grid points."""
        predictors = [
            np.array([0.2, 0.7, 0.8, 2.0]),  # pred1
            np.array([5.0, 5.0, 50.0, 5.0]),  # pred2
        ]
        codes, indices = classify_weather_types(
            predictors, sample_calibration
        )
        np.testing.assert_array_equal(codes, [101, 102, 103, 0])
        np.testing.assert_array_equal(indices, [0, 1, 2, -1])

    def test_boundary_value_lower(self, sample_calibration):
        """Exact lower boundary should be included (>=)."""
        predictors = [
            np.array([0.5]),  # pred1: exactly at boundary -> WT 102
            np.array([5.0]),
        ]
        codes, _ = classify_weather_types(predictors, sample_calibration)
        assert codes[0] == 102

    def test_boundary_value_upper(self, sample_calibration):
        """Exact upper boundary should be excluded (<)."""
        predictors = [
            np.array([1.0]),  # pred1: exactly at upper boundary -> no WT 102
            np.array([5.0]),
        ]
        codes, indices = classify_weather_types(
            predictors, sample_calibration
        )
        # pred1 = 1.0 is NOT < 1.0, so WT 102 doesn't match
        # and pred1 = 1.0 is NOT < 0.5, so WT 101 doesn't match
        # No WT should match
        assert codes[0] == 0
        assert indices[0] == -1


class TestApplyFERs:
    """Test FER application to create bias-corrected CDFs."""

    def test_basic_application(self, sample_calibration):
        """CDF value = predictand * (FER + 1) for the matching WT."""
        predictand = np.array([10.0])
        wt_indices = np.array([0])  # WT 101

        cdf_list = apply_fers(predictand, wt_indices, sample_calibration)

        assert len(cdf_list) == 5  # 5 FER columns
        # FERs for WT 101: [-0.5, -0.2, 0.0, 0.3, 0.8]
        # CDF = 10 * (FER + 1) = [5.0, 8.0, 10.0, 13.0, 18.0]
        expected = [5.0, 8.0, 10.0, 13.0, 18.0]
        for i, exp in enumerate(expected):
            np.testing.assert_allclose(cdf_list[i], [exp])

    def test_no_wt_match(self, sample_calibration):
        """Grid points with no WT match should use FER=0 (factor=1)."""
        predictand = np.array([10.0])
        wt_indices = np.array([-1])

        cdf_list = apply_fers(predictand, wt_indices, sample_calibration)
        # All CDF values should equal predictand * 1 = 10.0
        for cdf in cdf_list:
            np.testing.assert_allclose(cdf, [10.0])

    def test_zero_predictand(self, sample_calibration):
        """Zero predictand should give zero CDF regardless of FER."""
        predictand = np.array([0.0])
        wt_indices = np.array([1])  # WT 102

        cdf_list = apply_fers(predictand, wt_indices, sample_calibration)
        for cdf in cdf_list:
            np.testing.assert_allclose(cdf, [0.0])

    def test_multiple_grid_points(self, sample_calibration):
        """Test vectorized FER application."""
        predictand = np.array([10.0, 20.0, 0.0])
        wt_indices = np.array([0, 2, -1])

        cdf_list = apply_fers(predictand, wt_indices, sample_calibration)

        # Grid point 0: WT 101, FER[0,0] = -0.5 -> 10*(0.5) = 5.0
        np.testing.assert_allclose(cdf_list[0][0], 5.0)
        # Grid point 1: WT 103, FER[2,0] = -0.1 -> 20*(0.9) = 18.0
        np.testing.assert_allclose(cdf_list[0][1], 18.0)
        # Grid point 2: no WT, predictand=0 -> 0.0
        np.testing.assert_allclose(cdf_list[0][2], 0.0)

    def test_grid_bias_corrected_is_mean_of_cdf(self, sample_calibration):
        """The grid-scale bias correction should be the mean of CDF values."""
        predictand = np.array([10.0])
        wt_indices = np.array([0])  # WT 101

        cdf_list = apply_fers(predictand, wt_indices, sample_calibration)
        cdf_stack = np.stack(cdf_list, axis=1)
        grid_bc = np.mean(cdf_stack, axis=1)

        # Mean of [5.0, 8.0, 10.0, 13.0, 18.0] = 10.8
        np.testing.assert_allclose(grid_bc, [10.8])
