"""Tests for percentile computation.

Tests the statistical percentile logic using pure numpy arrays (no GRIB I/O).
Covers uniform distributions, multi-ensemble CDF matrices, single-field edge
cases, and all-zero fields to validate the np.percentile-based approach.
"""

import numpy as np
import pytest


class TestPercentileComputation:
    """Test the core percentile logic using pure numpy (no GRIB I/O needed)."""

    def test_basic_percentiles(self):
        """Percentiles of a uniform distribution should be linear."""
        # 100 values from 1 to 100
        data = np.arange(1, 101, dtype=np.float64).reshape(100, 1)
        percs = np.percentile(data, [25, 50, 75], axis=0)
        np.testing.assert_allclose(percs[:, 0], [25.75, 50.5, 75.25])

    def test_percentiles_across_ensemble(self):
        """Simulate computing percentiles across multiple ensemble CDF fields."""
        rng = np.random.default_rng(42)
        n_grid = 50
        n_ens = 10
        n_fers = 5

        # Simulate n_ens * n_fers CDF fields, each with n_grid points
        cdf_matrix = rng.random((n_ens * n_fers, n_grid))

        perc_values = [25, 50, 75]
        result = np.percentile(cdf_matrix, perc_values, axis=0)

        assert result.shape == (3, n_grid)
        # 25th percentile should be <= 50th <= 75th everywhere
        assert np.all(result[0] <= result[1])
        assert np.all(result[1] <= result[2])

    def test_single_field_percentiles(self):
        """With a single field, all percentiles should equal that field."""
        data = np.array([[1.0, 2.0, 3.0]])  # 1 field, 3 grid points
        result = np.percentile(data, [10, 50, 90], axis=0)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[2], [1.0, 2.0, 3.0])

    def test_all_zero_field(self):
        """Percentiles of all-zero fields should be zero."""
        data = np.zeros((50, 100))
        result = np.percentile(data, list(range(1, 100)), axis=0)
        np.testing.assert_allclose(result, 0.0)
