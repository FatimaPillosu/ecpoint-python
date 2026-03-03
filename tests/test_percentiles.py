"""Tests for percentile computation and merging (Section 8 of ecpoint.py)."""

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


class TestSubAreaMerging:
    """Test the sub-area to global field merging logic."""

    def test_basic_merge(self):
        """Merging sub-areas should concatenate along the grid axis."""
        num_sa = 5
        num_perc = 3
        n_grid_sa = 10

        # Create percentile arrays per sub-area
        sa_percentiles = [
            np.full((num_perc, n_grid_sa), fill_value=float(i))
            for i in range(num_sa)
        ]

        # Merge
        global_perc = np.concatenate(sa_percentiles, axis=1)

        assert global_perc.shape == (num_perc, num_sa * n_grid_sa)
        # First SA should have value 0, second SA value 1, etc.
        np.testing.assert_allclose(global_perc[:, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(global_perc[:, 10], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(global_perc[:, 40], [4.0, 4.0, 4.0])

    def test_merge_preserves_percentile_ordering(self):
        """After merging, percentile ordering should be preserved per grid point."""
        rng = np.random.default_rng(42)
        num_sa = 3
        n_grid_sa = 20
        perc_values = [10, 50, 90]

        sa_percentiles = []
        for _ in range(num_sa):
            # Generate random data and compute percentiles
            data = rng.random((100, n_grid_sa))
            perc = np.percentile(data, perc_values, axis=0)
            sa_percentiles.append(perc)

        global_perc = np.concatenate(sa_percentiles, axis=1)

        # p10 <= p50 <= p90 for all grid points
        assert np.all(global_perc[0] <= global_perc[1])
        assert np.all(global_perc[1] <= global_perc[2])

    def test_sub_area_slicing(self):
        """Verify the sub-area extraction formula from postprocess."""
        n_grid_sa = 100
        n_grid_global = 500  # 5 sub-areas
        num_sa = 5

        global_field = np.arange(n_grid_global, dtype=np.float64)

        for sa_code in range(1, num_sa + 1):
            grid_sa_start = n_grid_sa * (sa_code - 1)
            grid_sa_end = grid_sa_start + n_grid_sa
            sa_slice = global_field[grid_sa_start:grid_sa_end]

            assert len(sa_slice) == n_grid_sa
            assert sa_slice[0] == n_grid_sa * (sa_code - 1)
            assert sa_slice[-1] == n_grid_sa * sa_code - 1
