"""Tests for calibration data loading (Section 5 of ecpoint.py)."""

import textwrap

import numpy as np
import pytest

from ecpoint.ecpoint import CalibrationData, EcPointPaths, load_calibration


class TestLoadCalibration:
    """Test loading breakpoints and FERs from CSV files."""

    def test_load_breakpoints(
        self, tmp_path, sample_breakpoints_csv, sample_fers_csv
    ):
        paths = _make_paths(tmp_path, sample_breakpoints_csv, sample_fers_csv)
        cal = load_calibration(paths)

        assert cal.num_weather_types == 3
        assert cal.num_predictors == 2
        np.testing.assert_array_equal(
            cal.weather_type_codes, [101, 102, 103]
        )

    def test_load_breakpoint_thresholds(
        self, tmp_path, sample_breakpoints_csv, sample_fers_csv
    ):
        paths = _make_paths(tmp_path, sample_breakpoints_csv, sample_fers_csv)
        cal = load_calibration(paths)

        # First WT: pred1 in [-9999, 0.5), pred2 in [-9999, 10)
        np.testing.assert_allclose(cal.breakpoints_low[0], [-9999, -9999])
        np.testing.assert_allclose(cal.breakpoints_high[0], [0.5, 10])

        # Third WT: pred1 in [0.5, 1.0), pred2 in [10, 9999)
        np.testing.assert_allclose(cal.breakpoints_low[2], [0.5, 10])
        np.testing.assert_allclose(cal.breakpoints_high[2], [1.0, 9999])

    def test_load_fers(
        self, tmp_path, sample_breakpoints_csv, sample_fers_csv
    ):
        paths = _make_paths(tmp_path, sample_breakpoints_csv, sample_fers_csv)
        cal = load_calibration(paths)

        assert cal.num_fers == 5
        assert cal.fers.shape == (3, 5)
        np.testing.assert_allclose(
            cal.fers[0], [-0.5, -0.2, 0.0, 0.3, 0.8]
        )

    def test_mismatched_rows_raises(self, tmp_path, sample_breakpoints_csv):
        """FERs with different number of WTs than breakpoints should fail."""
        fers_content = textwrap.dedent("""\
            Wtcode,FER1,FER2
            101,-0.5,0.3
            102,-0.3,0.5
        """)
        fers_path = tmp_path / "FERs.txt"
        fers_path.write_text(fers_content)

        paths = _make_paths(tmp_path, sample_breakpoints_csv, fers_path)
        with pytest.raises(ValueError, match="rows"):
            load_calibration(paths)


class TestLoadRealCalibrationFiles:
    """Test loading the actual calibration files from the repository."""

    @pytest.fixture
    def real_bp_path(self):
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / (
            "ecPoint/CompFiles/MapFunc/Dev/"
            "ecPoint_Rainfall/012/Vers1.0.0/BreakPointsWT.txt"
        )
        if not p.exists():
            pytest.skip("Real calibration files not available")
        return p

    @pytest.fixture
    def real_fer_path(self):
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / (
            "ecPoint/CompFiles/MapFunc/Dev/"
            "ecPoint_Rainfall/012/Vers1.0.0/FERs.txt"
        )
        if not p.exists():
            pytest.skip("Real calibration files not available")
        return p

    def test_load_real_breakpoints(
        self, tmp_path, real_bp_path, real_fer_path
    ):
        paths = _make_paths(tmp_path, real_bp_path, real_fer_path)
        cal = load_calibration(paths)

        # The real file has 5 predictors (cpr, tp_acc, wspd700, cape, sr24h)
        assert cal.num_predictors == 5
        # It has many weather types
        assert cal.num_weather_types > 100
        # WT codes should be positive integers
        assert np.all(cal.weather_type_codes > 0)

    def test_load_real_fers(self, tmp_path, real_bp_path, real_fer_path):
        paths = _make_paths(tmp_path, real_bp_path, real_fer_path)
        cal = load_calibration(paths)

        # FERs file has 100 FER columns
        assert cal.num_fers == 100
        assert cal.fers.shape == (cal.num_weather_types, 100)


# --- Helpers ---

def _make_paths(tmp_path, bp_path, fer_path):
    """Create a minimal EcPointPaths-like object with just calibration paths."""
    return _CalibPaths(bp_path, fer_path)


class _CalibPaths:
    """Minimal stand-in for EcPointPaths with just calibration file paths."""

    def __init__(self, bp_path, fer_path):
        self.breakpoints_file = bp_path
        self.fers_file = fer_path
