"""Tests for GRIB utility functions and helper routines.

Tests GRIB read error handling (missing files) and the pure-Python helper
functions for date/time/step range generation that drive the main loop.
"""

from pathlib import Path

import numpy as np
import pytest

from ecpoint.ecpoint import read_grib


class TestReadGrib:
    """Test GRIB reading helper."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_grib(tmp_path / "nonexistent.grib")

    def test_missing_file_shows_path(self, tmp_path):
        bad_path = tmp_path / "some" / "deep" / "file.grib"
        with pytest.raises(FileNotFoundError) as exc_info:
            read_grib(bad_path)
        assert "file.grib" in str(exc_info.value)


class TestHelperFunctions:
    """Test helper utilities that don't need GRIB files."""

    def test_date_range(self):
        import datetime
        from ecpoint.ecpoint import _date_range

        dates = _date_range(
            datetime.date(2020, 1, 1), datetime.date(2020, 1, 3)
        )
        assert len(dates) == 3
        assert dates[0] == datetime.date(2020, 1, 1)
        assert dates[2] == datetime.date(2020, 1, 3)

    def test_date_range_single(self):
        import datetime
        from ecpoint.ecpoint import _date_range

        dates = _date_range(
            datetime.date(2020, 6, 15), datetime.date(2020, 6, 15)
        )
        assert len(dates) == 1

    def test_time_range(self):
        from ecpoint.ecpoint import _time_range

        times = _time_range(0, 12, 12)
        assert times == [0, 12]

    def test_time_range_single(self):
        from ecpoint.ecpoint import _time_range

        times = _time_range(0, 0, 12)
        assert times == [0]

    def test_step_range(self):
        from ecpoint.ecpoint import _step_range

        steps = _step_range(0, 24, 6)
        assert steps == [0, 6, 12, 18, 24]

    def test_step_range_single(self):
        from ecpoint.ecpoint import _step_range

        steps = _step_range(0, 0, 6)
        assert steps == [0]
