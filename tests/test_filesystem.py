"""Tests for filesystem creation."""

import datetime

from ecpoint.ecpoint import (
    EcPointConfig,
    build_paths,
    create_filesystem,
)


class TestBuildPaths:
    """Test path construction from config."""

    def test_work_dir(self, small_config, small_paths):
        assert "work" in str(small_paths.work_dir)
        assert str(small_config.main_dir) in str(small_paths.work_dir)

    def test_out_dir(self, small_paths):
        assert "forecasts" in str(small_paths.out_dir)

    def test_calibration_paths(self, small_paths):
        assert small_paths.breakpoints_file.name == "breakpoints_wt.txt"
        assert small_paths.fers_file.name == "fers.txt"

    def test_sample_paths(self, small_paths):
        assert small_paths.global_sample_file.name == "global.grib"

    def test_var_acc_ver_in_path(self, small_paths):
        path_str = str(small_paths.temp_dir)
        assert "ecpoint_rainfall" in path_str
        assert "012" in path_str
        assert "v1.0.0" in path_str


class TestCreateFilesystem:
    """Test directory tree creation."""

    def test_directories_created(self, small_config, small_paths):
        create_filesystem(small_config, small_paths)

        # Check a few representative directories exist
        bd_str = small_config.base_date_start.strftime("%Y%m%d")
        bt_str = str(small_config.base_time_start).zfill(
            small_config.num_digits_base_time
        )
        step_f = small_config.step_start + small_config.accumulation_hours
        sf_str = str(step_f).zfill(small_config.num_digits_step)
        dt_step = f"{bd_str}{bt_str}/{sf_str}"

        assert (small_paths.wdir_grid_rain / dt_step).is_dir()
        assert (small_paths.wdir_wt / dt_step).is_dir()
        assert (small_paths.out_pt_perc / dt_step).is_dir()

    def test_ensemble_member_directories(self, small_config, small_paths):
        create_filesystem(small_config, small_paths)

        bd_str = small_config.base_date_start.strftime("%Y%m%d")
        bt_str = str(small_config.base_time_start).zfill(
            small_config.num_digits_base_time
        )
        step_f = small_config.step_start + small_config.accumulation_hours
        sf_str = str(step_f).zfill(small_config.num_digits_step)
        dt_step = f"{bd_str}{bt_str}/{sf_str}"

        for em in range(
            small_config.ensemble_member_start,
            small_config.ensemble_member_end + 1,
        ):
            em_str = str(em).zfill(small_config.num_digits_ensemble_member)
            assert (small_paths.wdir_predict / dt_step / em_str).is_dir()

    def test_idempotent(self, small_config, small_paths):
        """Running create_filesystem twice should not fail."""
        create_filesystem(small_config, small_paths)
        create_filesystem(small_config, small_paths)
