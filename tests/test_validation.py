"""Tests for environment validation."""

import pytest

from ecpoint.ecpoint import (
    EcPointConfig,
    EcPointPaths,
    build_paths,
    validate_environment,
)


class TestValidateEnvironment:
    """Test filesystem validation checks."""

    def test_missing_main_dir(self, tmp_path):
        cfg = EcPointConfig(main_dir=tmp_path / "nonexistent")
        paths = build_paths(cfg)
        with pytest.raises(FileNotFoundError, match="main_dir"):
            validate_environment(cfg, paths)

    def test_missing_breakpoints_file(self, small_config, small_paths):
        # main_dir exists (it's tmp_path), but calibration files don't
        with pytest.raises(FileNotFoundError, match="Breakpoints"):
            validate_environment(small_config, small_paths)

    def test_multiple_errors_reported(self, small_config, small_paths):
        """All missing files should be listed in a single error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_environment(small_config, small_paths)
        msg = str(exc_info.value)
        assert "Breakpoints" in msg
        assert "FERs" in msg
        assert "Global sample" in msg
        assert "Sub-area sample" in msg

    def test_passes_when_all_files_exist(self, small_config, small_paths):
        """Validation should pass when all required files are present."""
        # Create all required files
        for f in [
            small_paths.breakpoints_file,
            small_paths.fers_file,
            small_paths.global_sample_file,
            small_paths.sub_area_sample_file,
        ]:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("dummy")

        # Should not raise
        validate_environment(small_config, small_paths)
