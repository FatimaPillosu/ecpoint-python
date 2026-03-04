"""Tests for EcPointConfig.

Validates default values, derived properties (computed from config fields),
cross-field validation (e.g., date ranges, valid accumulations), and
config loading from JSON files with optional CLI overrides.
"""

import datetime

import pytest
from pydantic import ValidationError

from ecpoint.ecpoint import EcPointConfig, load_config


class TestConfigDefaults:
    """Test that the default configuration is valid and consistent."""

    def test_default_config_is_valid(self):
        cfg = EcPointConfig()
        assert cfg.var_to_postprocess == "rainfall"
        assert cfg.accumulation_hours == 12
        assert cfg.calibration_version == "1.0.0"

    def test_default_percentiles(self):
        cfg = EcPointConfig()
        assert cfg.percentiles == list(range(1, 100))
        assert len(cfg.percentiles) == 99

    def test_default_float_precision(self):
        cfg = EcPointConfig()
        assert cfg.float_precision == "float32"


class TestDerivedProperties:
    """Test computed properties on the config."""

    def test_num_ensemble_members(self):
        cfg = EcPointConfig(ensemble_member_start=0, ensemble_member_end=50)
        assert cfg.num_ensemble_members == 51

    def test_num_ensemble_members_single(self):
        cfg = EcPointConfig(ensemble_member_start=5, ensemble_member_end=5)
        assert cfg.num_ensemble_members == 1

    def test_accumulation_str(self):
        cfg = EcPointConfig(accumulation_hours=12, num_digits_acc=3)
        assert cfg.accumulation_str == "012"

    def test_numpy_dtype(self):
        import numpy as np
        cfg = EcPointConfig(float_precision="float64")
        assert cfg.numpy_dtype == np.dtype("float64")

    def test_variable_info(self):
        cfg = EcPointConfig()
        info = cfg.variable_info
        assert info["predictand_code"] == 228.128
        assert info["min_predictand_value"] == 0.04

    def test_min_predictand_value(self):
        cfg = EcPointConfig()
        assert cfg.min_predictand_value == 0.04


class TestConfigValidation:
    """Test that invalid configurations are rejected."""

    def test_date_start_after_end(self):
        with pytest.raises(ValidationError, match="base_date_start"):
            EcPointConfig(
                base_date_start=datetime.date(2020, 3, 1),
                base_date_end=datetime.date(2020, 1, 1),
            )

    def test_time_start_after_end(self):
        with pytest.raises(ValidationError, match="base_time_start"):
            EcPointConfig(base_time_start=12, base_time_end=0)

    def test_time_out_of_range(self):
        with pytest.raises(ValidationError):
            EcPointConfig(base_time_start=25)

    def test_ensemble_start_after_end(self):
        with pytest.raises(ValidationError, match="ensemble_member_start"):
            EcPointConfig(
                ensemble_member_start=10, ensemble_member_end=5
            )

    def test_step_start_after_final(self):
        with pytest.raises(ValidationError, match="step_start"):
            EcPointConfig(step_start=120, step_final=12)

    def test_percentile_out_of_range_low(self):
        with pytest.raises(ValidationError, match="1-99"):
            EcPointConfig(percentiles=[0, 50, 99])

    def test_percentile_out_of_range_high(self):
        with pytest.raises(ValidationError, match="1-99"):
            EcPointConfig(percentiles=[1, 50, 100])

    def test_percentile_duplicates(self):
        with pytest.raises(ValidationError, match="duplicates"):
            EcPointConfig(percentiles=[1, 50, 50, 99])

    def test_percentile_empty(self):
        with pytest.raises(ValidationError, match="empty"):
            EcPointConfig(percentiles=[])

    def test_percentiles_sorted(self):
        cfg = EcPointConfig(percentiles=[99, 50, 1])
        assert cfg.percentiles == [1, 50, 99]

    def test_invalid_accumulation_for_variable(self):
        with pytest.raises(ValidationError, match="accumulation_hours"):
            EcPointConfig(
                var_to_postprocess="rainfall", accumulation_hours=6
            )

    def test_negative_step_start(self):
        with pytest.raises(ValidationError):
            EcPointConfig(step_start=-1)

    def test_negative_ensemble_member(self):
        with pytest.raises(ValidationError):
            EcPointConfig(ensemble_member_start=-1)


class TestLoadConfig:
    """Test loading config from file."""

    def test_load_from_json(self, tmp_path):
        import json

        config_data = {
            "accumulation_hours": 12,
            "ensemble_member_start": 0,
            "ensemble_member_end": 10,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        cfg = load_config(config_file)
        assert cfg.ensemble_member_end == 10

    def test_load_with_overrides(self, tmp_path):
        import json

        config_data = {"ensemble_member_end": 10}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        cfg = load_config(config_file, ensemble_member_end=20)
        assert cfg.ensemble_member_end == 20

    def test_load_no_file(self):
        cfg = load_config(None, ensemble_member_end=20)
        assert cfg.ensemble_member_end == 20
