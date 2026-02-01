"""Tests for strategy file parsing and validation."""

from pathlib import Path

import pytest

from neuroprice import StrategyConfig, load_strategy
from neuroprice.exceptions import StrategyFileNotFoundError, StrategyValidationError
from neuroprice.io import save_strategy, validate_strategy_dict


class TestLoadStrategy:
    """Tests for load_strategy function."""

    def test_load_valid_strategy(self, sample_strategy_file: Path):
        """Test loading a valid strategy file."""
        config = load_strategy(sample_strategy_file)

        assert isinstance(config, StrategyConfig)
        assert config.name == "test_strategy"
        assert config.version == "1.0"

    def test_load_strategy_all_fields(self, sample_strategy_file: Path):
        """Test that all fields are correctly parsed."""
        config = load_strategy(sample_strategy_file)

        # Check pricing config
        assert config.pricing.min_price == 5.0
        assert config.pricing.max_price == 100.0
        assert config.pricing.price_step == 0.50
        assert config.pricing.currency == "USD"

        # Check demand config
        assert config.demand.elasticity == -1.5
        assert config.demand.base_demand == 1000

        # Check inventory config
        assert config.inventory.initial_stock == 150
        assert config.inventory.salvage_value == 2.0

        # Check model config
        assert config.model.type.value == "rl"
        assert config.model.algorithm.value == "PPO"

    def test_load_strategy_file_not_found(self):
        """Test that missing file raises StrategyFileNotFoundError."""
        with pytest.raises(StrategyFileNotFoundError) as exc_info:
            load_strategy("/nonexistent/path/strategy.yaml")

        error = exc_info.value
        assert "not found" in error.message.lower()
        assert "template" in error.message.lower()  # Helpful guidance

    def test_load_strategy_file_not_found_message_quality(self):
        """Test that error message provides helpful guidance."""
        with pytest.raises(StrategyFileNotFoundError) as exc_info:
            load_strategy("missing.yaml")

        message = str(exc_info.value)
        # Should mention how to create a strategy file
        assert "create" in message.lower() or "template" in message.lower()

    def test_load_strategy_invalid_yaml(self, invalid_yaml_file: Path):
        """Test that invalid YAML raises StrategyValidationError."""
        with pytest.raises(StrategyValidationError) as exc_info:
            load_strategy(invalid_yaml_file)

        assert "yaml" in str(exc_info.value).lower()

    def test_load_strategy_missing_required_fields(self, incomplete_strategy_file: Path):
        """Test that missing required fields raise StrategyValidationError."""
        with pytest.raises(StrategyValidationError):
            load_strategy(incomplete_strategy_file)

    def test_load_strategy_empty_file(self, empty_strategy_file: Path):
        """Test that empty file raises StrategyValidationError."""
        with pytest.raises(StrategyValidationError) as exc_info:
            load_strategy(empty_strategy_file)

        assert "empty" in str(exc_info.value).lower()

    def test_load_strategy_path_types(self, sample_strategy_file: Path):
        """Test that both str and Path work."""
        # Path object
        config1 = load_strategy(sample_strategy_file)
        # String path
        config2 = load_strategy(str(sample_strategy_file))

        assert config1.name == config2.name


class TestSaveStrategy:
    """Tests for save_strategy function."""

    def test_save_and_reload_strategy(self, sample_config, tmp_path: Path):
        """Test that saved strategy can be reloaded."""
        file_path = tmp_path / "saved_strategy.yaml"

        save_strategy(sample_config, file_path)
        reloaded = load_strategy(file_path)

        assert reloaded.name == sample_config.name
        assert reloaded.pricing.min_price == sample_config.pricing.min_price
        assert reloaded.pricing.max_price == sample_config.pricing.max_price

    def test_save_creates_file(self, sample_config, tmp_path: Path):
        """Test that save creates the file."""
        file_path = tmp_path / "new_strategy.yaml"

        assert not file_path.exists()
        save_strategy(sample_config, file_path)
        assert file_path.exists()


class TestValidateStrategyDict:
    """Tests for validate_strategy_dict function."""

    def test_validate_valid_dict(self, sample_strategy_dict):
        """Test validation of valid dictionary."""
        config = validate_strategy_dict(sample_strategy_dict)

        assert isinstance(config, StrategyConfig)
        assert config.name == "test_strategy"

    def test_validate_invalid_dict(self):
        """Test that invalid dict raises StrategyValidationError."""
        with pytest.raises(StrategyValidationError):
            validate_strategy_dict({"name": "incomplete"})


class TestStrategyConfigValidation:
    """Tests for StrategyConfig validation."""

    def test_price_range_validation(self):
        """Test that max_price > min_price is enforced."""
        from neuroprice.models import PricingConfig

        with pytest.raises(ValueError) as exc_info:
            PricingConfig(min_price=100, max_price=50)

        assert "max_price" in str(exc_info.value).lower()

    def test_elasticity_must_be_negative(self, sample_strategy_dict):
        """Test that positive elasticity raises error."""
        sample_strategy_dict["demand"]["elasticity"] = 1.0

        with pytest.raises(Exception):
            StrategyConfig(**sample_strategy_dict)

    def test_required_features_not_empty(self, sample_strategy_dict):
        """Test that required features list cannot be empty."""
        sample_strategy_dict["features"]["required"] = []

        with pytest.raises(Exception):
            StrategyConfig(**sample_strategy_dict)

    def test_unknown_fields_rejected(self, sample_strategy_dict):
        """Test that unknown fields are rejected."""
        sample_strategy_dict["unknown_field"] = "value"

        with pytest.raises(Exception):
            StrategyConfig(**sample_strategy_dict)

    def test_model_type_enum(self, sample_strategy_dict):
        """Test that model type is validated."""
        sample_strategy_dict["model"]["type"] = "invalid_type"

        with pytest.raises(Exception):
            StrategyConfig(**sample_strategy_dict)

    def test_algorithm_enum(self, sample_strategy_dict):
        """Test that algorithm is validated."""
        sample_strategy_dict["model"]["algorithm"] = "INVALID"

        with pytest.raises(Exception):
            StrategyConfig(**sample_strategy_dict)


class TestConfigDefaults:
    """Tests for configuration default values."""

    def test_output_defaults(self, sample_strategy_dict):
        """Test that output config has sensible defaults."""
        del sample_strategy_dict["output"]
        config = StrategyConfig(**sample_strategy_dict)

        assert config.output.price_column == "recommended_price"
        assert config.output.confidence_column == "confidence_score"

    def test_rl_config_defaults(self, sample_strategy_dict):
        """Test that RL config has sensible defaults."""
        del sample_strategy_dict["rl_config"]
        config = StrategyConfig(**sample_strategy_dict)

        assert config.rl_config.learning_rate == 0.0003
        assert config.rl_config.gamma == 0.99

    def test_seasonality_defaults(self, sample_strategy_dict):
        """Test that seasonality config has sensible defaults."""
        del sample_strategy_dict["demand"]["seasonality"]
        config = StrategyConfig(**sample_strategy_dict)

        assert config.demand.seasonality.enabled is False
        assert config.demand.seasonality.period == 7
