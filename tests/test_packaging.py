"""Tests for verifying the library can be imported after installation."""

import pytest


class TestPackageImports:
    """Test that all main package components can be imported."""

    def test_import_neuroprice(self):
        """Test that neuroprice can be imported."""
        import neuroprice

        assert hasattr(neuroprice, '__version__')
        assert isinstance(neuroprice.__version__, str)

    def test_import_pricing_engine(self):
        """Test that PricingEngine can be imported."""
        from neuroprice import PricingEngine

        assert PricingEngine is not None
        assert callable(PricingEngine)

    def test_import_load_strategy(self):
        """Test that load_strategy can be imported."""
        from neuroprice import load_strategy

        assert callable(load_strategy)

    def test_import_save_strategy(self):
        """Test that save_strategy can be imported."""
        from neuroprice import save_strategy

        assert callable(save_strategy)

    def test_import_strategy_config(self):
        """Test that StrategyConfig can be imported."""
        from neuroprice import StrategyConfig

        assert StrategyConfig is not None

    def test_import_all_exceptions(self):
        """Test that all exceptions can be imported."""
        from neuroprice import (
            NeuroPriceError,
            StrategyFileNotFoundError,
            StrategyValidationError,
            StateRollbackError,
            TrainingError,
            PredictionError,
            ConfigurationError,
        )

        # Verify inheritance
        assert issubclass(StrategyFileNotFoundError, NeuroPriceError)
        assert issubclass(StrategyValidationError, NeuroPriceError)
        assert issubclass(StateRollbackError, NeuroPriceError)
        assert issubclass(TrainingError, NeuroPriceError)
        assert issubclass(PredictionError, NeuroPriceError)
        assert issubclass(ConfigurationError, NeuroPriceError)

    def test_import_model_types(self):
        """Test that model configuration types can be imported."""
        from neuroprice import ModelType, RLAlgorithm

        assert ModelType.RL.value == "rl"
        assert ModelType.CAUSAL.value == "causal"
        assert ModelType.HYBRID.value == "hybrid"

        assert RLAlgorithm.PPO.value == "PPO"
        assert RLAlgorithm.A2C.value == "A2C"
        assert RLAlgorithm.DQN.value == "DQN"

    def test_import_config_models(self):
        """Test that configuration models can be imported."""
        from neuroprice import (
            PricingConfig,
            DemandConfig,
            InventoryConfig,
            RLConfig,
            CausalConfig,
            FeaturesConfig,
            OutputConfig,
        )

        assert PricingConfig is not None
        assert DemandConfig is not None
        assert InventoryConfig is not None
        assert RLConfig is not None
        assert CausalConfig is not None
        assert FeaturesConfig is not None
        assert OutputConfig is not None

    def test_import_state_manager(self):
        """Test that StateManager can be imported."""
        from neuroprice import StateManager

        assert StateManager is not None
        assert callable(StateManager)

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import neuroprice

        version = neuroprice.__version__
        parts = version.split('.')

        assert len(parts) == 3, f"Version should have 3 parts, got: {version}"
        assert all(
            p.isdigit() for p in parts
        ), f"Version parts should be numeric: {version}"


class TestSubmoduleImports:
    """Test that submodules can be imported."""

    def test_import_rl_module(self):
        """Test that RL module can be imported."""
        from neuroprice.rl import DynamicPricingEnv, RLAgent

        assert DynamicPricingEnv is not None
        assert RLAgent is not None

    def test_import_causal_module(self):
        """Test that causal module can be imported."""
        from neuroprice.causal import CausalEstimator

        assert CausalEstimator is not None

    def test_import_io_module(self):
        """Test that IO module can be imported."""
        from neuroprice.io import load_strategy, save_strategy, validate_strategy_dict

        assert callable(load_strategy)
        assert callable(save_strategy)
        assert callable(validate_strategy_dict)

    def test_import_state_module(self):
        """Test that state module can be imported."""
        from neuroprice.state import StateManager

        manager = StateManager()
        assert len(manager) == 0


class TestModuleAttributes:
    """Test module-level attributes."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import neuroprice

        expected_exports = [
            "PricingEngine",
            "load_strategy",
            "StrategyConfig",
            "NeuroPriceError",
            "StrategyFileNotFoundError",
        ]

        for export in expected_exports:
            assert export in neuroprice.__all__, f"{export} not in __all__"

    def test_author_attribute(self):
        """Test that __author__ is defined."""
        import neuroprice

        assert hasattr(neuroprice, '__author__')
        assert isinstance(neuroprice.__author__, str)
