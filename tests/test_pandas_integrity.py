"""Tests for DataFrame integrity - ensuring input columns are preserved."""

from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from neuroprice import PricingEngine


class MockRLAgent:
    """Mock RL agent for testing without full training."""

    def __init__(self, config):
        self.config = config

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mock predictions."""
        n = len(df)
        prices = np.random.uniform(
            self.config.pricing.min_price,
            self.config.pricing.max_price,
            n,
        )
        confidences = np.random.uniform(0.5, 1.0, n)
        return prices, confidences


class TestDataFrameIntegrity:
    """Tests ensuring DataFrame integrity is preserved."""

    @pytest.fixture
    def mock_trained_engine(self, sample_config):
        """Create a mock trained engine for testing."""
        engine = PricingEngine(sample_config)
        engine._is_trained = True
        engine._rl_agent = MockRLAgent(sample_config)
        return engine

    def test_predict_preserves_original_columns(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that predict() does not modify original DataFrame columns."""
        original_columns = list(sample_dataframe.columns)

        result = mock_trained_engine.predict(sample_dataframe)

        # All original columns should exist in result
        for col in original_columns:
            assert col in result.columns, f"Column {col} missing from result"

    def test_predict_preserves_column_values(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that original column values are unchanged."""
        result = mock_trained_engine.predict(sample_dataframe)

        for col in sample_dataframe.columns:
            pd.testing.assert_series_equal(
                result[col],
                sample_dataframe[col],
                check_names=False,
                obj=f"Column {col}",
            )

    def test_predict_only_appends_new_columns(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that predict() only adds configured output columns."""
        original_columns = set(sample_dataframe.columns)
        config = mock_trained_engine.config

        result = mock_trained_engine.predict(sample_dataframe)

        new_columns = set(result.columns) - original_columns
        expected_new = {
            config.output.price_column,
            config.output.confidence_column,
        }

        assert new_columns == expected_new, (
            f"Expected new columns {expected_new}, got {new_columns}"
        )

    def test_predict_does_not_modify_input(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that predict() does not modify the input DataFrame."""
        original_df = sample_dataframe.copy()

        _ = mock_trained_engine.predict(sample_dataframe)

        pd.testing.assert_frame_equal(
            sample_dataframe,
            original_df,
            obj="Input DataFrame should be unchanged",
        )

    def test_output_columns_have_correct_types(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that output columns have numeric types."""
        config = mock_trained_engine.config
        result = mock_trained_engine.predict(sample_dataframe)

        price_col = config.output.price_column
        conf_col = config.output.confidence_column

        assert np.issubdtype(result[price_col].dtype, np.floating), (
            f"Price column should be float, got {result[price_col].dtype}"
        )
        assert np.issubdtype(result[conf_col].dtype, np.floating), (
            f"Confidence column should be float, got {result[conf_col].dtype}"
        )

    def test_output_prices_within_bounds(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that output prices are within configured bounds."""
        config = mock_trained_engine.config
        result = mock_trained_engine.predict(sample_dataframe)

        prices = result[config.output.price_column]

        assert prices.min() >= config.pricing.min_price, "Prices below minimum"
        assert prices.max() <= config.pricing.max_price, "Prices above maximum"

    def test_output_confidence_in_range(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that confidence scores are in [0, 1]."""
        config = mock_trained_engine.config
        result = mock_trained_engine.predict(sample_dataframe)

        confidences = result[config.output.confidence_column]

        assert confidences.min() >= 0.0, "Confidence below 0"
        assert confidences.max() <= 1.0, "Confidence above 1"

    def test_result_has_same_index(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that result DataFrame has same index as input."""
        # Set a custom index
        sample_dataframe.index = [f"idx_{i}" for i in range(len(sample_dataframe))]

        result = mock_trained_engine.predict(sample_dataframe)

        pd.testing.assert_index_equal(
            result.index,
            sample_dataframe.index,
            obj="Index should be preserved",
        )

    def test_result_has_same_row_count(
        self, sample_dataframe, mock_trained_engine
    ):
        """Test that result has same number of rows as input."""
        result = mock_trained_engine.predict(sample_dataframe)

        assert len(result) == len(sample_dataframe), "Row count mismatch"

    def test_minimal_dataframe_works(
        self, minimal_dataframe, mock_trained_engine
    ):
        """Test that minimal DataFrame with required columns works."""
        result = mock_trained_engine.predict(minimal_dataframe)

        assert len(result) == len(minimal_dataframe)
        assert mock_trained_engine.config.output.price_column in result.columns

    def test_empty_dataframe_returns_empty(
        self, mock_trained_engine
    ):
        """Test that empty DataFrame returns empty result."""
        empty_df = pd.DataFrame({
            "product_id": [],
            "current_price": [],
            "inventory_level": [],
        })

        result = mock_trained_engine.predict(empty_df)

        assert len(result) == 0
        assert mock_trained_engine.config.output.price_column in result.columns


class TestDataFrameValidation:
    """Tests for DataFrame validation."""

    def test_missing_required_columns_raises(self, sample_config):
        """Test that missing required columns raise ValueError."""
        engine = PricingEngine(sample_config)
        engine._is_trained = True
        engine._rl_agent = MockRLAgent(sample_config)

        incomplete_df = pd.DataFrame({
            "product_id": ["A", "B", "C"],
            # Missing: current_price, inventory_level
        })

        with pytest.raises(ValueError) as exc_info:
            engine.predict(incomplete_df)

        assert "Missing required columns" in str(exc_info.value)

    def test_extra_columns_allowed(self, sample_dataframe, sample_config):
        """Test that extra columns are allowed and preserved."""
        engine = PricingEngine(sample_config)
        engine._is_trained = True
        engine._rl_agent = MockRLAgent(sample_config)

        # Add extra column
        sample_dataframe["extra_column"] = "extra_value"

        result = engine.predict(sample_dataframe)

        assert "extra_column" in result.columns
        assert (result["extra_column"] == "extra_value").all()
