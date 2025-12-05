"""Tests for the data utility functions."""

import numpy as np
import pandas as pd
import pytest

from neuroprice.data_utils import (
    detect_outliers,
    fill_missing_values,
    generate_synthetic_data,
    split_train_test,
    validate_input_data,
)


# ---------------------------------------------------------------------------
# validate_input_data
# ---------------------------------------------------------------------------

class TestValidateInputData:
    """Tests for validate_input_data."""

    def test_clean_data_passes(self):
        """Clean data with all columns should pass validation."""
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0],
            "demand": [100, 80, 60],
        })

        result = validate_input_data(df, required_columns=["price", "demand"])
        assert len(result) == 3
        assert list(result.columns) == ["price", "demand"]

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({"price": [10.0, 20.0]})

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(df, required_columns=["price", "demand"])

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise ValueError."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            validate_input_data(df)

    def test_no_required_columns(self):
        """Without required_columns, any non-empty DF should pass."""
        df = pd.DataFrame({"any_col": [1, 2, 3]})
        result = validate_input_data(df)
        assert len(result) == 3

    def test_with_nan_values(self):
        """Data with NaN values should pass validation (not rejected)."""
        df = pd.DataFrame({
            "price": [10.0, np.nan, 30.0],
            "demand": [100, 80, 60],
        })

        result = validate_input_data(df, required_columns=["price", "demand"])
        assert len(result) == 3
        assert result["price"].isna().sum() == 1

    def test_with_outliers_flagged(self):
        """With check_outliers=True, outlier column should be added."""
        df = pd.DataFrame({
            "price": [10, 12, 11, 13, 10, 12, 1000],  # 1000 is an outlier
            "demand": [50, 55, 48, 52, 50, 55, 49],
        })

        result = validate_input_data(df, check_outliers=True)
        assert "_has_outlier" in result.columns

    def test_returns_copy(self):
        """Should return a copy, not modify the original."""
        df = pd.DataFrame({"price": [10.0, 20.0]})
        result = validate_input_data(df)

        result["new_col"] = [1, 2]
        assert "new_col" not in df.columns


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------

class TestDetectOutliers:
    """Tests for detect_outliers with IQR method."""

    def test_detects_obvious_outlier(self):
        """Should detect an obvious outlier in the data."""
        df = pd.DataFrame({
            "value": [10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 1000],
        })

        mask = detect_outliers(df, columns=["value"])
        assert mask["value"].iloc[-1]  # 1000 is an outlier

    def test_no_outliers_in_uniform_data(self):
        """Uniform data should have no outliers."""
        df = pd.DataFrame({
            "value": [10.0] * 20,
        })

        mask = detect_outliers(df, columns=["value"])
        assert not mask.any().any()

    def test_iqr_method_structure(self):
        """Should return boolean DataFrame with same shape."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        mask = detect_outliers(df, columns=["a", "b"])
        assert mask.shape == df[["a", "b"]].shape
        assert all(mask.dtypes == bool)

    def test_custom_factor(self):
        """Larger factor should detect fewer outliers."""
        df = pd.DataFrame({
            "value": [10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 100],
        })

        strict_mask = detect_outliers(df, columns=["value"], factor=1.0)
        lenient_mask = detect_outliers(df, columns=["value"], factor=3.0)

        # Strict should detect >= lenient
        assert strict_mask["value"].sum() >= lenient_mask["value"].sum()

    def test_zscore_method(self):
        """Z-score method should detect extreme values."""
        df = pd.DataFrame({
            "value": [10, 12, 11, 13, 10, 12, 11, 13, 10, 12, 1000],
        })

        mask = detect_outliers(df, columns=["value"], method="zscore", factor=2.0)
        assert mask["value"].iloc[-1]  # 1000 is an outlier by z-score

    def test_defaults_to_all_numeric(self):
        """Without specifying columns, should check all numeric columns."""
        df = pd.DataFrame({
            "num": [1, 2, 3, 100],
            "cat": ["a", "b", "c", "d"],
        })

        mask = detect_outliers(df)
        assert "num" in mask.columns
        assert "cat" not in mask.columns


# ---------------------------------------------------------------------------
# fill_missing_values
# ---------------------------------------------------------------------------

class TestFillMissingValues:
    """Tests for fill_missing_values with different strategies."""

    def test_median_strategy(self):
        """Median strategy should fill NaN with column median."""
        df = pd.DataFrame({
            "value": [1.0, 2.0, np.nan, 4.0, 5.0],
        })

        result = fill_missing_values(df, strategy="median")
        assert result["value"].isna().sum() == 0
        assert result["value"].iloc[2] == pytest.approx(3.0)  # median of [1,2,4,5] = 3.0

    def test_mean_strategy(self):
        """Mean strategy should fill NaN with column mean."""
        df = pd.DataFrame({
            "value": [1.0, 2.0, np.nan, 4.0, 5.0],
        })

        result = fill_missing_values(df, strategy="mean")
        assert result["value"].isna().sum() == 0
        # mean of [1,2,4,5] = 3.0
        assert result["value"].iloc[2] == pytest.approx(3.0)

    def test_zero_strategy(self):
        """Zero strategy should fill NaN with 0."""
        df = pd.DataFrame({
            "value": [1.0, np.nan, 3.0],
        })

        result = fill_missing_values(df, strategy="zero")
        assert result["value"].iloc[1] == 0.0

    def test_ffill_strategy(self):
        """Forward fill should propagate last valid value."""
        df = pd.DataFrame({
            "value": [1.0, np.nan, np.nan, 4.0],
        })

        result = fill_missing_values(df, strategy="ffill")
        assert result["value"].iloc[1] == 1.0
        assert result["value"].iloc[2] == 1.0

    def test_specific_columns(self):
        """Should only fill specified columns."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 2.0, 3.0],
        })

        result = fill_missing_values(df, strategy="zero", columns=["a"])
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 1  # b not filled

    def test_returns_copy(self):
        """Should return a copy, not modify original."""
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        result = fill_missing_values(df, strategy="zero")

        assert df["value"].isna().sum() == 1  # Original unchanged

    def test_no_missing_values(self):
        """Data without NaN should be returned unchanged."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = fill_missing_values(df)
        assert result["value"].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# generate_synthetic_data
# ---------------------------------------------------------------------------

class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data."""

    def test_produces_valid_dataframe(self):
        """Should produce a DataFrame with expected columns."""
        df = generate_synthetic_data(n_samples=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "product_id" in df.columns
        assert "price" in df.columns
        assert "current_price" in df.columns
        assert "demand" in df.columns
        assert "inventory_level" in df.columns

    def test_with_confounders(self):
        """Should include confounder columns when requested."""
        df = generate_synthetic_data(n_samples=50, include_confounders=True)

        assert "day_of_week" in df.columns
        assert "promotion_active" in df.columns
        assert "competitor_price" in df.columns

    def test_without_confounders(self):
        """Should not include confounders when not requested."""
        df = generate_synthetic_data(n_samples=50, include_confounders=False)

        assert "day_of_week" not in df.columns
        assert "promotion_active" not in df.columns

    def test_reproducibility(self):
        """Same seed should produce same data."""
        df1 = generate_synthetic_data(n_samples=50, seed=123)
        df2 = generate_synthetic_data(n_samples=50, seed=123)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        df1 = generate_synthetic_data(n_samples=50, seed=1)
        df2 = generate_synthetic_data(n_samples=50, seed=2)

        assert not df1["price"].equals(df2["price"])

    def test_demand_non_negative(self):
        """All demand values should be non-negative."""
        df = generate_synthetic_data(n_samples=200)
        assert (df["demand"] >= 0).all()

    def test_price_in_range(self):
        """Prices should be within the specified range."""
        df = generate_synthetic_data(n_samples=200, price_range=(20.0, 80.0))
        assert df["price"].min() >= 20.0
        assert df["price"].max() <= 80.0

    def test_product_id_count(self):
        """Should have the correct number of distinct products."""
        df = generate_synthetic_data(n_samples=100, n_products=5)
        assert df["product_id"].nunique() == 5


# ---------------------------------------------------------------------------
# split_train_test
# ---------------------------------------------------------------------------

class TestSplitTrainTest:
    """Tests for split_train_test."""

    def test_random_split_ratio(self):
        """Random split should produce correct ratio."""
        df = pd.DataFrame({"x": range(100)})
        train, test = split_train_test(df, test_ratio=0.2)

        assert len(train) == 80
        assert len(test) == 20

    def test_time_aware_split(self):
        """Time-aware split should put later data in test."""
        df = pd.DataFrame({
            "x": range(100),
            "time": range(100),
        })

        train, test = split_train_test(df, test_ratio=0.2, time_aware=True, time_column="time")

        assert len(train) == 80
        assert len(test) == 20
        # Train should have earlier times
        assert train["time"].max() < test["time"].min()

    def test_time_aware_without_column(self):
        """Time-aware split without time_column should use original order."""
        df = pd.DataFrame({"x": range(100)})
        train, test = split_train_test(df, test_ratio=0.2, time_aware=True)

        assert len(train) == 80
        assert len(test) == 20

    def test_invalid_ratio_raises(self):
        """Invalid test_ratio should raise ValueError."""
        df = pd.DataFrame({"x": range(10)})

        with pytest.raises(ValueError, match="test_ratio"):
            split_train_test(df, test_ratio=0.0)

        with pytest.raises(ValueError, match="test_ratio"):
            split_train_test(df, test_ratio=1.0)

    def test_returns_copies(self):
        """Should return copies, not views."""
        df = pd.DataFrame({"x": range(100)})
        train, test = split_train_test(df, test_ratio=0.2)

        train["new"] = 1
        assert "new" not in df.columns


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in data utilities."""

    def test_single_row_dataframe(self):
        """Single-row DataFrame should work with validate."""
        df = pd.DataFrame({"price": [50.0]})
        result = validate_input_data(df, required_columns=["price"])
        assert len(result) == 1

    def test_all_nan_column_fill(self):
        """All-NaN column should be filled with NaN (median of NaN)."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
        result = fill_missing_values(df, strategy="median", columns=["a"])
        # Median of all NaN is NaN, so it stays NaN
        assert result["a"].isna().all()

    def test_detect_outliers_single_value(self):
        """Single value should not be flagged as outlier."""
        df = pd.DataFrame({"value": [42.0]})
        mask = detect_outliers(df, columns=["value"])
        assert not mask["value"].iloc[0]

    def test_validate_single_row(self):
        """Single row should pass validation."""
        df = pd.DataFrame({"price": [50.0], "demand": [100]})
        result = validate_input_data(df, required_columns=["price", "demand"])
        assert len(result) == 1

    def test_split_small_dataset(self):
        """Splitting a very small dataset should still work."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        train, test = split_train_test(df, test_ratio=0.4)

        assert len(train) + len(test) == 5
