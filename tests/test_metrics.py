"""Tests for the evaluation metrics module."""

import numpy as np
import pytest

from neuroprice.metrics import mae, mape, price_distribution_summary, revenue_comparison, rmse


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self):
        """MAE should compute correctly for known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # |1-2| + |2-3| + |3-4| = 1+1+1 = 3.0, mean = 1.0
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_symmetric(self):
        """MAE should be symmetric (over/under predictions penalized equally)."""
        y_true = np.array([10.0, 20.0])
        y_pred_high = np.array([15.0, 25.0])  # +5 each
        y_pred_low = np.array([5.0, 15.0])    # -5 each

        assert mae(y_true, y_pred_high) == pytest.approx(mae(y_true, y_pred_low))

    def test_with_lists(self):
        """Should accept lists as input."""
        result = mae([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        assert result == pytest.approx(0.5)

    def test_single_value(self):
        """MAE with single value should be absolute difference."""
        assert mae([5.0], [8.0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    """Tests for Root Mean Squared Error."""

    def test_perfect_predictions(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self):
        """RMSE should compute correctly for known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # MSE = (1+1+1)/3 = 1.0, RMSE = 1.0
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_penalizes_large_errors(self):
        """RMSE should penalize large errors more than MAE."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 10.0])

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)

        # RMSE > MAE when there are large errors
        assert rmse_val > mae_val

    def test_with_lists(self):
        """Should accept lists as input."""
        result = rmse([1.0, 2.0], [1.0, 2.0])
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------

class TestMAPE:
    """Tests for Mean Absolute Percentage Error."""

    def test_perfect_predictions(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        assert mape(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self):
        """MAPE should compute correctly for known values."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |100-110|/100 = 0.1, |200-180|/200 = 0.1
        # mean = 0.1
        assert mape(y_true, y_pred) == pytest.approx(0.1)

    def test_with_zeros_handling(self):
        """MAPE with zeros should use epsilon to avoid division by zero."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 180.0])

        # With default epsilon, zero is replaced with epsilon
        result = mape(y_true, y_pred)
        # Should be finite (not inf/nan)
        assert np.isfinite(result)
        # Should be large due to the 0 -> epsilon replacement
        assert result > 0.0

    def test_custom_epsilon(self):
        """Custom epsilon should control zero-division behavior."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])

        result_small = mape(y_true, y_pred, epsilon=1e-10)
        result_large = mape(y_true, y_pred, epsilon=1.0)

        # Larger epsilon means the zero is replaced with a larger value
        # so the error from the zero row is smaller
        assert result_large < result_small

    def test_small_true_values(self):
        """MAPE with small true values should be computed correctly."""
        y_true = np.array([0.01, 0.02])
        y_pred = np.array([0.02, 0.04])
        # |0.01-0.02|/0.01 = 1.0, |0.02-0.04|/0.02 = 1.0
        assert mape(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# revenue_comparison
# ---------------------------------------------------------------------------

class TestRevenueComparison:
    """Tests for revenue_comparison."""

    def test_same_prices_zero_improvement(self):
        """Same actual and recommended prices should give zero improvement."""
        actual = np.array([50.0, 60.0, 70.0])
        recommended = np.array([50.0, 60.0, 70.0])
        demands = np.array([100, 80, 60])

        result = revenue_comparison(actual, recommended, demands)
        assert result["revenue_improvement"] == pytest.approx(0.0, abs=1e-6)
        assert result["improvement_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_structure(self):
        """Result should have expected keys."""
        result = revenue_comparison([50.0, 60.0], [45.0, 55.0], [100, 80])

        assert "actual_revenue" in result
        assert "recommended_revenue" in result
        assert "revenue_improvement" in result
        assert "improvement_pct" in result

    def test_higher_recommended_price(self):
        """Higher recommended prices should increase revenue (same demand)."""
        actual = np.array([50.0, 60.0])
        recommended = np.array([60.0, 70.0])
        demands = np.array([100, 80])

        result = revenue_comparison(actual, recommended, demands)
        # recommended_revenue > actual_revenue when prices go up and demand stays same
        assert result["revenue_improvement"] > 0
        assert result["improvement_pct"] > 0

    def test_revenue_calculation(self):
        """Revenue should be price * demand summed."""
        actual = np.array([50.0, 60.0])
        recommended = np.array([50.0, 60.0])
        demands = np.array([10, 20])

        result = revenue_comparison(actual, recommended, demands)
        # actual: 50*10 + 60*20 = 500 + 1200 = 1700
        assert result["actual_revenue"] == pytest.approx(1700.0)

    def test_with_lists(self):
        """Should accept list inputs."""
        result = revenue_comparison([50.0], [60.0], [100])
        assert "actual_revenue" in result


# ---------------------------------------------------------------------------
# price_distribution_summary
# ---------------------------------------------------------------------------

class TestPriceDistributionSummary:
    """Tests for price_distribution_summary."""

    def test_structure(self):
        """Should return dict with expected keys."""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = price_distribution_summary(prices)

        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "count" in result

    def test_known_values(self):
        """Should compute correct statistics."""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = price_distribution_summary(prices)

        assert result["mean"] == pytest.approx(30.0)
        assert result["min"] == pytest.approx(10.0)
        assert result["max"] == pytest.approx(50.0)
        assert result["count"] == 5

    def test_default_percentiles(self):
        """Should include default percentiles (25, 50, 75, 90, 95)."""
        prices = np.array(range(1, 101), dtype=float)
        result = price_distribution_summary(prices)

        assert "p25" in result
        assert "p50" in result
        assert "p75" in result
        assert "p90" in result
        assert "p95" in result

    def test_custom_percentiles(self):
        """Should support custom percentile values."""
        prices = np.array(range(1, 101), dtype=float)
        result = price_distribution_summary(prices, percentiles=[10.0, 99.0])

        assert "p10" in result
        assert "p99" in result
        assert "p50" not in result

    def test_single_value(self):
        """Single value should have zero std."""
        prices = np.array([42.0])
        result = price_distribution_summary(prices)

        assert result["mean"] == pytest.approx(42.0)
        assert result["std"] == pytest.approx(0.0)
        assert result["count"] == 1

    def test_with_list_input(self):
        """Should accept list input."""
        result = price_distribution_summary([1.0, 2.0, 3.0])
        assert result["mean"] == pytest.approx(2.0)
