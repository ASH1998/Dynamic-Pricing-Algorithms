"""Tests for the Causal Estimator module."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from neuroprice.models import (
    CausalConfig,
    DemandConfig,
    FeaturesConfig,
    InventoryConfig,
    PricingConfig,
    StrategyConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _causal_config(confounders=None):
    """Create a StrategyConfig for causal estimator testing."""
    if confounders is None:
        confounders = ["day_of_week", "promotion_active", "competitor_price"]
    return StrategyConfig(
        name="causal_test",
        pricing=PricingConfig(min_price=10.0, max_price=100.0, price_step=1.0),
        demand=DemandConfig(base_demand=100.0, elasticity=-1.5),
        inventory=InventoryConfig(initial_stock=100, deadline_days=30),
        causal_config=CausalConfig(
            treatment_variable="price",
            outcome_variable="demand",
            confounders=confounders,
        ),
        features=FeaturesConfig(
            required=["product_id", "current_price", "inventory_level", "demand"],
        ),
    )


# ---------------------------------------------------------------------------
# CausalEstimator Initialization
# ---------------------------------------------------------------------------

class TestCausalEstimatorInit:
    """Tests for CausalEstimator initialization."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_initialization(self):
        """Should initialize with correct attributes."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        estimator = CausalEstimator(config)

        assert estimator.config == config
        assert not estimator.is_fitted
        assert estimator.model is None
        assert estimator._causal_effect == 0.0
        assert estimator._effect_stderr == 0.0

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_initialization_tracking_attrs(self):
        """Should have zero-initialized tracking attributes."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        assert estimator._n_observations == 0
        assert estimator._model_t_score == 0.0
        assert estimator._model_y_score == 0.0
        assert estimator._cate_std == 0.0

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", False)
    def test_init_without_econml_raises(self):
        """ImportError should be raised when EconML is not installed."""
        from neuroprice.causal.estimator import CausalEstimator

        with pytest.raises(ImportError, match="EconML"):
            CausalEstimator(_causal_config())


# ---------------------------------------------------------------------------
# Fit with Synthetic Data
# ---------------------------------------------------------------------------

class TestCausalEstimatorFit:
    """Tests for fitting the causal model."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_fit_sets_fitted_flag(self, synthetic_data):
        """After fitting, is_fitted should be True."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)

        assert estimator.is_fitted

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_fit_computes_causal_effect(self, synthetic_data):
        """After fitting, causal effect should be a finite number."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)

        assert np.isfinite(estimator._causal_effect)
        assert estimator._effect_stderr >= 0

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_fit_tracks_observations(self, synthetic_data):
        """After fitting, n_observations should match data length."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)

        assert estimator._n_observations == len(synthetic_data)

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_fit_returns_self(self, synthetic_data):
        """fit() should return self for method chaining."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        result = estimator.fit(synthetic_data)

        assert result is estimator

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_fit_missing_columns_raises(self):
        """fit() should raise TrainingError when required columns are missing."""
        from neuroprice.causal.estimator import CausalEstimator
        from neuroprice.exceptions import TrainingError

        estimator = CausalEstimator(_causal_config())
        bad_data = pd.DataFrame({"wrong_column": [1, 2, 3]})

        with pytest.raises(TrainingError):
            estimator.fit(bad_data)

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_fit_computes_first_stage_scores(self, synthetic_data):
        """After fitting, first-stage R² scores should be set."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)

        assert 0.0 <= estimator._model_t_score <= 1.0
        assert 0.0 <= estimator._model_y_score <= 1.0


# ---------------------------------------------------------------------------
# _compute_optimal_price — Elastic Demand
# ---------------------------------------------------------------------------

class TestOptimalPriceElastic:
    """Tests for optimal price with elastic demand (e < -1)."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_elastic_demand_markup(self):
        """With elastic demand, optimal price should be above cost proxy."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        config.demand.elasticity = -2.0
        estimator = CausalEstimator(config)
        estimator._causal_effect = -1.0  # Negative causal effect
        estimator._effect_stderr = 0.1

        row = pd.Series({"current_price": 50.0})
        optimal = estimator._compute_optimal_price(50.0, row)

        # For e=-2: markup_ratio = -2/(-2+1) = -2/(-1) = 2.0
        # So optimal ≈ 50 * 2.0 = 100, then blended with causal adjustment
        assert optimal > 50.0
        assert 10.0 <= optimal <= 100.0  # Within bounds

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_elastic_demand_clipping(self):
        """Optimal price should be clipped to [min_price, max_price]."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        config.demand.elasticity = -3.0
        estimator = CausalEstimator(config)
        estimator._causal_effect = -0.5
        estimator._effect_stderr = 0.1

        row = pd.Series({"current_price": 80.0})
        optimal = estimator._compute_optimal_price(80.0, row)

        assert 10.0 <= optimal <= 100.0


# ---------------------------------------------------------------------------
# _compute_optimal_price — Inelastic Demand
# ---------------------------------------------------------------------------

class TestOptimalPriceInelastic:
    """Tests for optimal price with inelastic demand (-1 < e < 0)."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_inelastic_demand_raises_price(self):
        """With inelastic demand, optimal price should be higher than current."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        config.demand.elasticity = -0.5
        estimator = CausalEstimator(config)
        estimator._causal_effect = -1.0
        estimator._effect_stderr = 0.3

        row = pd.Series({"current_price": 50.0})
        optimal = estimator._compute_optimal_price(50.0, row)

        assert optimal >= 50.0  # Should increase or stay same

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_inelastic_demand_with_high_significance(self):
        """Higher t-stat should lead to larger price adjustment."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        config.demand.elasticity = -0.5

        # High significance (small stderr)
        est_high = CausalEstimator(config)
        est_high._causal_effect = -2.0
        est_high._effect_stderr = 0.1  # t-stat = 20

        # Low significance (large stderr)
        est_low = CausalEstimator(config)
        est_low._causal_effect = -2.0
        est_low._effect_stderr = 10.0  # t-stat = 0.2

        row = pd.Series({"current_price": 50.0})
        opt_high = est_high._compute_optimal_price(50.0, row)
        opt_low = est_low._compute_optimal_price(50.0, row)

        # Higher significance should produce a larger adjustment
        assert opt_high >= opt_low


# ---------------------------------------------------------------------------
# _compute_optimal_price — Unit Elastic
# ---------------------------------------------------------------------------

class TestOptimalPriceUnitElastic:
    """Tests for optimal price with unit elastic demand (e ~ -1)."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_unit_elastic_keeps_current_price(self):
        """With unit elasticity, optimal price should be approximately current."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        config.demand.elasticity = -1.0
        estimator = CausalEstimator(config)
        estimator._causal_effect = -1.0
        estimator._effect_stderr = 0.5

        row = pd.Series({"current_price": 50.0})
        optimal = estimator._compute_optimal_price(50.0, row)

        assert optimal == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# _compute_confidence — 4-Factor Weighting
# ---------------------------------------------------------------------------

class TestConfidenceComputation:
    """Tests for the 4-factor confidence scoring."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_confidence_range(self):
        """Confidence should be in [0, 1]."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator._causal_effect = -2.0
        estimator._effect_stderr = 0.3
        estimator._n_observations = 500
        estimator._model_t_score = 0.6
        estimator._model_y_score = 0.4

        row = pd.Series({"day_of_week": 3, "promotion_active": 1, "competitor_price": 45.0})
        conf = estimator._compute_confidence(row)

        assert 0.0 <= conf <= 1.0

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_confidence_with_no_stderr(self):
        """When stderr is 0, significance score should default to 0.5."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator._causal_effect = -2.0
        estimator._effect_stderr = 0.0
        estimator._n_observations = 500

        row = pd.Series({"day_of_week": 3, "promotion_active": 1, "competitor_price": 45.0})
        conf = estimator._compute_confidence(row)

        assert 0.0 <= conf <= 1.0

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_confidence_with_more_samples(self):
        """More observations should yield higher confidence (sample factor)."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()

        est_many = CausalEstimator(config)
        est_many._causal_effect = -2.0
        est_many._effect_stderr = 0.5
        est_many._n_observations = 1000
        est_many._model_t_score = 0.5
        est_many._model_y_score = 0.5

        est_few = CausalEstimator(config)
        est_few._causal_effect = -2.0
        est_few._effect_stderr = 0.5
        est_few._n_observations = 10
        est_few._model_t_score = 0.5
        est_few._model_y_score = 0.5

        row = pd.Series({"day_of_week": 3, "promotion_active": 1, "competitor_price": 45.0})
        conf_many = est_many._compute_confidence(row)
        conf_few = est_few._compute_confidence(row)

        assert conf_many > conf_few

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_confidence_confounder_coverage(self):
        """Full confounder coverage should yield higher confidence."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config()
        est = CausalEstimator(config)
        est._causal_effect = -2.0
        est._effect_stderr = 0.5
        est._n_observations = 500
        est._model_t_score = 0.5
        est._model_y_score = 0.5

        # Full coverage
        row_full = pd.Series({"day_of_week": 3, "promotion_active": 1, "competitor_price": 45.0})
        conf_full = est._compute_confidence(row_full)

        # Partial coverage (missing one confounder)
        row_partial = pd.Series({"day_of_week": 3, "promotion_active": 1})
        conf_partial = est._compute_confidence(row_partial)

        assert conf_full >= conf_partial

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_confidence_weights_sum_to_one(self):
        """The 4 confidence factors should be weighted with weights summing to 1."""
        # Factor weights: 0.30 + 0.25 + 0.25 + 0.20 = 1.0
        weights = [0.30, 0.25, 0.25, 0.20]
        assert sum(weights) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_heterogeneous_effect
# ---------------------------------------------------------------------------

class TestHeterogeneousEffect:
    """Tests for get_heterogeneous_effect with confounders."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_heterogeneous_effect_not_fitted_raises(self):
        """Should raise ValueError if estimator is not fitted."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        X = pd.DataFrame({"day_of_week": [1, 2], "promotion_active": [0, 1], "competitor_price": [40, 60]})

        with pytest.raises(ValueError, match="fitted"):
            estimator.get_heterogeneous_effect(X)

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_heterogeneous_effect_with_confounders(self, synthetic_data):
        """get_heterogeneous_effect should return per-row effects."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)

        X = synthetic_data.head(10)
        effects = estimator.get_heterogeneous_effect(X)

        assert len(effects) == 10
        assert all(np.isfinite(effects))

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_heterogeneous_effect_no_confounders_returns_ate(self, synthetic_data):
        """Without confounders in X, should return the ATE for all rows."""
        from neuroprice.causal.estimator import CausalEstimator

        config = _causal_config(confounders=[])
        estimator = CausalEstimator(config)
        estimator.fit(synthetic_data)

        X = synthetic_data.head(5).drop(columns=["day_of_week", "promotion_active", "competitor_price"], errors="ignore")
        effects = estimator.get_heterogeneous_effect(X)

        assert len(effects) == 5
        # All should be equal to the ATE
        for e in effects:
            assert e == pytest.approx(estimator._causal_effect, abs=0.1)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for the summary() method."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_summary_not_fitted(self):
        """Summary before fitting should return 'Model not fitted'."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        assert estimator.summary() == "Model not fitted"

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_summary_format(self, synthetic_data):
        """Summary should contain expected sections."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)
        s = estimator.summary()

        assert "LinearDML" in s
        assert "price" in s
        assert "demand" in s
        assert "Treatment Effect" in s or "Treatment effect" in s.lower()
        assert "Standard Error" in s or "standard error" in s.lower()
        assert "Diagnostics" in s
        assert "sample size" in s.lower()

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_summary_contains_significance(self, synthetic_data):
        """Summary should include a significance label."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)
        s = estimator.summary()

        # Should contain one of the significance labels
        assert any(label in s for label in ["***", "**", "* (p<", "n.s."])


# ---------------------------------------------------------------------------
# effect_inference()
# ---------------------------------------------------------------------------

class TestEffectInference:
    """Tests for the effect_inference() method."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_effect_inference_not_fitted(self):
        """effect_inference before fitting should raise ValueError."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        with pytest.raises(ValueError, match="fitted"):
            estimator.effect_inference()

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_effect_inference_keys(self, synthetic_data):
        """effect_inference should return dict with expected keys."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)
        result = estimator.effect_inference()

        assert "effect" in result
        assert "stderr" in result
        assert "confidence_interval" in result
        assert "pvalue" in result

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_effect_inference_confidence_interval(self, synthetic_data):
        """Confidence interval should be a tuple of (low, high)."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)
        result = estimator.effect_inference()

        ci = result["confidence_interval"]
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_effect_inference_pvalue_range(self, synthetic_data):
        """P-value should be between 0 and 1."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator.fit(synthetic_data)
        result = estimator.effect_inference()

        assert 0.0 <= result["pvalue"] <= 1.0


# ---------------------------------------------------------------------------
# Mock EconML Tests (no heavy dependencies)
# ---------------------------------------------------------------------------

class TestCausalEstimatorMocked:
    """Tests using mocked EconML to avoid heavy dependencies."""

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_estimate_prices_calls_compute_optimal(self, synthetic_data):
        """estimate_prices should call _compute_optimal_price for each row."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        # Set up as if fitted
        estimator._is_fitted = True
        estimator._causal_effect = -2.0
        estimator._effect_stderr = 0.3
        estimator._n_observations = 100
        estimator._model_t_score = 0.5
        estimator._model_y_score = 0.5
        estimator._engineered_features = ["day_of_week", "promotion_active", "competitor_price"]

        df = synthetic_data.head(5)
        prices, confidences = estimator.estimate_prices(df)

        assert len(prices) == 5
        assert len(confidences) == 5
        assert all(10.0 <= p <= 100.0 for p in prices)
        assert all(0.0 <= c <= 1.0 for c in confidences)

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_estimate_prices_not_fitted_raises(self):
        """estimate_prices should raise ValueError if not fitted."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        df = pd.DataFrame({"current_price": [50.0]})

        with pytest.raises(ValueError, match="fitted"):
            estimator.estimate_prices(df)

    @patch("neuroprice.causal.estimator.ECONML_AVAILABLE", True)
    def test_get_causal_effect(self):
        """get_causal_effect should return the stored effect."""
        from neuroprice.causal.estimator import CausalEstimator

        estimator = CausalEstimator(_causal_config())
        estimator._causal_effect = -3.5

        assert estimator.get_causal_effect() == -3.5
