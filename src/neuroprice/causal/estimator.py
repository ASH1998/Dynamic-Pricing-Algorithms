"""Causal inference estimator using EconML.

This module provides causal inference capabilities for understanding
the true price-demand relationship, accounting for confounding factors.
"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from neuroprice.exceptions import TrainingError
from neuroprice.models import StrategyConfig

# Import EconML with graceful fallback
try:
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, RidgeCV

    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    LinearDML = None  # type: ignore


class CausalEstimator:
    """EconML-based causal inference for price-demand relationships.

    This class uses Double Machine Learning (DML) to understand the true
    causal effect of price changes on demand, controlling for confounding
    factors like day of week, promotions, etc.

    Attributes:
        config: Strategy configuration
        model: EconML LinearDML model instance
        treatment_effect: Estimated average treatment effect

    Example:
        >>> estimator = CausalEstimator(config)
        >>> estimator.fit(historical_data)
        >>> prices, confidences = estimator.estimate_prices(products_df)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize the causal estimator.

        Args:
            config: Strategy configuration with causal parameters

        Raises:
            ImportError: If econml is not installed
        """
        if not ECONML_AVAILABLE:
            raise ImportError(
                "EconML is required for causal inference. "
                "Install it with: pip install econml"
            )

        self.config = config
        self.model: Optional[Any] = None
        self._is_fitted = False
        self._causal_effect: float = 0.0
        self._effect_stderr: float = 0.0

    @property
    def is_fitted(self) -> bool:
        """Check if the estimator has been fitted."""
        return self._is_fitted

    def fit(self, historical_data: pd.DataFrame, verbose: bool = False) -> "CausalEstimator":
        """Fit the causal model on historical data.

        Uses Double Machine Learning (DML) which:
        1. Residualizes the treatment (price) on confounders
        2. Residualizes the outcome (demand) on confounders
        3. Estimates treatment effect from residualized values

        Args:
            historical_data: DataFrame with treatment, outcome, and confounders
            verbose: Whether to print progress information

        Returns:
            Self for method chaining

        Raises:
            TrainingError: If fitting fails
        """
        try:
            treatment = self.config.causal_config.treatment_variable
            outcome = self.config.causal_config.outcome_variable
            confounders = self.config.causal_config.confounders

            # Validate required columns
            required_cols = [treatment, outcome] + confounders
            missing = set(required_cols) - set(historical_data.columns)
            if missing:
                raise ValueError(f"Missing columns in data: {missing}")

            # Prepare data for EconML
            Y = historical_data[outcome].values  # Outcome (demand)
            T = historical_data[treatment].values  # Treatment (price)
            W = historical_data[confounders].values if confounders else None  # Confounders

            # Create and fit LinearDML model
            # Uses Lasso for treatment model and Ridge for outcome model
            self.model = LinearDML(
                model_t=LassoCV(cv=3),
                model_y=RidgeCV(cv=3),
                discrete_treatment=False,
                cv=3,
                random_state=42,
            )

            self.model.fit(Y=Y, T=T, W=W)

            # Get the average treatment effect (ATE)
            self._causal_effect = float(self.model.ate())

            # Get standard error for confidence estimation
            ate_inference = self.model.ate_inference()
            self._effect_stderr = float(ate_inference.stderr)

            self._is_fitted = True

            if verbose:
                print(f"Causal effect of {treatment} on {outcome}: {self._causal_effect:.4f}")
                print(f"Standard error: {self._effect_stderr:.4f}")

            return self

        except Exception as e:
            raise TrainingError(
                message=str(e),
                model_type="causal",
            )

    def estimate_prices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate optimal prices based on causal model.

        Uses the estimated causal effect to determine optimal pricing
        that maximizes expected revenue.

        Args:
            df: Input DataFrame with product information

        Returns:
            Tuple of (prices array, confidences array)

        Raises:
            ValueError: If estimator is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before making predictions")

        prices = []
        confidences = []

        for _, row in df.iterrows():
            current_price = row.get("current_price", 50.0)

            # Compute optimal price based on causal effect
            optimal_price = self._compute_optimal_price(current_price, row)
            prices.append(optimal_price)

            # Compute confidence
            confidence = self._compute_confidence(row)
            confidences.append(confidence)

        return np.array(prices), np.array(confidences)

    def get_causal_effect(self) -> float:
        """Get the estimated causal effect.

        Returns:
            Estimated effect of treatment on outcome
        """
        return self._causal_effect

    def get_heterogeneous_effect(self, X: pd.DataFrame) -> np.ndarray:
        """Get heterogeneous treatment effects for specific observations.

        EconML allows estimating conditional average treatment effects
        (CATE) that vary across different contexts.

        Args:
            X: DataFrame with features for effect estimation

        Returns:
            Array of treatment effects for each observation
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting effects")

        confounders = self.config.causal_config.confounders
        if confounders and all(c in X.columns for c in confounders):
            W = X[confounders].values
            return self.model.effect(X=None, T0=0, T1=1)
        else:
            # Return ATE for all observations
            return np.full(len(X), self._causal_effect)

    def _compute_optimal_price(self, current_price: float, row: pd.Series) -> float:
        """Compute optimal price based on causal effect and elasticity.

        Uses the relationship:
        - If causal effect is negative (price increase -> demand decrease),
          we want to find the revenue-maximizing price
        - Optimal price formula from elasticity: P* = P / (1 + 1/elasticity)

        Args:
            current_price: Current product price
            row: DataFrame row with product features

        Returns:
            Recommended optimal price
        """
        elasticity = self.config.demand.elasticity

        # Compute revenue-maximizing price using elasticity
        # For demand D = a * P^e, revenue R = P * D = a * P^(e+1)
        # dR/dP = a * (e+1) * P^e = 0 when e = -1
        # For e != -1, optimal is at boundary or where marginal revenue = 0

        if abs(elasticity + 1) < 0.01:
            # Unit elastic - any price gives same revenue
            optimal = current_price
        elif elasticity < -1:
            # Elastic demand - lower price increases revenue
            # Adjust based on causal effect magnitude
            adjustment = 1 - 0.1 * abs(self._causal_effect)
            optimal = current_price * max(0.8, adjustment)
        else:
            # Inelastic demand - higher price increases revenue
            adjustment = 1 + 0.1 * abs(self._causal_effect)
            optimal = current_price * min(1.2, adjustment)

        # Clamp to configured bounds
        return float(np.clip(
            optimal,
            self.config.pricing.min_price,
            self.config.pricing.max_price,
        ))

    def _compute_confidence(self, row: pd.Series) -> float:
        """Compute confidence score for the price prediction.

        Confidence is based on:
        - Standard error of the treatment effect estimate
        - Whether we have confounder values
        - Statistical significance (effect / stderr)

        Args:
            row: DataFrame row with product features

        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.7

        # Adjust based on statistical significance
        if self._effect_stderr > 0:
            t_stat = abs(self._causal_effect / self._effect_stderr)
            # Higher t-stat = more confidence
            if t_stat > 2.0:  # ~95% confidence
                base_confidence += 0.15
            elif t_stat > 1.65:  # ~90% confidence
                base_confidence += 0.1

        # Increase confidence if confounders are present
        confounders_present = sum(
            1 for c in self.config.causal_config.confounders
            if c in row and pd.notna(row.get(c))
        )
        total_confounders = len(self.config.causal_config.confounders)

        if total_confounders > 0:
            confounder_bonus = 0.1 * (confounders_present / total_confounders)
        else:
            confounder_bonus = 0.05

        confidence = base_confidence + confounder_bonus
        return min(1.0, confidence)

    def effect_inference(self) -> dict:
        """Get inference results for the treatment effect.

        Returns:
            Dictionary with effect estimate, confidence interval, and p-value
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before inference")

        inference = self.model.ate_inference()

        return {
            "effect": self._causal_effect,
            "stderr": self._effect_stderr,
            "confidence_interval": (
                float(inference.conf_int()[0][0]),
                float(inference.conf_int()[0][1]),
            ),
            "pvalue": float(inference.pvalue()),
        }

    def summary(self) -> str:
        """Get a summary of the causal model.

        Returns:
            String summary of the model and estimates
        """
        if not self.is_fitted:
            return "Model not fitted"

        inference = self.effect_inference()
        ci_low, ci_high = inference["confidence_interval"]

        return (
            f"EconML LinearDML Causal Estimator\n"
            f"{'=' * 40}\n"
            f"Treatment: {self.config.causal_config.treatment_variable}\n"
            f"Outcome: {self.config.causal_config.outcome_variable}\n"
            f"Confounders: {', '.join(self.config.causal_config.confounders)}\n"
            f"\nAverage Treatment Effect: {self._causal_effect:.4f}\n"
            f"Standard Error: {self._effect_stderr:.4f}\n"
            f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n"
            f"P-value: {inference['pvalue']:.4f}"
        )
