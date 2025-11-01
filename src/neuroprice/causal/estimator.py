"""Causal inference estimator using EconML.

This module provides causal inference capabilities for understanding
the true price-demand relationship, accounting for confounding factors.

Uses Double Machine Learning (DML) with first-stage nuisance models to
produce unbiased estimates of the causal effect of price on demand.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from neuroprice.exceptions import TrainingError
from neuroprice.models import StrategyConfig

# Import EconML with graceful fallback
try:
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.preprocessing import PolynomialFeatures

    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    LinearDML = None  # type: ignore


class CausalEstimator:
    """EconML-based causal inference for price-demand relationships.

    This class uses Double Machine Learning (DML) to understand the true
    causal effect of price changes on demand, controlling for confounding
    factors like day of week, promotions, competitor prices, etc.

    The estimator supports:
    - Average treatment effect (ATE) estimation
    - Conditional average treatment effect (CATE) for heterogeneous effects
    - Revenue-maximizing price computation via Lerner index / grid search
    - Confidence scoring based on statistical significance, sample size,
      and first-stage model fit quality

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

        # Enhanced tracking attributes
        self._n_observations: int = 0
        self._model_t_score: float = 0.0  # R² of treatment nuisance model
        self._model_y_score: float = 0.0  # R² of outcome nuisance model
        self._cate_std: float = 0.0  # Std of CATE estimates (heterogeneity)
        self._engineered_features: List[str] = []
        self._training_W: Optional[np.ndarray] = None  # Stored for inference

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

        Also computes first-stage R² scores to assess how well confounders
        explain variation in treatment and outcome.

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

            # Store sample size
            self._n_observations = len(historical_data)

            # Prepare data for EconML
            Y = historical_data[outcome].values  # Outcome (demand)
            T = historical_data[treatment].values  # Treatment (price)

            # Engineer features from confounders (adds interactions / polynomial terms)
            if confounders:
                W, confounder_names = self._build_confounder_matrix(historical_data, confounders)
            else:
                W = None
                confounder_names = []

            # Compute first-stage R² scores before DML fitting
            self._compute_first_stage_scores(T, Y, W, confounder_names)

            # Create and fit LinearDML model
            # Uses Lasso for treatment model and Ridge for outcome model
            self.model = LinearDML(
                model_t=LassoCV(cv=3),
                model_y=RidgeCV(cv=3),
                discrete_treatment=False,
                cv=3,
                random_state=42,
            )

            self.model.fit(Y=Y, T=T, X=W, W=W)

            # Store training W for later inference calls
            self._training_W = W

            # Get the average treatment effect (ATE)
            self._causal_effect = float(self.model.ate(X=W))

            # Get standard error for confidence estimation
            ate_inference = self.model.ate_inference(X=W)
            self._effect_stderr = float(ate_inference.stderr_mean)

            # Compute CATE heterogeneity on training data
            self._compute_cate_heterogeneity(W)

            self._is_fitted = True

            if verbose:
                print(f"Causal effect of {treatment} on {outcome}: {self._causal_effect:.4f}")
                print(f"Standard error: {self._effect_stderr:.4f}")
                print(f"First-stage treatment model R²: {self._model_t_score:.4f}")
                print(f"First-stage outcome model R²: {self._model_y_score:.4f}")
                print(f"Training observations: {self._n_observations}")

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
        """Get heterogeneous treatment effects (CATE) for specific observations.

        EconML allows estimating conditional average treatment effects
        that vary across different contexts defined by confounders.

        Args:
            X: DataFrame with features for effect estimation. Should contain
               the confounder columns used during fitting.

        Returns:
            Array of treatment effects for each observation
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting effects")

        confounders = self.config.causal_config.confounders

        if confounders and all(c in X.columns for c in confounders):
            # Build W from confounders, using same engineering as in fit()
            W, _ = self._build_confounder_matrix(X, confounders)
            return self.model.effect(X=W, T0=0, T1=1)
        elif confounders:
            # Some confounders present — return ATE (model was fit with W, not X)
            return np.full(len(X), self._causal_effect)
        else:
            # Return ATE for all observations
            return np.full(len(X), self._causal_effect)

    def estimate_optimal_prices(
        self,
        df: pd.DataFrame,
        n_candidates: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate per-row revenue-maximizing prices using CATE grid search.

        For each row, searches over a grid of candidate prices and picks
        the one that maximizes expected revenue: R(P) = P * D(P), where
        D(P) = current_demand + CATE(current_price, P).

        Args:
            df: DataFrame with product information. Must contain 'current_price'
                and confounder columns.
            n_candidates: Number of price candidates to evaluate per row.

        Returns:
            Tuple of:
                - optimal_prices: Per-row revenue-maximizing prices
                - revenues: Expected revenue at each optimal price
                - confidences: Confidence scores for each recommendation

        Raises:
            ValueError: If estimator is not fitted or required columns are missing
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before making predictions")

        treatment = self.config.causal_config.treatment_variable
        outcome = self.config.causal_config.outcome_variable
        confounders = self.config.causal_config.confounders
        min_p = self.config.pricing.min_price
        max_p = self.config.pricing.max_price

        # Build candidate price grid
        candidates = np.linspace(min_p, max_p, n_candidates)

        # Build confounder matrix if available
        W = None
        if confounders and all(c in df.columns for c in confounders):
            W, _ = self._build_confounder_matrix(df, confounders)

        optimal_prices = np.empty(len(df))
        optimal_revenues = np.empty(len(df))
        confidences = np.empty(len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            current_price = float(row.get("current_price", 50.0))
            current_demand = float(row.get(outcome, self.config.demand.base_demand))

            best_revenue = -np.inf
            best_price = current_price

            for candidate in candidates:
                # Estimate demand at candidate price via CATE
                # CATE gives E[D(candidate) - D(current_price) | X]
                if W is not None:
                    w_row = W[i : i + 1]
                    demand_change = float(
                        self.model.effect(X=w_row, T0=current_price, T1=candidate)[0]
                    )
                else:
                    demand_change = float(
                        self.model.effect(X=None, T0=current_price, T1=candidate)[0]
                    )

                expected_demand = current_demand + demand_change
                revenue = candidate * max(0.0, expected_demand)

                if revenue > best_revenue:
                    best_revenue = revenue
                    best_price = candidate

            optimal_prices[i] = best_price
            optimal_revenues[i] = best_revenue
            confidences[i] = self._compute_confidence(row)

        return optimal_prices, optimal_revenues, confidences

    def _build_confounder_matrix(
        self, df: pd.DataFrame, confounders: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Build and optionally engineer features from confounders.

        Extracts confounder values and applies feature engineering:
        - Interaction terms between confounders (if 2+ confounders)
        - Polynomial (quadratic) features for numeric confounders

        This enriches the W matrix so the nuisance models can better
        capture confounding variation, improving CATE estimates.

        Args:
            df: Input DataFrame
            confounders: List of confounder column names

        Returns:
            Tuple of (W matrix, feature names)
        """
        W_raw = df[confounders].values.astype(float)

        # Drop rows with NaN confounders, fill with column median for partial data
        col_medians = np.nanmedian(W_raw, axis=0)
        nan_mask = np.isnan(W_raw)
        if nan_mask.any():
            # Broadcast median to fill NaN
            inds = np.where(nan_mask)
            W_raw[inds] = np.take(col_medians, inds[1])

        # Apply feature engineering: interactions + quadratic terms
        # Only when there are enough confounders to benefit
        if len(confounders) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            W_engineered = poly.fit_transform(W_raw)
            feature_names = poly.get_feature_names_out(confounders).tolist()
            self._engineered_features = feature_names
            return W_engineered, feature_names
        else:
            self._engineered_features = confounders
            return W_raw, confounders

    def _compute_first_stage_scores(
        self,
        T: np.ndarray,
        Y: np.ndarray,
        W: Optional[np.ndarray],
        feature_names: List[str],
    ) -> None:
        """Compute R² scores for first-stage nuisance models.

        Fits simple models predicting treatment from confounders and
        outcome from confounders, storing their R² scores. These scores
        quantify how well confounders explain variation in T and Y,
        which directly affects the quality of causal estimation.

        Args:
            T: Treatment array
            Y: Outcome array
            W: Confounder matrix (may be None)
            feature_names: Names of confounder features
        """
        if W is None or W.shape[1] == 0:
            self._model_t_score = 0.0
            self._model_y_score = 0.0
            return

        try:
            from sklearn.model_selection import cross_val_score

            # Treatment model: how well do confounders explain price?
            t_model = RidgeCV(cv=min(3, max(2, self._n_observations // 10)))
            t_scores = cross_val_score(
                t_model, W, T, cv=min(3, max(2, self._n_observations // 10)), scoring="r2"
            )
            self._model_t_score = float(np.clip(np.mean(t_scores), 0.0, 1.0))

            # Outcome model: how well do confounders explain demand?
            y_model = RidgeCV(cv=min(3, max(2, self._n_observations // 10)))
            y_scores = cross_val_score(
                y_model, W, Y, cv=min(3, max(2, self._n_observations // 10)), scoring="r2"
            )
            self._model_y_score = float(np.clip(np.mean(y_scores), 0.0, 1.0))

        except Exception:
            # Graceful fallback if cross-validation fails
            self._model_t_score = 0.0
            self._model_y_score = 0.0

    def _compute_cate_heterogeneity(self, W: Optional[np.ndarray]) -> None:
        """Compute the standard deviation of CATE estimates.

        Measures how much the treatment effect varies across the
        population — a key indicator of effect heterogeneity.

        Args:
            W: Confounder matrix used during fitting
        """
        if W is not None and len(W) > 0:
            try:
                cate_estimates = self.model.effect(X=W, T0=0, T1=1)
                self._cate_std = float(np.std(cate_estimates))
            except Exception:
                self._cate_std = 0.0
        else:
            self._cate_std = 0.0

    def _compute_optimal_price(self, current_price: float, row: pd.Series) -> float:
        """Compute optimal price based on causal effect and elasticity.

        Uses principled economic formulas instead of hardcoded adjustments:

        1. **Lerner index** (elastic demand, known elasticity):
           P* = current_price / (1 + 1/elasticity)
           This is the profit-maximizing markup when current_price ≈ marginal cost.

        2. **Revenue-maximizing price** (elastic demand):
           For D(P) = D_ref * (P/P_ref)^e, the monopoly price with
           current_price as cost proxy: P* = current_price * e / (e+1)

        3. **Causal effect fallback** (insufficient elasticity data):
           Shifts price by the estimated causal effect magnitude.

        Args:
            current_price: Current product price
            row: DataFrame row with product features

        Returns:
            Recommended optimal price, clipped to [min_price, max_price]
        """
        elasticity = self.config.demand.elasticity
        min_p = self.config.pricing.min_price
        max_p = self.config.pricing.max_price

        if abs(elasticity + 1) < 0.01:
            # Unit elastic — any price gives same revenue, keep current
            return float(np.clip(current_price, min_p, max_p))

        if elasticity < -1:
            # Elastic demand: use monopoly pricing formula
            # P* = cost / (1 + 1/e). With current_price as cost proxy:
            # P* = current_price * e / (e + 1)
            # Since e < -1: e/(e+1) > 1, so this is a markup
            markup_ratio = elasticity / (elasticity + 1.0)
            optimal = current_price * markup_ratio

            # Sanity check: if the causal model suggests a large negative
            # effect, blend in the causal signal for robustness
            if self._causal_effect < 0:
                # Causal effect says price increases reduce demand.
                # Use it as a dampening factor on extreme markups.
                causal_adjustment = 1.0 / (1.0 + abs(self._causal_effect))
                # Blend: 80% formula, 20% causal-adjusted
                optimal = 0.8 * optimal + 0.2 * current_price * (1.0 + causal_adjustment)

        elif -1 < elasticity < 0:
            # Inelastic demand: higher price → higher revenue
            # Move price upward, bounded by causal effect confidence
            if self._effect_stderr > 0:
                t_stat = abs(self._causal_effect / self._effect_stderr)
                # Scale adjustment by significance — more significant = more confident
                adjustment = min(0.15, 0.05 * t_stat)
            else:
                adjustment = 0.05
            optimal = current_price * (1.0 + adjustment)

        else:
            # Fallback: shift by causal effect magnitude
            # Causal effect is dD/dP; a negative effect means we should lower price
            # to increase demand. Use a conservative adjustment.
            if abs(self._causal_effect) > 0:
                # Normalize effect relative to base demand
                base_demand = self.config.demand.base_demand
                effect_ratio = self._causal_effect / base_demand if base_demand > 0 else 0.0
                # Move price in the direction that improves revenue
                adjustment = -effect_ratio * 0.1  # conservative scaling
                optimal = current_price * (1.0 + adjustment)
            else:
                optimal = current_price

        return float(np.clip(optimal, min_p, max_p))

    def _compute_confidence(self, row: pd.Series) -> float:
        """Compute confidence score for the price prediction.

        Confidence is a composite of four factors:

        1. **Statistical significance** (t-stat of ATE):
           Higher t-stat means the effect is reliably different from zero.

        2. **Sample size**: More training data yields more stable estimates.
           Scaled with diminishing returns (log-like curve).

        3. **First-stage model quality** (R²):
           If confounders explain treatment/outcome well, the residualized
           estimates are cleaner → higher confidence.

        4. **Confounder coverage**: Bonus when row has all confounder values
           available for CATE estimation.

        Args:
            row: DataFrame row with product features

        Returns:
            Confidence score between 0 and 1
        """
        # --- Factor 1: Statistical significance ---
        if self._effect_stderr > 0:
            t_stat = abs(self._causal_effect / self._effect_stderr)
            # Map t-stat to [0, 1]: 0 at t=0, ~0.9 at t=3, saturating
            sig_score = min(1.0, t_stat / 3.0)
        else:
            sig_score = 0.5  # no stderr info — neutral

        # --- Factor 2: Sample size ---
        # Logarithmic scaling: diminishing returns after ~100 observations
        if self._n_observations > 0:
            sample_score = min(1.0, np.log1p(self._n_observations) / np.log1p(1000))
        else:
            sample_score = 0.0

        # --- Factor 3: First-stage model quality ---
        # Average R² of treatment and outcome nuisance models
        stage_score = (self._model_t_score + self._model_y_score) / 2.0
        # If no confounders, first-stage is trivially 0 — don't penalize
        if len(self.config.causal_config.confounders) == 0:
            stage_score = 0.5  # neutral

        # --- Factor 4: Confounder coverage ---
        total_confounders = len(self.config.causal_config.confounders)
        if total_confounders > 0:
            confounders_present = sum(
                1 for c in self.config.causal_config.confounders
                if c in row.index and pd.notna(row.get(c))
            )
            coverage_score = confounders_present / total_confounders
        else:
            coverage_score = 1.0  # no confounders needed

        # Weighted combination
        confidence = (
            0.30 * sig_score
            + 0.25 * sample_score
            + 0.25 * stage_score
            + 0.20 * coverage_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def effect_inference(self) -> dict:
        """Get inference results for the treatment effect.

        Returns:
            Dictionary with effect estimate, confidence interval, and p-value
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before inference")

        inference = self.model.ate_inference(X=self._training_W)

        ci_lower, ci_upper = inference.conf_int_mean()
        return {
            "effect": self._causal_effect,
            "stderr": self._effect_stderr,
            "confidence_interval": (
                float(np.asarray(ci_lower).flat[0]),
                float(np.asarray(ci_upper).flat[0]),
            ),
            "pvalue": float(inference.pvalue()),
        }

    def summary(self) -> str:
        """Get a detailed summary of the causal model.

        Includes:
        - Model configuration and variables
        - Average treatment effect with confidence interval
        - First-stage nuisance model performance (R²)
        - Effective sample size
        - Treatment effect heterogeneity (std of CATE)

        Returns:
            Multi-line string summary of the model and estimates
        """
        if not self.is_fitted:
            return "Model not fitted"

        inference = self.effect_inference()
        ci_low, ci_high = inference["confidence_interval"]

        # Significance indicator
        if inference["pvalue"] < 0.01:
            sig_label = "*** (p<0.01)"
        elif inference["pvalue"] < 0.05:
            sig_label = "** (p<0.05)"
        elif inference["pvalue"] < 0.10:
            sig_label = "* (p<0.10)"
        else:
            sig_label = "n.s."

        # Stage model quality label
        avg_stage_r2 = (self._model_t_score + self._model_y_score) / 2.0
        if avg_stage_r2 > 0.5:
            stage_quality = "Strong"
        elif avg_stage_r2 > 0.2:
            stage_quality = "Moderate"
        else:
            stage_quality = "Weak"

        return (
            f"EconML LinearDML Causal Estimator\n"
            f"{'=' * 50}\n"
            f"Treatment: {self.config.causal_config.treatment_variable}\n"
            f"Outcome: {self.config.causal_config.outcome_variable}\n"
            f"Confounders: {', '.join(self.config.causal_config.confounders) or 'none'}\n"
            f"\nAverage Treatment Effect: {self._causal_effect:.4f} {sig_label}\n"
            f"Standard Error: {self._effect_stderr:.4f}\n"
            f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n"
            f"P-value: {inference['pvalue']:.6f}\n"
            f"\n{'─' * 50}\n"
            f"Diagnostics\n"
            f"{'─' * 50}\n"
            f"Effective sample size: {self._n_observations}\n"
            f"First-stage treatment model R²: {self._model_t_score:.4f}\n"
            f"First-stage outcome model R²: {self._model_y_score:.4f}\n"
            f"Confounder strength: {stage_quality}\n"
            f"CATE heterogeneity (std): {self._cate_std:.4f}\n"
            f"Engineered features: {len(self._engineered_features)}\n"
        )
