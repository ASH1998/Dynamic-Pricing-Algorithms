"""Core module containing the main PricingEngine class."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from neuroprice.exceptions import ConfigurationError, PredictionError, TrainingError
from neuroprice.models import ModelType, StrategyConfig
from neuroprice.state import StateManager

logger = logging.getLogger(__name__)

__all__ = ["PricingEngine"]


class PricingEngine:
    """Main dynamic pricing engine combining RL and Causal Inference.

    This is the primary interface for the neuroprice library. It provides
    methods for training models, generating price recommendations,
    evaluating model performance, and persisting trained models to disk.

    Key Features:
    - Train on historical data using RL, Causal Inference, or both
    - Generate price predictions that preserve input DataFrame structure
    - Batch prediction over multiple DataFrames
    - Evaluate model with MAE, RMSE, MAPE, and revenue comparison
    - Save and load trained models to/from disk
    - Preprocessing: NaN filling, outlier detection / warnings
    - Save and restore configuration states for rollback
    - Context manager support (``with PricingEngine(config) as engine:``)

    Attributes:
        config: Strategy configuration
        is_trained: Whether the model has been trained

    Example:
        >>> from neuroprice import load_strategy, PricingEngine
        >>> config = load_strategy("strategy.yaml")
        >>> engine = PricingEngine(config)
        >>> engine.train(historical_data)
        >>> result = engine.predict(products_df)
        >>> metrics = engine.evaluate(test_df)
        >>> engine.save_model("models/my_engine")
    """

    def __init__(self, config: StrategyConfig):
        """Initialize the pricing engine with a configuration.

        Args:
            config: StrategyConfig object with all pricing parameters
        """
        self._config = config
        self._state_manager = StateManager()
        self._rl_agent: Optional[object] = None
        self._causal_estimator: Optional[object] = None
        self._is_trained = False

        # Training statistics
        self._training_stats: Dict[str, Any] = {
            "train_rows": 0,
            "training_time_seconds": 0.0,
            "warnings": [],
        }

        # Save initial state for rollback
        self._state_manager.save_state("initial", self._config)

        logger.info(
            "PricingEngine initialised: name=%s, model=%s",
            self._config.name,
            self._config.model.type.value,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PricingEngine":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager — no special cleanup needed."""
        if exc_type is not None:
            logger.error(
                "PricingEngine exiting context with exception: %s: %s",
                exc_type.__name__,
                exc_val,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> StrategyConfig:
        """Get the current configuration."""
        return self._config

    @property
    def is_trained(self) -> bool:
        """Check if the engine has been trained."""
        return self._is_trained

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        historical_data: pd.DataFrame,
        verbose: int = 0,
    ) -> "PricingEngine":
        """Train the pricing model on historical data.

        Depending on the model type configured:
        - 'rl': Trains a reinforcement learning agent
        - 'causal': Fits a causal inference model
        - 'hybrid': Trains both and combines predictions

        If preprocessing is enabled in the configuration, input data
        is validated and cleaned before training.

        Args:
            historical_data: DataFrame with historical sales data
            verbose: Verbosity level (0=silent, 1=progress, 2=debug)

        Returns:
            Self for method chaining

        Raises:
            TrainingError: If training fails
            ValueError: If required columns are missing
        """
        import time

        # Preprocessing
        data, warnings_list = self._preprocess(historical_data)
        self._training_stats["warnings"] = warnings_list

        # Validate DataFrame has required columns
        self._validate_dataframe(data)

        model_type = self._config.model.type
        logger.info("Starting training: model_type=%s, rows=%d", model_type.value, len(data))

        t0 = time.monotonic()

        try:
            # Train RL agent if needed
            if model_type in (ModelType.RL, ModelType.HYBRID):
                from neuroprice.rl.agents import RLAgent

                self._rl_agent = RLAgent(self._config)
                self._rl_agent.train(data, verbose=verbose)

                logger.info("RL agent training complete")
                if verbose > 0:
                    print("RL agent training complete")

            # Fit causal model if needed
            if model_type in (ModelType.CAUSAL, ModelType.HYBRID):
                from neuroprice.causal.estimator import CausalEstimator

                self._causal_estimator = CausalEstimator(self._config)
                self._causal_estimator.fit(data, verbose=bool(verbose))

                logger.info("Causal model fitting complete")
                if verbose > 0:
                    print("Causal model fitting complete")

            self._is_trained = True
            elapsed = time.monotonic() - t0

            # Record training stats
            self._training_stats.update({
                "train_rows": len(data),
                "training_time_seconds": round(elapsed, 3),
            })

            # Save trained state
            self._state_manager.save_state("trained", self._config)

            logger.info("Training complete in %.2fs (%d rows)", elapsed, len(data))
            return self

        except ImportError as e:
            raise TrainingError(
                message=f"Missing dependency: {e}",
                model_type=model_type.value,
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price recommendations for input DataFrame.

        IMPORTANT: This method preserves all original DataFrame columns.
        Only new columns are appended (configured price and confidence columns).
        The input DataFrame is never modified.

        If preprocessing is enabled, the input data is validated and cleaned
        before prediction.

        Args:
            df: Input DataFrame with required feature columns

        Returns:
            New DataFrame with original columns plus:
            - price_column: Recommended price
            - confidence_column: Prediction confidence (0-1)

        Raises:
            PredictionError: If prediction fails
            ValueError: If model is not trained or columns are missing
        """
        if not self._is_trained:
            raise ValueError(
                "Engine must be trained before making predictions. "
                "Call engine.train(historical_data) first."
            )

        # Preprocessing
        cleaned, _ = self._preprocess(df)

        # Validate input DataFrame
        self._validate_dataframe(cleaned)

        # Create a copy to avoid modifying input
        result = cleaned.copy()

        try:
            # Get predictions based on model type
            prices, confidences = self._generate_predictions(cleaned)

            # Append new columns (never modify existing)
            result[self._config.output.price_column] = prices
            result[self._config.output.confidence_column] = confidences

            logger.debug("Predicted %d rows", len(result))
            return result

        except Exception as e:
            raise PredictionError(str(e))

    def predict_batch(
        self,
        dfs: List[pd.DataFrame],
    ) -> List[pd.DataFrame]:
        """Generate price recommendations for multiple DataFrames.

        Convenience method that iterates over a list of DataFrames,
        calling :meth:`predict` on each. Shared preprocessing and
        validation are applied per-frame.

        Args:
            dfs: List of DataFrames with required feature columns.

        Returns:
            List of result DataFrames (one per input).

        Raises:
            PredictionError: If any prediction fails.
        """
        if not self._is_trained:
            raise ValueError(
                "Engine must be trained before making predictions. "
                "Call engine.train(historical_data) first."
            )

        results: List[pd.DataFrame] = []
        total_rows = 0
        for i, df in enumerate(dfs):
            logger.debug("predict_batch: processing DataFrame %d/%d", i + 1, len(dfs))
            results.append(self.predict(df))
            total_rows += len(df)

        logger.info("predict_batch complete: %d DataFrames, %d total rows", len(dfs), total_rows)
        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_df: pd.DataFrame,
        actual_demand_column: str = "demand",
    ) -> Dict[str, Any]:
        """Evaluate model performance on test data with actual outcomes.

        Generates predictions and computes regression metrics against the
        actual demand column. Also compares revenue between the model's
        recommended prices and the current prices in the data.

        Args:
            test_df: DataFrame with actual outcomes. Must contain the
                ``actual_demand_column`` and other required features.
            actual_demand_column: Name of the column with actual demand
                values used as ground truth.

        Returns:
            Dictionary with keys:
            - ``mae``: Mean absolute error of price predictions vs demand.
            - ``rmse``: Root mean squared error.
            - ``mape``: Mean absolute percentage error (fraction).
            - ``n_samples``: Number of evaluated samples.
            - ``revenue_comparison``: Revenue comparison dict from
              :func:`neuroprice.metrics.revenue_comparison`.
            - ``price_summary``: Price distribution summary of
              recommended prices.

        Raises:
            ValueError: If the actual demand column is missing.
        """
        from neuroprice.metrics import (
            mae as calc_mae,
            mape as calc_mape,
            price_distribution_summary,
            revenue_comparison,
            rmse as calc_rmse,
        )

        if actual_demand_column not in test_df.columns:
            raise ValueError(
                f"Actual demand column '{actual_demand_column}' not found in test data"
            )

        # Generate predictions
        result = self.predict(test_df)
        price_col = self._config.output.price_column

        recommended_prices = result[price_col].values
        actual_demand = result[actual_demand_column].values

        # Use current_price as baseline if available, else use recommended prices
        if "current_price" in result.columns:
            baseline_prices = result["current_price"].values
        else:
            baseline_prices = recommended_prices  # no baseline available

        # Compute metrics — treating recommended prices as "predicted" and
        # actual demand as "target" for a demand-matching signal
        metrics: Dict[str, Any] = {
            "mae": calc_mae(actual_demand, recommended_prices),
            "rmse": calc_rmse(actual_demand, recommended_prices),
            "mape": calc_mape(actual_demand, recommended_prices),
            "n_samples": len(result),
        }

        # Revenue comparison: baseline pricing vs recommended pricing
        # Use actual demand as the demand signal for both scenarios
        metrics["revenue_comparison"] = revenue_comparison(
            actual_prices=baseline_prices,
            recommended_prices=recommended_prices,
            demands=actual_demand,
        )

        # Price distribution summary
        metrics["price_summary"] = price_distribution_summary(recommended_prices)

        logger.info(
            "Evaluation complete: MAE=%.4f, RMSE=%.4f, MAPE=%.4f (n=%d)",
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
            metrics["n_samples"],
        )

        return metrics

    # ------------------------------------------------------------------
    # Prediction internals
    # ------------------------------------------------------------------

    def _generate_predictions(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions based on configured model type.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (prices, confidences) arrays
        """
        model_type = self._config.model.type

        if model_type == ModelType.RL:
            return self._rl_agent.predict(df)  # type: ignore[union-attr]

        elif model_type == ModelType.CAUSAL:
            return self._causal_estimator.estimate_prices(df)  # type: ignore[union-attr]

        else:  # HYBRID
            # Get predictions from both models
            rl_prices, rl_conf = self._rl_agent.predict(df)  # type: ignore[union-attr]
            causal_prices, causal_conf = self._causal_estimator.estimate_prices(df)  # type: ignore[union-attr]

            # Weighted combination (equal weights by default)
            # Weight by confidence scores
            total_conf = rl_conf + causal_conf
            rl_weight = rl_conf / np.maximum(total_conf, 1e-6)
            causal_weight = causal_conf / np.maximum(total_conf, 1e-6)

            prices = rl_weight * rl_prices + causal_weight * causal_prices
            confidences = (rl_conf + causal_conf) / 2

            # Clamp to price bounds
            prices = np.clip(
                prices,
                self._config.pricing.min_price,
                self._config.pricing.max_price,
            )

            return prices, confidences

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean input data based on preprocessing config.

        When preprocessing is enabled:
        1. NaN values in numeric columns are filled (default: median).
        2. Outliers in required feature columns trigger warnings.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (cleaned_df, warnings_list).
        """
        if not self._config.preprocessing.enable_cleaning:
            return df, []

        from neuroprice.data_utils import detect_outliers, fill_missing_values

        warnings_list: List[str] = []
        cleaned = df.copy()

        # Fill NaN values in numeric columns
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        nan_cols = [c for c in numeric_cols if cleaned[c].isna().any()]
        if nan_cols:
            strategy = self._config.preprocessing.fill_strategy
            logger.warning("Filling NaN values in columns: %s (strategy=%s)", nan_cols, strategy)
            warnings_list.append(f"Filled NaN values in columns: {nan_cols}")
            cleaned = fill_missing_values(cleaned, strategy=strategy)

        # Outlier detection on required feature columns
        for col in self._config.features.required:
            if col not in cleaned.columns or not np.issubdtype(cleaned[col].dtype, np.number):
                continue
            try:
                mask = detect_outliers(
                    cleaned,
                    columns=[col],
                    method=self._config.preprocessing.outlier_method,
                    factor=self._config.preprocessing.outlier_threshold,
                )
                n_outliers = int(mask.sum().sum())
                if n_outliers > 0:
                    pct = n_outliers / len(cleaned) * 100
                    warnings_list.append(
                        f"Column '{col}': {n_outliers} outliers detected ({pct:.1f}%)"
                    )
                    logger.warning(
                        "Outliers in column '%s': %d (%.1f%%)", col, n_outliers, pct
                    )
            except ValueError:
                pass  # column not numeric or method unsupported

        return cleaned, warnings_list

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = set(self._config.features.required)
        actual_cols = set(df.columns)
        missing = required_cols - actual_cols

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Required columns are: {self._config.features.required}"
            )

    # ------------------------------------------------------------------
    # Model save / load to disk
    # ------------------------------------------------------------------

    def save_model(self, path: Union[str, Path]) -> None:
        """Persist the trained engine (RL agent and/or causal estimator) to disk.

        Creates a directory at ``path`` containing:
        - ``config.json``: Serialised strategy configuration.
        - ``rl_agent/``: RL agent model (if present), via stable-baselines3.
        - ``causal_estimator/``: Causal estimator (if present), via joblib.
        - ``meta.json``: Training stats and metadata.

        Args:
            path: Directory path where the model artefacts are saved.

        Raises:
            ValueError: If the engine has not been trained.
        """
        if not self._is_trained:
            raise ValueError("Cannot save an untrained engine. Call train() first.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save config
        config_path = save_dir / "config.json"
        config_path.write_text(self._config.model_dump_json(indent=2))
        logger.info("Config saved to %s", config_path)

        # 2. Save RL agent
        if self._rl_agent is not None:
            rl_dir = save_dir / "rl_agent"
            rl_dir.mkdir(exist_ok=True)
            try:
                self._rl_agent.save(str(rl_dir / "model"))  # type: ignore[union-attr]
                logger.info("RL agent saved to %s", rl_dir)
            except Exception as e:
                logger.error("Failed to save RL agent: %s", e)

        # 3. Save causal estimator
        if self._causal_estimator is not None:
            causal_dir = save_dir / "causal_estimator"
            causal_dir.mkdir(exist_ok=True)
            try:
                import joblib

                joblib.dump(self._causal_estimator, str(causal_dir / "estimator.joblib"))
                logger.info("Causal estimator saved to %s", causal_dir)
            except ImportError:
                # joblib not available — fall back to pickle
                import pickle

                with open(causal_dir / "estimator.pkl", "wb") as f:
                    pickle.dump(self._causal_estimator, f)
                logger.info("Causal estimator saved (pickle) to %s", causal_dir)

        # 4. Save metadata
        meta = {
            "model_type": self._config.model.type.value,
            "is_trained": self._is_trained,
            "has_rl_agent": self._rl_agent is not None,
            "has_causal_estimator": self._causal_estimator is not None,
            "training_stats": self._training_stats,
        }
        meta_path = save_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Model saved to %s", save_dir)

    def load_model(self, path: Union[str, Path]) -> "PricingEngine":
        """Load a previously saved engine from disk.

        Restores the configuration, RL agent, and/or causal estimator
        that were saved with :meth:`save_model`.

        Args:
            path: Directory path where the model artefacts were saved.

        Returns:
            Self for method chaining.

        Raises:
            FileNotFoundError: If the save directory or config doesn't exist.
        """
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")

        # 1. Load config
        config_path = load_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self._config = StrategyConfig(**json.loads(config_path.read_text()))
        logger.info("Config loaded from %s", config_path)

        # 2. Load RL agent
        rl_model_file = load_dir / "rl_agent" / "model.zip"
        if rl_model_file.exists():
            try:
                from neuroprice.rl.agents import RLAgent

                self._rl_agent = RLAgent(self._config)
                self._rl_agent.load(str(load_dir / "rl_agent" / "model"))  # type: ignore[union-attr]
                logger.info("RL agent loaded from %s", rl_model_file)
            except Exception as e:
                logger.warning("Could not load RL agent: %s", e)
                self._rl_agent = None

        # 3. Load causal estimator
        causal_joblib = load_dir / "causal_estimator" / "estimator.joblib"
        causal_pkl = load_dir / "causal_estimator" / "estimator.pkl"
        if causal_joblib.exists():
            try:
                import joblib

                self._causal_estimator = joblib.load(str(causal_joblib))
                logger.info("Causal estimator loaded from %s", causal_joblib)
            except Exception as e:
                logger.warning("Could not load causal estimator: %s", e)
                self._causal_estimator = None
        elif causal_pkl.exists():
            try:
                import pickle

                with open(causal_pkl, "rb") as f:
                    self._causal_estimator = pickle.load(f)
                logger.info("Causal estimator loaded (pickle) from %s", causal_pkl)
            except Exception as e:
                logger.warning("Could not load causal estimator: %s", e)
                self._causal_estimator = None

        # 4. Load metadata
        meta_path = load_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._training_stats = meta.get("training_stats", self._training_stats)

        self._is_trained = True
        logger.info("Model loaded from %s", load_dir)
        return self

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def save_state(self, name: str) -> None:
        """Save current configuration state for later rollback.

        Args:
            name: Unique identifier for this state
        """
        self._state_manager.save_state(name, self._config)

    def rollback(self, state_name: str = "initial") -> "PricingEngine":
        """Rollback to a previously saved configuration state.

        Args:
            state_name: Name of the state to restore (default: "initial")

        Returns:
            Self for method chaining

        Raises:
            StateRollbackError: If the state doesn't exist
        """
        self._config = self._state_manager.restore_state(state_name)
        return self

    def list_states(self) -> List[str]:
        """List all saved state names.

        Returns:
            List of state names
        """
        return self._state_manager.list_states()

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Get information about the current model configuration and training.

        Returns:
            Dictionary with model information including training stats
            when available.
        """
        info: Dict[str, Any] = {
            "name": self._config.name,
            "model_type": self._config.model.type.value,
            "algorithm": self._config.model.algorithm.value,
            "is_trained": self._is_trained,
            "price_range": (
                self._config.pricing.min_price,
                self._config.pricing.max_price,
            ),
            "features_required": self._config.features.required,
            "output_columns": (
                self._config.output.price_column,
                self._config.output.confidence_column,
            ),
            "preprocessing_enabled": self._config.preprocessing.enable_cleaning,
        }

        # Include training stats if available
        if self._is_trained:
            info["training_stats"] = self._training_stats

            # RL agent info
            if self._rl_agent is not None:
                info["rl_agent"] = {
                    "rich_observation": getattr(self._rl_agent, "rich_observation", None),
                    "feature_names": getattr(self._rl_agent, "feature_names", None),
                }

            # Causal estimator info
            if self._causal_estimator is not None:
                info["causal_estimator"] = {
                    "is_fitted": getattr(self._causal_estimator, "is_fitted", False),
                    "causal_effect": getattr(self._causal_estimator, "_causal_effect", None),
                    "n_observations": getattr(self._causal_estimator, "_n_observations", None),
                }

        return info

    # ------------------------------------------------------------------
    # Config update
    # ------------------------------------------------------------------

    def update_config(self, **kwargs) -> "PricingEngine":
        """Update configuration parameters.

        This creates a new configuration with updated values.
        Previous configuration is preserved in state manager.

        Args:
            **kwargs: Configuration fields to update

        Returns:
            Self for method chaining
        """
        # Save current state before update
        self.save_state("before_update")

        # Create updated config dict
        config_dict = self._config.model_dump()

        # Update nested fields
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "pricing.min_price"
                parts = key.split(".")
                target = config_dict
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            else:
                config_dict[key] = value

        # Create new config
        self._config = StrategyConfig(**config_dict)

        # Model needs retraining after config change
        self._is_trained = False

        logger.info("Configuration updated — model requires retraining")
        return self

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation of the engine."""
        return (
            f"PricingEngine(name={self._config.name!r}, "
            f"model={self._config.model.type.value}, "
            f"trained={self._is_trained})"
        )
