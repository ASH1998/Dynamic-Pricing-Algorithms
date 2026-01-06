# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-01-06

### Added

- **Rich RL observation space** (7 features): normalized inventory, time, price, recent demand trend, price gap from reference, inventory depletion rate, demand volatility
- **Reward shaping** in RL environment: inventory waste penalty (scales near deadline), price smoothness regularization
- **Entropy-based confidence** for RL agents: value-function confidence for PPO/A2C, Q-value spread for DQN (replaces hand-tuned heuristic)
- **Batch prediction** in RL agent: vectorized observation matrix instead of row-by-row iteration
- **CATE-based pricing** in causal estimator: heterogeneous treatment effects via EconML `X`+`W` fit
- **Principled pricing formulas**: Lerner index markup for elastic demand, significance-scaled adjustment for inelastic
- **Polynomial feature engineering** for confounders (interaction + quadratic terms)
- **4-factor confidence scoring**: statistical significance (t-stat), sample size, first-stage R², confounder coverage
- **Causal diagnostics summary**: first-stage model R², CATE heterogeneity, significance stars, confounder strength
- **`estimate_optimal_prices()`**: grid-search revenue maximization using CATE per row
- **Data utilities module** (`data_utils.py`): `validate_input_data()`, `detect_outliers()`, `fill_missing_values()`, `generate_synthetic_data()`, `split_train_test()`
- **Evaluation metrics module** (`metrics.py`): `mae()`, `rmse()`, `mape()`, `revenue_comparison()`, `price_distribution_summary()`
- **Model persistence**: `engine.save_model(path)` / `engine.load_model(path)` for trained RL agents and causal estimators
- **`evaluate()` method**: MAE, RMSE, MAPE, revenue comparison, and price distribution against test data
- **`predict_batch()`**: process multiple DataFrames efficiently
- **Automatic preprocessing**: NaN filling, outlier detection/warning (configurable via `PreprocessingConfig`)
- **Context manager support**: `with PricingEngine(config) as engine: ...`
- **Structured logging** throughout the engine
- **`py.typed` marker** for PEP 561 type stub support
- **240 tests** across 10 test files covering RL, causal, data utilities, metrics, core engine, parsing, rollback, packaging, and DataFrame integrity

### Changed

- **Python requirement** broadened from `>=3.13` to `>=3.10` for wider adoption
- **Dependencies** converted from Poetry parentheses format to PEP 508 with broader lower bounds
- **Pydantic v2**: migrated `class Config` to `model_config = ConfigDict(extra="forbid")`
- **EconML API**: updated to `stderr_mean` and `conf_int_mean()` for v0.16.0 compatibility
- **Causal fit**: uses `X=W, W=W` for CATE estimation with proper deconfounding
- **RL observation space**: default upgraded from 2 to 7 features (legacy mode preserved)
- **`get_model_info()`**: now includes training stats, RL agent details, and causal estimator details

### Fixed

- **Legacy `YieldManager.py`**: fixed indentation error, missing colon, missing imports, added return value
- **README badge**: corrected from Python 3.9+ to match actual requirement
- **`detect_outliers` call**: fixed parameter names (`columns`/`factor`) to match actual API
- **`PreprocessingConfig` integration**: outlier detection uses correct DataFrame sum

### Removed

- Duplicate `data.py` module (consolidated on `data_utils.py`)

## [0.1.0] - 2025-09-04

### Added

- Initial release: `PricingEngine`, RL module (PPO/A2C/DQN), causal module (EconML LinearDML), hybrid mode
- YAML strategy configuration with Pydantic validation
- State management and rollback
- DataFrame integrity preservation
- Strategy file I/O (`load_strategy`, `save_strategy`)
- Custom exception hierarchy with helpful error messages
