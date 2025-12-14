<pre>
 ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██████╗ ██████╗ ██╗ ██████╗███████╗
 ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██║██╔════╝██╔════╝
 ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██████╔╝██████╔╝██║██║     █████╗  
 ██║╚██╗██║██╔══╝  ██║   ██║██╔═══╝ ██║   ██║██╔══██╗██╔══██╗██║██║     ██╔══╝  
 ██║ ╚████║███████╗╚██████╔╝██║     ╚██████╔╝██║  ██║██║  ██║██║╚██████╗███████╗
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝
            ┌─────────────────────────────────────────────────┐
            │   Dynamic Pricing via RL + Causal Inference     │
            └─────────────────────────────────────────────────┘
    ╭──────────╮   ╭──────────╮   ╭──────────╮   ╭──────────╮
    │  Demand  │──▶│  Causal  │──▶│   RL     │──▶│  Optimal │
    │  Signal  │   │  Engine  │   │  Agent   │   │  Price   │
    ╰──────────╯   ╰──────────╯   ╰──────────╯   ╰──────────╯
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
    │Confounds│   │  CATE    │   │ PPO/A2C │   │ Revenue  │
    │  Day/WoW│   │  Effects │   │  /DQN   │   │  Max     │
    │Promos   │   │  Hetero. │   │ Entropy │   │  Guard   │
    └─────────┘   └──────────┘   └─────────┘   └──────────┘
</pre>

# neuroprice

A production-ready Dynamic Pricing Engine using Reinforcement Learning and Causal Inference.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-240%20passed-brightgreen.svg)](#development)

## What is this?

**neuroprice** is a Python library for building intelligent dynamic pricing systems. It combines:

- **Reinforcement Learning** (PPO, A2C, DQN) — learns optimal pricing through simulated interaction
- **Causal Inference** (EconML Double ML) — understands *true* price-demand relationships, controlling for confounders
- **Hybrid mode** — blends both approaches for robust pricing

Use it for e-commerce, SaaS, travel, hospitality, or any domain where price optimization matters.

## Installation

```bash
pip install neuroprice
```

For development:

```bash
pip install neuroprice[dev]
```

## Quick Start

### 1. Define a Strategy

```yaml
# my_strategy.yaml
version: "1.0"
name: "retail_pricing"

model:
  type: "rl"          # "rl", "causal", or "hybrid"
  algorithm: "PPO"    # "PPO", "A2C", or "DQN"

pricing:
  min_price: 10.0
  max_price: 200.0
  price_step: 1.0

demand:
  elasticity: -1.5
  base_demand: 1000

inventory:
  initial_stock: 500
  salvage_value: 5.0
  deadline_days: 30

features:
  required:
    - "product_id"
    - "current_price"
    - "inventory_level"

output:
  price_column: "recommended_price"
  confidence_column: "confidence"
```

### 2. Train and Predict

```python
import pandas as pd
from neuroprice import load_strategy, PricingEngine

# Load strategy
config = load_strategy("my_strategy.yaml")

# Create and train engine
engine = PricingEngine(config)
historical_data = pd.read_csv("sales_history.csv")
engine.train(historical_data)

# Generate recommendations (original columns preserved, new columns appended)
products = pd.DataFrame({
    "product_id": ["SKU001", "SKU002", "SKU003"],
    "current_price": [49.99, 79.99, 29.99],
    "inventory_level": [100, 50, 200],
})

recommendations = engine.predict(products)
print(recommendations[["product_id", "recommended_price", "confidence"]])
```

### 3. Evaluate Model Quality

```python
from neuroprice import mae, rmse, mape, revenue_comparison

# Evaluate predictions against actual outcomes
metrics = engine.evaluate(test_data, actual_demand_column="demand")
print(metrics)
# {'mae': 4.2, 'rmse': 6.1, 'mape': 0.08, 'revenue_comparison': {...}}

# Or use standalone metrics
rev = revenue_comparison(actual_prices, recommended_prices, demands)
print(f"Revenue improvement: {rev['improvement_pct']:.1f}%")
```

### 4. Save/Load and Rollback

```python
# Save trained model to disk
engine.save_model("my_model")

# Load later
engine2 = PricingEngine(config)
engine2.load_model("my_model")

# State management for config rollback
engine.save_state("before_experiment")
engine.update_config(pricing__min_price=15.0)
engine.rollback("before_experiment")
```

## Features

| Feature | Description |
|---------|-------------|
| **3 Model Types** | RL (PPO/A2C/DQN), Causal (EconML DML), Hybrid |
| **Rich RL Observations** | 7-feature state: inventory, time, price, demand trend, volatility, price gap, depletion rate |
| **CATE Pricing** | Heterogeneous treatment effects for per-product optimal pricing |
| **Reward Shaping** | Inventory waste penalties, price smoothness regularization |
| **Data Validation** | Automatic NaN filling, outlier detection, column validation |
| **Evaluation Metrics** | MAE, RMSE, MAPE, revenue comparison, price distribution analysis |
| **Model Persistence** | Save/load trained models to disk |
| **State Rollback** | Checkpoint and restore configurations |
| **Synthetic Data** | Generate test data with known causal structure |
| **Pandas-First** | Input/output are DataFrames, columns always preserved |
| **Type-Safe Config** | Pydantic v2 validation with YAML strategy files |
| **Logging** | Structured Python logging throughout |

## Model Types

| Type | Description | Best For |
|------|-------------|----------|
| `rl` | Reinforcement Learning | Learning optimal pricing through simulated interaction |
| `causal` | Causal Inference (EconML) | Understanding true price-demand with confounders |
| `hybrid` | RL + Causal blended | Balanced exploration with causal grounding |

### RL Module

The RL module wraps stable-baselines3 with a custom Gymnasium environment:

- **Rich observation space** (7 features): normalized inventory, time, price, recent demand trend, price gap from reference, inventory depletion rate, demand volatility
- **Log-linear demand model**: `D = D_base × (P/P_ref)^ε` with Poisson stochasticity
- **Reward shaping**: inventory waste penalty (scales near deadline), price smoothness penalty
- **Confidence**: entropy-based (PPO/A2C) or Q-value spread (DQN), not heuristic

### Causal Module

The causal module uses EconML's Double Machine Learning:

- **Confounder-controlled**: estimates true price→demand effect, not just correlation
- **CATE estimation**: heterogeneous treatment effects across product segments
- **Feature engineering**: polynomial/interaction terms from confounders
- **Diagnostics**: first-stage R², treatment effect significance, heterogeneity stats

## API Reference

### Main Functions

```python
load_strategy(file_path) -> StrategyConfig
save_strategy(config, file_path)
validate_strategy_dict(config_dict) -> StrategyConfig
generate_synthetic_data(n_samples, ...) -> DataFrame
```

### PricingEngine

```python
engine = PricingEngine(config: StrategyConfig)

# Training
engine.train(historical_data, verbose=0) -> PricingEngine

# Prediction
engine.predict(df) -> DataFrame              # Single batch
engine.predict_batch(dfs) -> list[DataFrame]  # Multiple batches

# Evaluation
engine.evaluate(test_df, actual_demand_column) -> dict

# Model persistence
engine.save_model(path)
engine.load_model(path)

# State management
engine.save_state(name)
engine.rollback(state_name) -> PricingEngine
engine.list_states() -> list[str]

# Configuration
engine.update_config(**kwargs) -> PricingEngine
engine.get_model_info() -> dict
```

### Metrics

```python
from neuroprice import mae, rmse, mape, revenue_comparison, price_distribution_summary

mae(y_true, y_pred) -> float
rmse(y_true, y_pred) -> float
mape(y_true, y_pred) -> float
revenue_comparison(actual, recommended, demands) -> dict
price_distribution_summary(prices) -> dict
```

### Data Utilities

```python
from neuroprice import (
    validate_input_data,
    detect_outliers,
    fill_missing_values,
    generate_synthetic_data,
    split_train_test,
)
```

## Project Structure

```
neuroprice/
├── src/neuroprice/
│   ├── __init__.py          # Public API
│   ├── core.py              # PricingEngine class
│   ├── models.py            # Pydantic config models
│   ├── io.py                # YAML strategy I/O
│   ├── state.py             # State management
│   ├── exceptions.py        # Custom exceptions
│   ├── metrics.py           # Evaluation metrics
│   ├── data_utils.py        # Data validation & synthetic data
│   ├── rl/
│   │   ├── environment.py   # Gymnasium pricing environment
│   │   └── agents.py        # SB3 RL agent wrappers
│   └── causal/
│       └── estimator.py     # EconML causal estimator
├── templates/
│   └── strategy_template.yaml
├── tests/                   # 240 tests
└── pyproject.toml
```

## Dependencies

- Python >= 3.10
- pandas >= 2.0, numpy >= 1.24
- stable-baselines3 >= 2.0, gymnasium >= 0.29
- econml >= 0.15, scikit-learn >= 1.6
- pydantic >= 2.0, PyYAML >= 6.0

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=neuroprice --cov-report=term-missing

# Lint
ruff check src/ tests/
```

## License

MIT License - see LICENSE file for details.
