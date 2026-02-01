# neuroprice

A Dynamic Pricing Engine using Reinforcement Learning and Causal Inference.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation & Quick Start

### Installation

```bash
pip install neuroprice
```

For development with testing tools:

```bash
pip install neuroprice[dev]
```

### Quick Start

#### 1. Create a Strategy File

Create a YAML file defining your pricing strategy:

```yaml
# my_strategy.yaml
version: "1.0"
name: "retail_pricing"

model:
  type: "rl"
  algorithm: "PPO"

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

features:
  required:
    - "product_id"
    - "current_price"
    - "inventory_level"

output:
  price_column: "recommended_price"
  confidence_column: "confidence"
```

See `templates/strategy_template.yaml` for a complete example with all options.

#### 2. Load Strategy and Create Engine

```python
import pandas as pd
from neuroprice import load_strategy, PricingEngine

# Load your strategy configuration
config = load_strategy("my_strategy.yaml")

# Create the pricing engine
engine = PricingEngine(config)

# Train on historical data
historical_data = pd.read_csv("sales_history.csv")
engine.train(historical_data)
```

#### 3. Generate Price Recommendations

```python
# Your current product data
products = pd.DataFrame({
    "product_id": ["SKU001", "SKU002", "SKU003"],
    "current_price": [49.99, 79.99, 29.99],
    "inventory_level": [100, 50, 200],
})

# Get recommendations (original columns preserved, new columns appended)
recommendations = engine.predict(products)
print(recommendations[["product_id", "recommended_price", "confidence"]])
```

#### 4. State Management & Rollback

```python
# Save current state
engine.save_state("after_initial_training")

# Later, rollback if needed
engine.rollback("after_initial_training")

# List available states
print(engine.list_states())
```

## Features

- **Multiple Model Types**: Choose from RL, Causal Inference, or Hybrid approaches
- **Pandas-First API**: Input and output are DataFrames with preserved column integrity
- **Configurable via YAML**: Define pricing strategies in human-readable configuration files
- **State Management**: Save and restore configuration states for easy rollback
- **Graceful Error Handling**: Helpful error messages guide you when things go wrong

## Model Types

| Type | Description | Best For |
|------|-------------|----------|
| `rl` | Reinforcement Learning (PPO, A2C, DQN) | Learning optimal pricing through interaction |
| `causal` | Causal Inference (DoWhy) | Understanding true price-demand relationships |
| `hybrid` | Combination of RL + Causal | Balanced approach with exploration and causal understanding |

## Error Handling

```python
from neuroprice import load_strategy
from neuroprice.exceptions import StrategyFileNotFoundError

try:
    config = load_strategy("nonexistent.yaml")
except StrategyFileNotFoundError as e:
    print(f"Error: {e.message}")
    # Error message includes guidance on creating a strategy file
```

## API Reference

### Main Functions

- `load_strategy(file_path: str) -> StrategyConfig` - Load pricing strategy from YAML
- `save_strategy(config, file_path: str)` - Save strategy to YAML file

### PricingEngine Class

```python
engine = PricingEngine(config: StrategyConfig)
```

**Methods:**
- `train(historical_data: pd.DataFrame) -> PricingEngine` - Train the model
- `predict(df: pd.DataFrame) -> pd.DataFrame` - Generate price predictions
- `save_state(name: str)` - Save current configuration state
- `rollback(state_name: str) -> PricingEngine` - Restore saved state
- `list_states() -> list[str]` - List available states
- `get_model_info() -> dict` - Get model configuration info

## Dependencies

- pandas >= 2.0
- numpy >= 1.24
- stable-baselines3 >= 2.0
- gymnasium >= 0.29
- econml >= 0.15
- scikit-learn >= 1.6
- pydantic >= 2.0
- PyYAML >= 6.0

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Project Structure

```
neuroprice/
├── src/neuroprice/
│   ├── __init__.py      # Main API
│   ├── core.py          # PricingEngine class
│   ├── io.py            # Strategy file loading
│   ├── models.py        # Pydantic configuration models
│   ├── state.py         # State management
│   ├── exceptions.py    # Custom exceptions
│   ├── rl/              # Reinforcement Learning module
│   └── causal/          # Causal Inference module
├── templates/           # Strategy file templates
└── tests/               # Test suite
```

## Background: Dynamic Pricing Theory

This library is built on dynamic pricing models characterized by:

1. **Unlimited potential customers**: Population size is not a model parameter
2. **Single item type**: Focused optimization for one product category
3. **Monopoly situation**: No direct competition modeling
4. **Myopic customers**: Customers buy when price ≤ willingness to pay

The mathematical foundations include:
- Deterministic models with time-dated items
- Stochastic models with Poisson arrival processes
- Salvage value optimization

For more details, see the academic literature on revenue management and the included `dynamic_pricing_algo.txt` reference material.

## License

MIT License - see LICENSE file for details.
