"""Pydantic models for configuration validation."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelType(str, Enum):
    """Supported model types for pricing engine."""

    RL = "rl"
    CAUSAL = "causal"
    HYBRID = "hybrid"


class RLAlgorithm(str, Enum):
    """Supported RL algorithms from stable-baselines3."""

    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"


class PricingConfig(BaseModel):
    """Configuration for price boundaries and steps."""

    min_price: float = Field(gt=0, description="Minimum allowed price")
    max_price: float = Field(gt=0, description="Maximum allowed price")
    price_step: float = Field(default=0.01, gt=0, description="Price increment step")
    currency: str = Field(default="USD", description="Currency code")

    @model_validator(mode="after")
    def validate_price_range(self) -> "PricingConfig":
        """Ensure max_price is greater than min_price."""
        if self.max_price <= self.min_price:
            raise ValueError(
                f"max_price ({self.max_price}) must be greater than "
                f"min_price ({self.min_price})"
            )
        return self


class SeasonalityConfig(BaseModel):
    """Configuration for demand seasonality."""

    enabled: bool = Field(default=False, description="Enable seasonality adjustment")
    period: int = Field(default=7, ge=1, description="Seasonality period in days")


class PreprocessingConfig(BaseModel):
    """Configuration for input data preprocessing."""

    fill_strategy: str = Field(
        default="median",
        description="Strategy for filling missing values: 'median', 'mean', or 'zero'",
    )
    outlier_method: str = Field(
        default="iqr",
        description="Outlier detection method: 'iqr'",
    )
    outlier_threshold: float = Field(
        default=1.5,
        gt=0,
        description="IQR multiplier for outlier fence computation",
    )
    enable_cleaning: bool = Field(
        default=True,
        description="Whether to automatically clean input data during predict/train",
    )


class DemandConfig(BaseModel):
    """Configuration for demand modeling."""

    elasticity: float = Field(
        default=-1.0, le=0, description="Price elasticity of demand (should be negative)"
    )
    base_demand: float = Field(gt=0, description="Base demand level")
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)


class InventoryConfig(BaseModel):
    """Configuration for inventory constraints."""

    initial_stock: int = Field(ge=0, description="Initial inventory level")
    salvage_value: float = Field(
        default=0.0, ge=0, description="Value per unsold item at deadline"
    )
    deadline_days: Optional[int] = Field(
        default=None, ge=1, description="Selling period deadline in days"
    )


class RLConfig(BaseModel):
    """Configuration for reinforcement learning model."""

    learning_rate: float = Field(
        default=0.0003, gt=0, le=1, description="Learning rate for optimizer"
    )
    gamma: float = Field(
        default=0.99, ge=0, le=1, description="Discount factor for future rewards"
    )
    n_steps: int = Field(
        default=2048, ge=1, description="Number of steps per update"
    )
    batch_size: int = Field(default=64, ge=1, description="Minibatch size")
    training_episodes: int = Field(
        default=10000, ge=1, description="Total training timesteps"
    )


class CausalConfig(BaseModel):
    """Configuration for causal inference model."""

    treatment_variable: str = Field(
        default="price", description="Variable representing the treatment (price)"
    )
    outcome_variable: str = Field(
        default="demand", description="Variable representing the outcome (demand)"
    )
    confounders: List[str] = Field(
        default_factory=list, description="List of confounding variables"
    )
    estimation_method: str = Field(
        default="backdoor.linear_regression",
        description="DoWhy estimation method to use",
    )


class FeaturesConfig(BaseModel):
    """Configuration for input DataFrame features."""

    required: List[str] = Field(description="Required column names in input DataFrame")
    optional: List[str] = Field(
        default_factory=list, description="Optional column names"
    )

    @field_validator("required")
    @classmethod
    def validate_required_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one required feature is specified."""
        if not v:
            raise ValueError("At least one required feature must be specified")
        return v


class OutputConfig(BaseModel):
    """Configuration for output DataFrame columns."""

    price_column: str = Field(
        default="recommended_price",
        description="Column name for recommended price output",
    )
    confidence_column: str = Field(
        default="confidence_score",
        description="Column name for prediction confidence output",
    )


class ModelConfig(BaseModel):
    """Configuration for model selection."""

    type: ModelType = Field(default=ModelType.RL, description="Model type to use")
    algorithm: RLAlgorithm = Field(
        default=RLAlgorithm.PPO, description="RL algorithm (when type is 'rl' or 'hybrid')"
    )


class StrategyConfig(BaseModel):
    """Root configuration model for pricing strategy.

    This is the main configuration object that holds all settings
    for the dynamic pricing engine.
    """

    version: str = Field(default="1.0", description="Configuration schema version")
    name: str = Field(description="Name of the pricing strategy")
    model: ModelConfig = Field(default_factory=ModelConfig)
    pricing: PricingConfig
    demand: DemandConfig
    inventory: InventoryConfig
    rl_config: RLConfig = Field(default_factory=RLConfig)
    causal_config: CausalConfig = Field(default_factory=CausalConfig)
    features: FeaturesConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    model_config = ConfigDict(extra="forbid")
