"""Data utility functions for the neuroprice library.

Provides helpers for data validation, cleaning, synthetic data generation,
and train/test splitting for pricing model development.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def validate_input_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_outliers: bool = False,
    outlier_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Validate and optionally clean input data.

    Checks for required columns, NaN values, and optionally detects
    and flags outliers.

    Args:
        df: Input DataFrame to validate.
        required_columns: Columns that must be present.
        check_outliers: If True, detect and flag outliers.
        outlier_columns: Columns to check for outliers (defaults to all numeric).

    Returns:
        Cleaned DataFrame (copy).

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    if check_outliers:
        cols = outlier_columns or result.select_dtypes(include=[np.number]).columns.tolist()
        outlier_mask = detect_outliers(result, columns=cols)
        if outlier_mask.any().any():
            result["_has_outlier"] = outlier_mask.any(axis=1)

    return result


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers in numeric columns.

    Uses the IQR method by default: a value is an outlier if it falls
    below Q1 - factor*IQR or above Q3 + factor*IQR.

    Args:
        df: Input DataFrame.
        columns: Columns to check (defaults to all numeric).
        method: Detection method ('iqr' or 'zscore').
        factor: IQR multiplier or z-score threshold.

    Returns:
        Boolean DataFrame indicating outlier positions.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            result[col] = (df[col] < lower) | (df[col] > upper)
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std > 0:
                z = (df[col] - mean).abs() / std
                result[col] = z > factor

    return result


def fill_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill missing values in a DataFrame.

    Args:
        df: Input DataFrame.
        strategy: Fill strategy ('median', 'mean', 'zero', 'ffill').
        columns: Columns to fill (defaults to all numeric).

    Returns:
        DataFrame with filled values (copy).
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        if strategy == "median":
            fill_val = result[col].median()
            result[col] = result[col].fillna(fill_val)
        elif strategy == "mean":
            fill_val = result[col].mean()
            result[col] = result[col].fillna(fill_val)
        elif strategy == "zero":
            result[col] = result[col].fillna(0)
        elif strategy == "ffill":
            result[col] = result[col].ffill()

    return result


def generate_synthetic_data(
    n_samples: int = 1000,
    n_products: int = 10,
    price_range: Tuple[float, float] = (10.0, 100.0),
    elasticity: float = -1.5,
    base_demand: float = 100.0,
    noise_std: float = 10.0,
    include_confounders: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic pricing data with known causal structure.

    Creates a DataFrame with price-demand relationships, optional confounders,
    and known ground-truth parameters for testing.

    Args:
        n_samples: Number of data points.
        n_products: Number of distinct products.
        price_range: (min, max) price range.
        elasticity: True price elasticity of demand.
        base_demand: Base demand level.
        noise_std: Standard deviation of demand noise.
        include_confounders: Whether to include confounding variables.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: product_id, price, current_price, demand,
        and optionally day_of_week, promotion_active, competitor_price.
    """
    rng = np.random.RandomState(seed)

    prices = rng.uniform(price_range[0], price_range[1], n_samples)
    ref_price = (price_range[0] + price_range[1]) / 2.0

    # Demand model: D = base_demand * (P / ref_price)^elasticity + noise
    demands = base_demand * (prices / ref_price) ** elasticity
    demands = demands + rng.normal(0, noise_std, n_samples)
    demands = np.maximum(0, demands).astype(int)

    data = {
        "product_id": [f"PROD_{i % n_products:03d}" for i in range(n_samples)],
        "price": prices,
        "current_price": prices,
        "demand": demands,
        "inventory_level": rng.randint(10, 500, n_samples),
    }

    if include_confounders:
        data["day_of_week"] = rng.randint(0, 7, n_samples)
        data["promotion_active"] = rng.choice([0, 1], n_samples)
        data["competitor_price"] = prices + rng.uniform(-10, 10, n_samples)

    return pd.DataFrame(data)


def split_train_test(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    time_aware: bool = False,
    time_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    Supports both random and time-aware (temporal) splitting.

    Args:
        df: Input DataFrame.
        test_ratio: Fraction of data for testing.
        time_aware: If True, split based on time order (later data = test).
        time_column: Column to sort by for time-aware split.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If test_ratio is invalid or time_column not found.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    if time_aware:
        if time_column and time_column in df.columns:
            sorted_df = df.sort_values(time_column)
        else:
            sorted_df = df.copy()

        split_idx = int(len(sorted_df) * (1 - test_ratio))
        train = sorted_df.iloc[:split_idx].copy()
        test = sorted_df.iloc[split_idx:].copy()
    else:
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train = shuffled.iloc[:split_idx].copy()
        test = shuffled.iloc[split_idx:].copy()

    return train, test
