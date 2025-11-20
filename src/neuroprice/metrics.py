"""Evaluation metrics for pricing model assessment.

Provides standard regression metrics (MAE, RMSE, MAPE) plus
domain-specific revenue comparison and price distribution utilities.
"""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

__all__ = [
    "mae",
    "rmse",
    "mape",
    "revenue_comparison",
    "price_distribution_summary",
]


def _to_array(y: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert input to a float64 numpy array."""
    return np.asarray(y, dtype=np.float64)


# ---------------------------------------------------------------------------
# Standard regression metrics
# ---------------------------------------------------------------------------


def mae(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray]) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAE as a float.
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray]) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error.

    Handles division by zero by replacing zero denominators with
    ``epsilon`` so the metric remains finite.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        MAPE as a fraction (e.g. 0.15 = 15%).
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    denom = np.where(np.abs(yt) < epsilon, epsilon, np.abs(yt))
    return float(np.mean(np.abs(yt - yp) / denom))


# ---------------------------------------------------------------------------
# Domain-specific metrics
# ---------------------------------------------------------------------------


def revenue_comparison(
    actual_prices: Union[Sequence[float], np.ndarray],
    recommended_prices: Union[Sequence[float], np.ndarray],
    demands: Union[Sequence[float], np.ndarray],
) -> Dict[str, float]:
    """Compare revenue between actual and recommended pricing.

    Computes ``revenue = price * demand`` for both the actual and
    recommended price scenarios and reports totals and improvement.

    Args:
        actual_prices: Prices that were actually charged.
        recommended_prices: Prices recommended by the model.
        demands: Observed (or estimated) demand units per product.

    Returns:
        Dictionary with keys:
        - ``actual_revenue``: Total revenue at actual prices.
        - ``recommended_revenue``: Total revenue at recommended prices.
        - ``revenue_improvement``: Absolute difference (recommended - actual).
        - ``improvement_pct``: Percentage improvement.
    """
    ap = _to_array(actual_prices)
    rp = _to_array(recommended_prices)
    d = _to_array(demands)

    actual_rev = float(np.sum(ap * d))
    rec_rev = float(np.sum(rp * d))
    improvement = rec_rev - actual_rev
    pct = (improvement / actual_rev * 100.0) if actual_rev != 0 else 0.0

    return {
        "actual_revenue": actual_rev,
        "recommended_revenue": rec_rev,
        "revenue_improvement": improvement,
        "improvement_pct": pct,
    }


def price_distribution_summary(
    prices: Union[Sequence[float], np.ndarray],
    percentiles: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Summarize the distribution of a price array.

    Args:
        prices: Array of prices.
        percentiles: Percentile values to include (default 25, 50, 75, 90, 95).

    Returns:
        Dictionary with mean, std, min, max, and requested percentiles.
    """
    p = _to_array(prices)
    if percentiles is None:
        percentiles = [25.0, 50.0, 75.0, 90.0, 95.0]

    result: Dict[str, Any] = {
        "mean": float(np.mean(p)),
        "std": float(np.std(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "count": len(p),
    }

    for pct in percentiles:
        result[f"p{pct:.0f}"] = float(np.percentile(p, pct))

    return result
