"""Legacy YieldManager pricing algorithms.

This module provides basic cost-plus pricing and constrained optimization
for dynamic pricing using scipy. It is retained for reference and backward
compatibility; prefer the main neuroprice API for new work.
"""

import random
from typing import Optional

import numpy as np
import scipy.optimize as optimize
from numpy.typing import NDArray


def cost_plus(costs: float, stock: int = 0) -> float:
    """Calculate cost-plus price with a fixed 40% markup.

    Args:
        costs: Base cost of the product.
        stock: Number of units in stock (used for profit simulation).

    Returns:
        The price after applying a 40% markup on costs.
    """
    total_costs = costs
    price = total_costs + (0.4 * total_costs)
    # sp = [tc / (100 - % of margin)] * 100
    p_0 = (total_costs / (100 - 50)) * 100

    for x in range(1, stock + 1):
        sales = price * x
        s_0 = p_0 * x
        pro_0 = s_0 - (total_costs * x)
        profit = sales - (total_costs * x)

    return price


def price(
    x: NDArray[np.floating],
    a: float = 200,
    b: float = 10,
    d: float = 10,
    t: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Calculate price given demand *x* and time vector *t*.

    Uses the model:  P(x, t) = (a - b*x) * d / (d + t)

    Args:
        x: Demand values (scalar or array).
        a: Maximum willingness-to-pay intercept.
        b: Price sensitivity coefficient.
        d: Time-decay denominator parameter.
        t: Time periods. Defaults to np.linspace(1, 10, 10).

    Returns:
        Array of prices for each demand/time combination.
    """
    if t is None:
        t = np.linspace(1, 10, 10)
    return (a - b * x) * d / (d + t)


def price_values(p: float, steps: int = 9, decay: float = 0.033) -> list[float]:
    """Generate a sequence of decaying price values.

    Args:
        p: Starting price.
        steps: Number of price steps to generate.
        decay: Fraction to reduce price by at each step.

    Returns:
        List of price values after successive decay.
    """
    p_vals: list[float] = []
    for _ in range(1, steps + 1):
        p = p - (decay * p)
        p_vals.append(p)
    return p_vals


def demand(
    p: NDArray[np.floating],
    a: float = 200,
    b: float = 10,
    d: float = 10,
    t: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Calculate demand given an array of prices *p* for times *t*.

    Uses the model:  D(p, t) = (1/b) * (a - p*(d+t)/d)

    Args:
        p: Price values (scalar or array).
        a: Maximum willingness-to-pay intercept.
        b: Price sensitivity coefficient.
        d: Time-decay denominator parameter.
        t: Time periods. Defaults to np.linspace(1, 10, 10).

    Returns:
        Array of demand quantities.
    """
    if t is None:
        t = np.linspace(1, 10, 10)
    return 1.0 / b * (a - p * (d + t) / d)


def objective(
    x_t: NDArray[np.floating],
    a: float = 512,
    b: float = 10,
    d: float = 10,
    t: Optional[NDArray[np.floating]] = None,
) -> float:
    """Negative total revenue (to be *minimised* by the optimiser).

    Args:
        x_t: Demand allocation vector.
        a: Maximum willingness-to-pay intercept.
        b: Price sensitivity coefficient.
        d: Time-decay denominator parameter.
        t: Time periods.

    Returns:
        Negative total revenue (scalar).
    """
    if t is None:
        t = np.linspace(1, 10, 10)
    return float(-1.0 * np.sum(x_t * price(x_t, a=a, b=b, d=d, t=t)))


def constraint_1(x_t: NDArray[np.floating], s_0: float = 150) -> float:
    """Total-demand constraint: sum(x_t) <= s_0.

    Args:
        x_t: Demand allocation vector.
        s_0: Total available stock.

    Returns:
        Non-negative slack when constraint is satisfied.
    """
    return s_0 - np.sum(x_t)


def constraint_2(x_t: NDArray[np.floating]) -> NDArray[np.floating]:
    """Non-negativity constraint: x_t >= 0.

    Args:
        x_t: Demand allocation vector.

    Returns:
        The vector itself (non-negative when constraint satisfied).
    """
    return x_t


def constraint_3(
    x_t: NDArray[np.floating], a: float = 200, b: float = 10
) -> NDArray[np.floating]:
    """Upper-bound constraint: x_t <= a/b.

    Args:
        x_t: Demand allocation vector.
        a: Maximum willingness-to-pay intercept.
        b: Price sensitivity coefficient.

    Returns:
        Non-negative slack when constraint is satisfied.
    """
    return (a / b) - x_t


def dynamic_pricing(
    time: Optional[NDArray[np.floating]] = None,
    stock: int = 0,
    a: float = 1650,
    b: float = 10.0,
    d: float = 10.0,
) -> optimize.OptimizeResult:
    """Run constrained revenue-optimisation for a single product.

    Maximises total revenue across time periods subject to stock,
    non-negativity, and demand-cap constraints using SLSQP.

    Args:
        time: Time-period vector. Defaults to np.linspace(1, 10, 10).
        stock: Total available inventory (s_0).
        a: Maximum willingness-to-pay intercept.
        b: Price sensitivity coefficient.
        d: Time-decay denominator parameter.

    Returns:
        A scipy OptimizeResult containing the optimal demand allocation,
        the maximised revenue, and convergence information.

    Example::

        result = dynamic_pricing(stock=150)
        print("Optimal demand:", result.x)
        print("Max revenue:  ", -result.fun)
    """
    if time is None:
        time = np.linspace(1, 10, 10)

    s_0 = stock
    t = time

    # Starting values
    x_start = 3.0 * np.ones(len(t))

    # Bounds on the values
    bounds = tuple((0, 20.0) for _ in x_start)

    # Constraints
    constraints = (
        {"type": "ineq", "fun": lambda x, s_0=s_0: constraint_1(x, s_0=s_0)},
        {"type": "ineq", "fun": lambda x: constraint_2(x)},
        {"type": "ineq", "fun": lambda x, a=a, b=b: constraint_3(x, a=a, b=b)},
    )

    opt_result = optimize.minimize(
        objective,
        x_start,
        args=(a, b, d, t),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return opt_result
