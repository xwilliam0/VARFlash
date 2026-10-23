"""Kalshi trading fees.

Kalshi charges a "quadratic" fee per order: ceil_to_cent(rate * C * P * (1 - P))
where C is the contract count and P the price in dollars. Takers pay 7%;
resting maker orders on fee-bearing series pay 1.75%.
"""

from __future__ import annotations

import math

TAKER_RATE = 0.07
MAKER_RATE = 0.0175


def kalshi_fee(contracts: float, price: float, rate: float = TAKER_RATE) -> float:
    if contracts <= 0:
        return 0.0
    raw = rate * contracts * price * (1.0 - price)
    return math.ceil(round(raw * 100, 6)) / 100


def fee_per_contract(price: float, rate: float = TAKER_RATE) -> float:
    """Marginal fee for large orders (ignores the per-order rounding)."""
    return rate * price * (1.0 - price)
