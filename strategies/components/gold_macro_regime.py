from __future__ import annotations


def directional_trend_state(
    *,
    close_value: float,
    fast_value: float,
    slow_value: float,
    ret_fast_value: float,
    ret_slow_value: float,
    min_fast_ret_pct: float = 0.0,
    min_slow_ret_pct: float = 0.0,
) -> int:
    """
    Return a simple signed slow-trend state.

    1  -> confirmed uptrend
    0  -> mixed / unavailable
    -1 -> confirmed downtrend
    """
    values = (
        close_value,
        fast_value,
        slow_value,
        ret_fast_value,
        ret_slow_value,
    )
    if any(v != v for v in values):  # NaN-safe without importing numpy
        return 0

    if (
        close_value >= fast_value >= slow_value
        and ret_fast_value >= min_fast_ret_pct
        and ret_slow_value >= min_slow_ret_pct
    ):
        return 1

    if (
        close_value <= fast_value <= slow_value
        and ret_fast_value <= -min_fast_ret_pct
        and ret_slow_value <= -min_slow_ret_pct
    ):
        return -1

    return 0


def weighted_gold_macro_regime_score(
    *,
    dollar_state: int,
    dollar_weight: float,
    rates_state: int,
    rates_weight: float,
    long_rates_state: int,
    long_rates_weight: float,
    peer_state: int,
    peer_weight: float,
    miners_state: int,
    miners_weight: float,
    riskoff_state: int,
    riskoff_weight: float,
) -> float:
    """
    Build a signed gold macro regime score.

    Positive values are bullish for gold, negative values are bearish.
    Dollar strength is inverted because a stronger dollar is usually hostile
    to gold, while downtrend in the dollar is supportive.
    """
    return (
        (-float(dollar_state) * dollar_weight)
        + (float(rates_state) * rates_weight)
        + (float(long_rates_state) * long_rates_weight)
        + (float(peer_state) * peer_weight)
        + (float(miners_state) * miners_weight)
        + (float(riskoff_state) * riskoff_weight)
    )


def gold_macro_regime_state(
    *,
    score: float,
    bullish_threshold: float,
    bearish_threshold: float,
) -> str:
    if score >= bullish_threshold:
        return "bullish"
    if score <= bearish_threshold:
        return "bearish"
    return "neutral"
