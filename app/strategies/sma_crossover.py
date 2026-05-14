import pandas as pd

from app.indicators.moving_averages import add_sma


def apply_sma_crossover_strategy(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
) -> pd.DataFrame:
    """Mark buy/sell signals when a fast SMA crosses a slow SMA."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("SMA periods must be greater than 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be lower than slow_period")

    result = data.copy()
    fast_column = f"sma_{fast_period}"
    slow_column = f"sma_{slow_period}"

    if fast_column not in result.columns:
        result = add_sma(result, fast_period)
    if slow_column not in result.columns:
        result = add_sma(result, slow_period)

    fast_above_slow = result[fast_column] > result[slow_column]
    previous_fast_above_slow = fast_above_slow.shift(1)

    result["signal"] = 0
    result.loc[fast_above_slow & (previous_fast_above_slow == False), "signal"] = 1
    result.loc[(~fast_above_slow) & (previous_fast_above_slow == True), "signal"] = -1
    result["strategy"] = f"SMA {fast_period}/{slow_period} Crossover"

    return result
