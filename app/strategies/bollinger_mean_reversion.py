import pandas as pd

from app.indicators.volatility import add_bollinger_bands


def apply_bollinger_mean_reversion_strategy(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Buy when price re-enters above lower band and sell when it re-enters below upper band."""
    result = data.copy()
    lower_column = f"bb_lower_{period}"
    upper_column = f"bb_upper_{period}"

    if lower_column not in result.columns or upper_column not in result.columns:
        result = add_bollinger_bands(result, period=period, std_dev=std_dev)

    was_below_lower = result["close"].shift(1) < result[lower_column].shift(1)
    reenters_above_lower = (result["close"] >= result[lower_column]) & was_below_lower
    was_above_upper = result["close"].shift(1) > result[upper_column].shift(1)
    reenters_below_upper = (result["close"] <= result[upper_column]) & was_above_upper

    result["signal"] = 0
    result.loc[reenters_above_lower, "signal"] = 1
    result.loc[reenters_below_upper, "signal"] = -1
    result["strategy"] = f"Bollinger {period}/{std_dev:g} Mean Reversion"
    return result
