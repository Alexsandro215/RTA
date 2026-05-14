import pandas as pd

from app.indicators.momentum import add_rsi


def apply_rsi_mean_reversion_strategy(
    data: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """Buy when RSI leaves oversold and sell when RSI leaves overbought."""
    if oversold >= overbought:
        raise ValueError("oversold must be lower than overbought")

    result = data.copy()
    rsi_column = f"rsi_{period}"
    if rsi_column not in result.columns:
        result = add_rsi(result, period)

    was_oversold = result[rsi_column].shift(1) < oversold
    leaves_oversold = (result[rsi_column] >= oversold) & was_oversold
    was_overbought = result[rsi_column].shift(1) > overbought
    leaves_overbought = (result[rsi_column] <= overbought) & was_overbought

    result["signal"] = 0
    result.loc[leaves_oversold, "signal"] = 1
    result.loc[leaves_overbought, "signal"] = -1
    result["strategy"] = f"RSI {period} Mean Reversion"
    return result
