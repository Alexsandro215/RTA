import pandas as pd


def add_sma(data: pd.DataFrame, period: int, source: str = "close") -> pd.DataFrame:
    """Add a simple moving average column to a copy of the DataFrame."""
    if period <= 0:
        raise ValueError("period must be greater than 0")

    result = data.copy()
    result[f"sma_{period}"] = result[source].rolling(window=period).mean()
    return result


def add_ema(data: pd.DataFrame, period: int, source: str = "close") -> pd.DataFrame:
    """Add an exponential moving average column to a copy of the DataFrame."""
    if period <= 0:
        raise ValueError("period must be greater than 0")

    result = data.copy()
    result[f"ema_{period}"] = result[source].ewm(span=period, adjust=False).mean()
    return result


def add_default_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Add the moving averages currently used by the historical viewer."""
    result = add_sma(data, period=20)
    result = add_sma(result, period=50)
    result = add_ema(result, period=200)
    return result
