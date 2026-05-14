import pandas as pd


def add_rsi(data: pd.DataFrame, period: int = 14, source: str = "close") -> pd.DataFrame:
    """Add a Relative Strength Index column to a copy of the DataFrame."""
    if period <= 0:
        raise ValueError("period must be greater than 0")

    result = data.copy()
    delta = result[source].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = average_gain / average_loss

    result[f"rsi_{period}"] = 100 - (100 / (1 + relative_strength))
    return result
