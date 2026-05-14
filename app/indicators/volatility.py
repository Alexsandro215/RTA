import pandas as pd


def add_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    source: str = "close",
) -> pd.DataFrame:
    """Add Bollinger Band columns to a copy of the DataFrame."""
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0")

    result = data.copy()
    middle = result[source].rolling(window=period).mean()
    deviation = result[source].rolling(window=period).std()

    result[f"bb_middle_{period}"] = middle
    result[f"bb_upper_{period}"] = middle + deviation * std_dev
    result[f"bb_lower_{period}"] = middle - deviation * std_dev
    return result
