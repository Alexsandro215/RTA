from typing import cast

import pandas as pd

from app.data.market_data_provider import (
    EmptyMarketDataError,
    MarketDataValidationError,
)


EXPECTED_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "datetime",
]

NUMERIC_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_ohlcv_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a clean OHLCV DataFrame."""
    if data.empty:
        raise EmptyMarketDataError("The OHLCV DataFrame is empty.")

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise MarketDataValidationError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    data = cast(pd.DataFrame, data[EXPECTED_COLUMNS].copy())

    has_nulls = bool(data[EXPECTED_COLUMNS].isna().to_numpy().any())
    if has_nulls:
        raise MarketDataValidationError("OHLCV data contains null values.")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="raise")

    if data["timestamp"].duplicated().any():
        raise MarketDataValidationError("OHLCV data contains duplicated timestamps.")

    if not data["timestamp"].is_monotonic_increasing:
        data = data.sort_values("timestamp").reset_index(drop=True)

    return data
