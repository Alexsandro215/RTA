from pathlib import Path

import pandas as pd

from app.data.validators import EXPECTED_COLUMNS, validate_ohlcv_dataframe


class CsvMarketDataStorage:
    """Store normalized OHLCV datasets as local CSV files."""

    def __init__(self, base_dir: str | Path = "data/raw") -> None:
        self.base_dir = Path(base_dir)

    def get_path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return self.base_dir / exchange.lower() / safe_symbol / f"{timeframe}.csv"

    def load(self, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.get_path(exchange, symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        data = pd.read_csv(path)
        data = self._drop_incomplete_rows(data)
        data = data.drop_duplicates(subset=["timestamp"], keep="last")
        return validate_ohlcv_dataframe(data)

    def save(
        self,
        data: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> Path:
        path = self.get_path(exchange, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = self.load(exchange, symbol, timeframe)
        combined = pd.concat([existing, data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = validate_ohlcv_dataframe(combined)

        combined.to_csv(path, index=False)
        return path

    def _drop_incomplete_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        available_columns = [column for column in EXPECTED_COLUMNS if column in data.columns]
        if not available_columns:
            return data

        return data.dropna(subset=available_columns, how="any").reset_index(drop=True)
