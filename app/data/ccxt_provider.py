from __future__ import annotations

import time
from typing import Any

import ccxt
import pandas as pd

from app.data.market_data_provider import (
    EmptyMarketDataError,
    InvalidExchangeError,
    InvalidSymbolError,
    MarketDataConnectionError,
    MarketDataProvider,
)
from app.data.validators import validate_ohlcv_dataframe


class CCXTMarketDataProvider(MarketDataProvider):
    """Market data provider backed by the ccxt exchange library."""

    COLUMNS: list[str] = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        exchange_client = self._create_exchange(exchange)
        self._load_markets(exchange_client, exchange)
        self._validate_symbol(exchange_client, symbol, exchange)

        try:
            rows = exchange_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
        except ccxt.BadSymbol as exc:
            raise InvalidSymbolError(
                f"Symbol '{symbol}' is not available on exchange '{exchange}'."
            ) from exc
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            raise MarketDataConnectionError(
                f"Could not connect to exchange '{exchange}'."
            ) from exc
        except ccxt.ExchangeError as exc:
            raise MarketDataConnectionError(
                f"Exchange '{exchange}' returned an error: {exc}"
            ) from exc

        data = self._normalize_ohlcv(rows)
        return validate_ohlcv_dataframe(data)

    def fetch_ohlcv_history(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        since: str,
        until: str | None = None,
        limit_per_request: int = 1000,
        max_batches: int | None = None,
    ) -> pd.DataFrame:
        if limit_per_request <= 0:
            raise ValueError("limit_per_request must be greater than 0")

        exchange_client = self._create_exchange(exchange)
        self._load_markets(exchange_client, exchange)
        self._validate_symbol(exchange_client, symbol, exchange)

        since_ms = self._parse_datetime_to_ms(since, "since")
        until_ms = self._parse_datetime_to_ms(until, "until") if until else None

        if until_ms is not None and until_ms <= since_ms:
            raise ValueError("until must be greater than since")

        timeframe_ms = int(exchange_client.parse_timeframe(timeframe) * 1000)
        all_rows: list[list[Any]] = []
        cursor_ms = since_ms
        batches = 0

        while True:
            if until_ms is not None and cursor_ms >= until_ms:
                break

            if max_batches is not None and batches >= max_batches:
                break

            rows = self._fetch_ohlcv_batch(
                exchange_client=exchange_client,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since_ms=cursor_ms,
                limit=limit_per_request,
            )

            if not rows:
                break

            if until_ms is not None:
                rows = [row for row in rows if row[0] < until_ms]

            if not rows:
                break

            all_rows.extend(rows)
            batches += 1

            last_timestamp = int(rows[-1][0])
            next_cursor_ms = last_timestamp + timeframe_ms
            if next_cursor_ms <= cursor_ms:
                break

            cursor_ms = next_cursor_ms

            if len(rows) < limit_per_request:
                break

            if exchange_client.rateLimit:
                time.sleep(exchange_client.rateLimit / 1000)

        if not all_rows:
            raise EmptyMarketDataError("The exchange returned no historical OHLCV data.")

        data = self._normalize_ohlcv(all_rows)
        data = data.drop_duplicates(subset=["timestamp"], keep="last")
        return validate_ohlcv_dataframe(data)

    def _create_exchange(self, exchange: str) -> ccxt.Exchange:
        exchange_id = exchange.lower().strip()

        if exchange_id not in ccxt.exchanges:
            raise InvalidExchangeError(f"Exchange '{exchange}' is not supported by ccxt.")

        exchange_class: type[ccxt.Exchange] = getattr(ccxt, exchange_id)
        exchange_client = exchange_class(
            {
                "enableRateLimit": True,
                "timeout": 30_000,
            }
        )

        if not exchange_client.has.get("fetchOHLCV"):
            raise InvalidExchangeError(
                f"Exchange '{exchange}' does not support OHLCV data."
            )

        return exchange_client

    def _load_markets(self, exchange_client: ccxt.Exchange, exchange: str) -> None:
        try:
            exchange_client.load_markets()
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            raise MarketDataConnectionError(
                f"Could not connect to exchange '{exchange}'."
            ) from exc
        except ccxt.ExchangeError as exc:
            raise MarketDataConnectionError(
                f"Could not load markets from exchange '{exchange}': {exc}"
            ) from exc

    def _validate_symbol(
        self,
        exchange_client: ccxt.Exchange,
        symbol: str,
        exchange: str,
    ) -> None:
        if symbol not in exchange_client.markets:
            raise InvalidSymbolError(
                f"Symbol '{symbol}' is not available on exchange '{exchange}'."
            )

    def _fetch_ohlcv_batch(
        self,
        exchange_client: ccxt.Exchange,
        exchange: str,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int,
    ) -> list[list[Any]]:
        try:
            return exchange_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
            )
        except ccxt.BadSymbol as exc:
            raise InvalidSymbolError(
                f"Symbol '{symbol}' is not available on exchange '{exchange}'."
            ) from exc
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            raise MarketDataConnectionError(
                f"Could not connect to exchange '{exchange}'."
            ) from exc
        except ccxt.ExchangeError as exc:
            raise MarketDataConnectionError(
                f"Exchange '{exchange}' returned an error: {exc}"
            ) from exc

    def _parse_datetime_to_ms(self, value: str, field_name: str) -> int:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{field_name} must be a valid date or datetime")

        return int(parsed.timestamp() * 1000)

    def _normalize_ohlcv(self, rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            raise EmptyMarketDataError("The exchange returned no OHLCV data.")

        data = pd.DataFrame(rows, columns=self.COLUMNS)
        data["datetime"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)

        return data[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "datetime",
            ]
        ]
