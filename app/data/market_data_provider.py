from abc import ABC, abstractmethod

import pandas as pd


class MarketDataError(Exception):
    """Base exception for market data provider errors."""


class InvalidExchangeError(MarketDataError):
    """Raised when the requested exchange is not supported."""


class InvalidSymbolError(MarketDataError):
    """Raised when the requested market symbol is not available."""


class MarketDataConnectionError(MarketDataError):
    """Raised when the provider cannot connect to the market data source."""


class EmptyMarketDataError(MarketDataError):
    """Raised when the provider returns no OHLCV rows."""


class MarketDataValidationError(MarketDataError):
    """Raised when normalized market data does not pass quality checks."""


class MarketDataProvider(ABC):
    """Interface for OHLCV market data providers."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fetch normalized OHLCV data as a pandas DataFrame."""

    @abstractmethod
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
        """Fetch normalized historical OHLCV data by paginating exchange candles."""
