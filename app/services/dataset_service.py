import time
from pathlib import Path

from app.data.ccxt_provider import CCXTMarketDataProvider
from app.data.csv_storage import CsvMarketDataStorage
from app.data.market_data_provider import MarketDataError
from app.data.validators import EXPECTED_COLUMNS


class DatasetService:
    """Dataset coverage and bulk download workflows."""

    def __init__(
        self,
        storage: CsvMarketDataStorage,
        provider: CCXTMarketDataProvider,
        symbols: list[str],
        timeframes: list[str],
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.symbols = symbols
        self.timeframes = timeframes
        self._status_cache: dict[str, tuple[float, list[dict[str, object]]]] = {}

    def build_status(self, exchange: str) -> list[dict[str, object]]:
        cache_key = exchange.lower()
        cached = self._status_cache.get(cache_key)
        if cached is not None and time.monotonic() - cached[0] < 30:
            return cached[1]

        rows: list[dict[str, object]] = []
        for candidate_symbol in self.symbols:
            timeframe_rows = []
            total_rows = 0
            available = 0
            for candidate_timeframe in self.timeframes:
                path = self.storage.get_path(
                    exchange, candidate_symbol, candidate_timeframe
                )
                row_count, status = self._read_lightweight_status(path)
                total_rows += row_count
                if row_count:
                    available += 1
                timeframe_rows.append(
                    {
                        "timeframe": candidate_timeframe,
                        "rows": row_count,
                        "status": status,
                    }
                )

            rows.append(
                {
                    "symbol": candidate_symbol,
                    "available": available,
                    "total": len(self.timeframes),
                    "total_rows": total_rows,
                    "timeframes": timeframe_rows,
                }
            )

        self._status_cache[cache_key] = (time.monotonic(), rows)
        return rows

    def download_all_listed(
        self,
        exchange: str,
        since: str,
        until: str | None,
        limit_per_request: int,
        max_batches: int | None,
    ) -> tuple[list[str], list[str]]:
        messages: list[str] = []
        errors: list[str] = []

        for candidate_symbol in self.symbols:
            for candidate_timeframe in self.timeframes:
                try:
                    fetched_data = self.provider.fetch_ohlcv_history(
                        exchange=exchange,
                        symbol=candidate_symbol,
                        timeframe=candidate_timeframe,
                        since=since,
                        until=until,
                        limit_per_request=limit_per_request,
                        max_batches=max_batches,
                    )
                    path = self.storage.save(
                        data=fetched_data,
                        exchange=exchange,
                        symbol=candidate_symbol,
                        timeframe=candidate_timeframe,
                    )
                    saved_data = self.storage.load(
                        exchange,
                        candidate_symbol,
                        candidate_timeframe,
                    )
                    messages.append(
                        f"Saved {len(saved_data)} candles for {exchange} "
                        f"{candidate_symbol} {candidate_timeframe} into {path}"
                    )
                except (MarketDataError, ValueError, OSError) as exc:
                    errors.append(
                        f"{exchange} {candidate_symbol} {candidate_timeframe}: {exc}"
                    )

        self._status_cache.pop(exchange.lower(), None)
        return messages, errors

    def redownload_dataset(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        since: str,
        until: str | None,
        limit_per_request: int,
        max_batches: int | None,
    ) -> tuple[str, str | None]:
        try:
            fetched_data = self.provider.fetch_ohlcv_history(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                until=until,
                limit_per_request=limit_per_request,
                max_batches=max_batches,
            )
            path = self.storage.save(
                data=fetched_data,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
            )
            saved_data = self.storage.load(exchange, symbol, timeframe)
            self._status_cache.pop(exchange.lower(), None)
            return (
                f"Saved {len(saved_data)} candles for {exchange} {symbol} "
                f"{timeframe} into {path}",
                None,
            )
        except (MarketDataError, ValueError, OSError) as exc:
            return "", str(exc)

    def _read_lightweight_status(self, path: Path) -> tuple[int, str]:
        if not path.exists():
            return 0, "Missing"

        try:
            with path.open("rb") as handle:
                header = handle.readline().decode("utf-8", errors="replace").strip()
                expected_header = ",".join(EXPECTED_COLUMNS)
                if header != expected_header:
                    return 0, "Invalid header"

                row_count = 0
                while chunk := handle.read(1024 * 1024):
                    row_count += chunk.count(b"\n")
        except OSError as exc:
            return 0, f"Invalid: {exc}"

        return row_count, "OK" if row_count else "Empty"
