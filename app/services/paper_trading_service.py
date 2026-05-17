import json
import time
from pathlib import Path
from typing import cast

import pandas as pd

from app.data.ccxt_provider import CCXTMarketDataProvider
from app.data.csv_storage import CsvMarketDataStorage
from app.data.market_data_provider import MarketDataError
from app.indicators.moving_averages import add_default_moving_averages
from app.services.strategy_registry import apply_strategy
from app.systems.master_config import MasterConfigStorage


class PaperTradingService:
    """Persist and advance fictional IRT paper-trading runs."""

    def __init__(
        self,
        storage: CsvMarketDataStorage,
        provider: CCXTMarketDataProvider,
        master_storage: MasterConfigStorage,
        runs_path: str | Path = "data/irt_runs.json",
        closed_runs_path: str | Path = "data/irt_closed_runs.json",
        poll_seconds: int = 60,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.master_storage = master_storage
        self.runs_path = Path(runs_path)
        self.closed_runs_path = Path(closed_runs_path)
        self.poll_seconds = poll_seconds

    def load_runs(self) -> list[dict]:
        if not self.runs_path.exists():
            return []

        try:
            raw = json.loads(self.runs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        return raw if isinstance(raw, list) else []

    def get_run(self, run: dict) -> dict | None:
        key = self.run_key(run)
        for item in self.load_runs():
            if self.run_key(item) == key:
                return item

        return None

    def save_run(self, run: dict) -> dict:
        self.runs_path.parent.mkdir(parents=True, exist_ok=True)
        runs = self.load_runs()
        key = self.run_key(run)
        now = pd.Timestamp.utcnow().isoformat()
        next_runs = []
        existing_created_at = None
        existing_started_timestamp = None
        existing_started_datetime = None

        for item in runs:
            if self.run_key(item) == key:
                existing_created_at = item.get("created_at")
                existing_started_timestamp = item.get("started_timestamp")
                existing_started_datetime = item.get("started_datetime")
                continue
            next_runs.append(item)

        saved_run = self.normalize_run(
            {
                **run,
                "created_at": existing_created_at or now,
                "started_timestamp": existing_started_timestamp
                if existing_started_timestamp is not None
                else run.get("started_timestamp"),
                "started_datetime": existing_started_datetime
                if existing_started_datetime is not None
                else run.get("started_datetime"),
                "last_run_at": now,
            }
        )
        next_runs.append(saved_run)
        self._write_runs(next_runs)
        return saved_run

    def replace_run(self, updated_run: dict) -> dict:
        runs = self.load_runs()
        key = self.run_key(updated_run)
        next_runs = [item for item in runs if self.run_key(item) != key]
        normalized = self.normalize_run(updated_run)
        normalized["last_run_at"] = pd.Timestamp.utcnow().isoformat()
        next_runs.append(normalized)
        self._write_runs(next_runs)
        return normalized

    def delete_run(self, run: dict) -> None:
        key = self.run_key(run)
        next_runs = [item for item in self.load_runs() if self.run_key(item) != key]
        self._write_runs(next_runs)

    def load_closed_runs(self) -> list[dict]:
        if not self.closed_runs_path.exists():
            return []

        try:
            raw = json.loads(self.closed_runs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        return raw if isinstance(raw, list) else []

    def save_closed_run(self, run: dict) -> None:
        self.closed_runs_path.parent.mkdir(parents=True, exist_ok=True)
        runs = self.load_closed_runs()
        runs.insert(0, run)
        self.closed_runs_path.write_text(
            json.dumps(runs[:200], indent=2),
            encoding="utf-8",
        )

    def normalize_run(self, run: dict) -> dict:
        initial_capital = float(run.get("initial_capital", 1000.0))
        result = dict(run)
        result.setdefault("enabled", True)
        result.setdefault("equity", initial_capital)
        result.setdefault("cash_equity", result.get("equity", initial_capital))
        result.setdefault("position", "flat")
        result.setdefault("entry_price", None)
        result.setdefault("entry_datetime", None)
        result.setdefault("entry_equity", None)
        result.setdefault("last_processed_timestamp", result.get("started_timestamp"))
        result.setdefault("last_signal", 0)
        result.setdefault("trades", 0)
        result.setdefault("winning_trades", 0)
        result.setdefault("losing_trades", 0)
        result.setdefault("max_drawdown_pct", 0.0)
        result.setdefault("peak_equity", initial_capital)
        result.setdefault("trade_log", [])
        result.setdefault("equity_curve", [])
        return result

    def trading_loop(self) -> None:
        while True:
            try:
                self.refresh_runs()
            except Exception:
                pass
            time.sleep(self.poll_seconds)

    def refresh_runs(self) -> None:
        for run in self.load_runs():
            if not self.is_run_enabled(run):
                continue
            try:
                self.refresh_single_run(run)
            except Exception:
                continue

    def refresh_single_run(self, run: dict | None) -> dict | None:
        if run is None:
            return None

        run = self.normalize_run(run)
        if not self.is_run_enabled(run):
            return run
        symbol = str(run.get("symbol", "BTC/USDT"))
        timeframe = str(run.get("timeframe", "4h"))
        strategy = str(run.get("strategy", "none"))
        lookback = int(run.get("lookback", 500))
        started_timestamp = run.get("started_timestamp")
        if started_timestamp is None:
            return self.replace_run(run)

        live_data = self.provider.fetch_ohlcv(
            exchange="binance",
            symbol=symbol,
            timeframe=timeframe,
            limit=lookback,
        )
        self.storage.save(live_data, "binance", symbol, timeframe)
        analyzed = add_default_moving_averages(live_data)
        analyzed = apply_strategy(
            data=analyzed,
            strategy=strategy,
            fast_period=20,
            slow_period=50,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            bollinger_period=40,
            bollinger_std=2.2,
            master_storage=self.master_storage,
        )
        last_processed = int(run.get("last_processed_timestamp") or started_timestamp)
        new_rows = analyzed[analyzed["timestamp"] > last_processed]

        for row in new_rows.itertuples(index=False):
            run = self.process_paper_row(run, row)

        if not analyzed.empty:
            latest = analyzed.iloc[-1]
            run["latest_price"] = float(latest["close"])
            run["latest_datetime"] = str(latest["datetime"])

        return self.replace_run(run)

    def finalize_run(self, run: dict) -> dict:
        run = self.normalize_run(run)
        latest = self.latest_market_point(run)
        close = float(latest["close"])
        timestamp = int(latest["timestamp"])
        latest_datetime = str(latest["datetime"])
        run["latest_price"] = close
        run["latest_datetime"] = latest_datetime

        if run.get("position") == "long":
            run = self.close_paper_long_position(
                run=run,
                close=close,
                timestamp=timestamp,
                exit_datetime=latest_datetime,
                forced=True,
            )
        else:
            marked_equity = self.marked_equity(run, close)
            run["equity"] = marked_equity
            curve = list(run.get("equity_curve", []))
            curve.append({"datetime": latest_datetime, "equity": marked_equity})
            run["equity_curve"] = curve[-1000:]

        final_equity = self.marked_equity(run, close)
        run["status"] = "closed"
        run["closed_at"] = pd.Timestamp.utcnow().isoformat()
        run["closed_timestamp"] = timestamp
        run["closed_datetime"] = latest_datetime
        run["final_equity"] = final_equity
        run["final_profit_loss"] = final_equity - float(
            run.get("initial_capital", 1000.0)
        )
        run["position"] = "flat"
        run["entry_price"] = None
        run["entry_datetime"] = None
        run["entry_equity"] = None
        return run

    def process_paper_row(self, run: dict, row) -> dict:
        signal = int(getattr(row, "signal", 0))
        close = float(getattr(row, "close"))
        timestamp = int(getattr(row, "timestamp"))
        row_datetime = str(getattr(row, "datetime"))
        fee_rate = float(run.get("fee_bps", 0.0)) / 10_000
        slippage_rate = float(run.get("slippage_bps", 0.0)) / 10_000
        equity = float(run.get("equity", run.get("initial_capital", 1000.0)))
        position = run.get("position", "flat")
        run["last_signal"] = signal
        run["last_processed_timestamp"] = timestamp

        if position == "flat" and signal == 1:
            entry_price = close * (1 + slippage_rate)
            entry_equity = equity * (1 - fee_rate)
            run.update(
                {
                    "position": "long",
                    "entry_price": entry_price,
                    "entry_datetime": row_datetime,
                    "entry_equity": entry_equity,
                    "equity": entry_equity,
                }
            )
        elif position == "long" and signal == -1:
            run = self.close_paper_long_position(
                run=run,
                close=close,
                timestamp=timestamp,
                exit_datetime=row_datetime,
            )

        marked_equity = self.marked_equity(run, close)
        peak_equity = max(float(run.get("peak_equity", marked_equity)), marked_equity)
        drawdown_pct = (
            (marked_equity / peak_equity - 1) * 100 if peak_equity > 0 else 0.0
        )
        run["peak_equity"] = peak_equity
        run["max_drawdown_pct"] = min(
            float(run.get("max_drawdown_pct", 0.0)), drawdown_pct
        )
        curve = list(run.get("equity_curve", []))
        curve.append({"datetime": row_datetime, "equity": marked_equity})
        run["equity_curve"] = curve[-1000:]
        return run

    def close_paper_long_position(
        self,
        run: dict,
        close: float,
        timestamp: int,
        exit_datetime: str,
        forced: bool = False,
    ) -> dict:
        fee_rate = float(run.get("fee_bps", 0.0)) / 10_000
        slippage_rate = float(run.get("slippage_bps", 0.0)) / 10_000
        equity = float(run.get("equity", run.get("initial_capital", 1000.0)))
        entry_price = float(run.get("entry_price") or close)
        entry_equity = float(run.get("entry_equity") or equity)
        exit_price = close * (1 - slippage_rate)
        gross_return = exit_price / entry_price if entry_price > 0 else 1.0
        final_equity = entry_equity * gross_return * (1 - fee_rate)
        trade_pl = final_equity - entry_equity
        trade_return_pct = (
            (final_equity / entry_equity - 1) * 100 if entry_equity > 0 else 0.0
        )
        trade_log = list(run.get("trade_log", []))
        trade_log.append(
            {
                "entry_datetime": run.get("entry_datetime"),
                "exit_datetime": exit_datetime,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": trade_return_pct,
                "pl_usd": trade_pl,
                "equity_after": final_equity,
                "forced_exit": forced,
            }
        )
        run.update(
            {
                "position": "flat",
                "entry_price": None,
                "entry_datetime": None,
                "entry_equity": None,
                "equity": final_equity,
                "last_signal": -1 if forced else int(run.get("last_signal", -1)),
                "last_processed_timestamp": timestamp,
                "trade_log": trade_log[-100:],
                "trades": int(run.get("trades", 0)) + 1,
                "winning_trades": int(run.get("winning_trades", 0))
                + (1 if trade_pl > 0 else 0),
                "losing_trades": int(run.get("losing_trades", 0))
                + (1 if trade_pl < 0 else 0),
            }
        )
        marked_equity = self.marked_equity(run, close)
        peak_equity = max(float(run.get("peak_equity", marked_equity)), marked_equity)
        drawdown_pct = (
            (marked_equity / peak_equity - 1) * 100 if peak_equity > 0 else 0.0
        )
        run["peak_equity"] = peak_equity
        run["max_drawdown_pct"] = min(
            float(run.get("max_drawdown_pct", 0.0)), drawdown_pct
        )
        return run

    def latest_market_point(self, run: dict) -> dict[str, float | int | str]:
        symbol = str(run.get("symbol", "BTC/USDT"))
        timeframe = str(run.get("timeframe", "4h"))
        try:
            live_data = self.provider.fetch_ohlcv(
                exchange="binance",
                symbol=symbol,
                timeframe=timeframe,
                limit=2,
            )
            self.storage.save(live_data, "binance", symbol, timeframe)
            if not live_data.empty:
                latest = live_data.iloc[-1]
                return {
                    "close": float(latest["close"]),
                    "timestamp": int(latest["timestamp"]),
                    "datetime": str(latest["datetime"]),
                }
        except (MarketDataError, ValueError):
            pass

        data = self.storage.load("binance", symbol, timeframe)
        if not data.empty:
            latest = data.iloc[-1]
            return {
                "close": float(latest["close"]),
                "timestamp": int(latest["timestamp"]),
                "datetime": str(latest["datetime"]),
            }

        close = float(run.get("latest_price") or 0.0)
        return {
            "close": close,
            "timestamp": int(
                run.get("last_processed_timestamp") or run.get("started_timestamp") or 0
            ),
            "datetime": str(
                run.get("latest_datetime") or run.get("started_datetime") or "-"
            ),
        }

    def latest_close(self, run: dict) -> float | None:
        if run.get("latest_price") is not None:
            return float(run["latest_price"])

        data = self.storage.load(
            "binance",
            str(run.get("symbol", "BTC/USDT")),
            str(run.get("timeframe", "4h")),
        )
        if data.empty:
            return None

        return float(data.iloc[-1]["close"])

    def marked_equity(self, run: dict, latest_close: float | None) -> float:
        equity = float(run.get("equity", run.get("initial_capital", 1000.0)))
        if latest_close is None or run.get("position") != "long":
            return equity

        entry_price = float(run.get("entry_price") or latest_close)
        entry_equity = float(run.get("entry_equity") or equity)
        if entry_price <= 0:
            return equity

        return entry_equity * (latest_close / entry_price)

    def equity_curve(self, run: dict) -> pd.DataFrame:
        curve = run.get("equity_curve", [])
        if not curve:
            return pd.DataFrame(columns=["datetime", "equity"])

        return pd.DataFrame(curve)

    def run_key(self, run: dict) -> str:
        return "|".join(
            [
                str(run.get("symbol", "")),
                str(run.get("timeframe", "")),
                str(run.get("strategy", "")),
            ]
        )

    def hide_pre_irt_signals(
        self, data: pd.DataFrame, started_timestamp: int
    ) -> pd.DataFrame:
        result = data.copy()
        if "timestamp" in result.columns and "signal" in result.columns:
            result.loc[result["timestamp"] < started_timestamp, "signal"] = 0

        return result

    def slice_irt_data(
        self, data: pd.DataFrame, started_timestamp: int
    ) -> pd.DataFrame:
        if data.empty or "timestamp" not in data.columns:
            return data.copy()

        result = cast(pd.DataFrame, data[data["timestamp"] >= started_timestamp].copy())
        return result.reset_index(drop=True)

    def is_run_enabled(self, run: dict) -> bool:
        return bool(run.get("enabled", True))

    def _write_runs(self, runs: list[dict]) -> None:
        runs = sorted(
            runs,
            key=lambda item: str(item.get("last_run_at", "")),
            reverse=True,
        )
        self.runs_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
