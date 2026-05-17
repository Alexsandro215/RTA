from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from app.backtesting.simple_backtester import BacktestResult, run_long_only_signal_backtest
from app.backtesting.system_backtester import run_master_system_backtest
from app.data.csv_storage import CsvMarketDataStorage
from app.indicators.moving_averages import add_default_moving_averages
from app.services.strategy_registry import (
    apply_strategy,
    module_strategy_options,
    trade_mode_for_strategy,
)
from app.systems.master_config import MasterConfig, MasterConfigStorage


@dataclass(frozen=True)
class BacktestAnalysisSettings:
    start_date: str
    end_date: str
    fast_period: int
    slow_period: int
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    bollinger_period: int
    bollinger_std: float
    initial_capital: float
    fee_bps: float
    slippage_bps: float
    compound: bool
    validation_pct: float


class BacktestAnalysisService:
    """Build comparison and validation views for backtest results."""

    def __init__(
        self,
        storage: CsvMarketDataStorage,
        master_storage: MasterConfigStorage,
        supported_timeframes: list[str],
        cache_dir: str | Path = "data/cache/backtest_analysis",
    ) -> None:
        self.storage = storage
        self.master_storage = master_storage
        self.supported_timeframes = supported_timeframes
        self._cache: dict[tuple[object, ...], tuple[tuple[object, ...], object]] = {}
        self.cache_dir = Path(cache_dir)

    def build_timeframe_comparison(
        self,
        exchange: str,
        symbol: str,
        strategy: str,
        settings: BacktestAnalysisSettings,
    ) -> list[dict[str, str | int]]:
        cache_key = ("timeframes", exchange, symbol, strategy, settings)
        fingerprint = self._dataset_fingerprint(exchange, symbol, self.supported_timeframes)
        cached = self._get_cached(cache_key, fingerprint)
        if cached is not None:
            return cast(list[dict[str, str | int]], cached)

        rows: list[dict[str, str | int]] = []

        for candidate_timeframe in self.supported_timeframes:
            dataset = self.storage.load(exchange, symbol, candidate_timeframe)
            if dataset.empty:
                rows.append(_empty_timeframe_row(candidate_timeframe, "Missing dataset"))
                continue

            filtered = cast(
                pd.DataFrame,
                _filter_by_date_range(
                dataset, settings.start_date, settings.end_date
                ),
            )
            if filtered.empty:
                rows.append(
                    _empty_timeframe_row(candidate_timeframe, "No rows in date range")
                )
                continue

            result, split = self._run_strategy_or_master(
                data=filtered,
                symbol=symbol,
                strategy=strategy,
                timeframe=candidate_timeframe,
                settings=settings,
            )
            alpha = result.total_return_pct - result.buy_and_hold_return_pct
            rows.append(
                {
                    "timeframe": candidate_timeframe,
                    "status": "OK",
                    "rows": len(filtered),
                    "from": str(filtered.iloc[0]["datetime"]),
                    "to": str(filtered.iloc[-1]["datetime"]),
                    "strategy_gain": _format_pct(result.total_return_pct),
                    "buy_hold": _format_pct(result.buy_and_hold_return_pct),
                    "alpha": _format_pct(alpha),
                    "drawdown": _format_pct(result.max_drawdown_pct),
                    "trades": result.trades,
                    "profit_factor": _format_ratio(result.profit_factor),
                    "validation_gain": split["validation_return_pct"],
                    "validation_alpha": split["validation_alpha_pct"],
                    "validation_profit_factor": split["validation_profit_factor"],
                    "verdict": split["validation_verdict"],
                }
            )

        self._set_cached(cache_key, fingerprint, rows)
        return rows

    def build_walk_forward_results(
        self,
        data: pd.DataFrame,
        selected_master_config: MasterConfig | None,
        strategy: str,
        settings: BacktestAnalysisSettings,
        window_years: int = 2,
    ) -> list[dict[str, str | int]]:
        if data.empty:
            return []

        cache_key = (
            "walk_forward",
            selected_master_config.name if selected_master_config else "",
            strategy,
            settings,
            window_years,
            len(data),
            str(data.iloc[0]["timestamp"]) if "timestamp" in data.columns else "",
            str(data.iloc[-1]["timestamp"]) if "timestamp" in data.columns else "",
        )
        fingerprint = cache_key
        cached = self._get_cached(cache_key, fingerprint)
        if cached is not None:
            return cast(list[dict[str, str | int]], cached)

        filtered = cast(
            pd.DataFrame,
            _filter_by_date_range(data, settings.start_date, settings.end_date),
        )
        if filtered.empty:
            return []

        start = pd.to_datetime(filtered.iloc[0]["datetime"])
        end = pd.to_datetime(filtered.iloc[-1]["datetime"])
        if pd.isna(start) or pd.isna(end) or start >= end:
            return []

        rows: list[dict[str, str | int]] = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + pd.DateOffset(years=window_years), end)
            window_data = cast(pd.DataFrame, filtered[
                (pd.to_datetime(filtered["datetime"]) >= window_start)
                & (pd.to_datetime(filtered["datetime"]) < window_end)
            ].reset_index(drop=True))

            label = f"{window_start.date()} to {window_end.date()}"
            if len(window_data) < 20:
                rows.append(
                    {
                        "window": label,
                        "status": "Too few rows",
                        "rows": len(window_data),
                        "strategy_gain": "-",
                        "buy_hold": "-",
                        "alpha": "-",
                        "drawdown": "-",
                        "trades": "-",
                        "profit_factor": "-",
                        "exposure": "-",
                        "avg_trade_days": "-",
                        "verdict": "-",
                    }
                )
                window_start = window_end
                continue

            if selected_master_config is not None:
                result = run_master_system_backtest(
                    data=window_data,
                    master=selected_master_config,
                    initial_equity=settings.initial_capital,
                    fee_bps=settings.fee_bps,
                    slippage_bps=settings.slippage_bps,
                    compound=settings.compound,
                ).backtest
            else:
                analyzed = apply_strategy(
                    data=add_default_moving_averages(window_data),
                    strategy=strategy,
                    fast_period=settings.fast_period,
                    slow_period=settings.slow_period,
                    rsi_period=settings.rsi_period,
                    rsi_oversold=settings.rsi_oversold,
                    rsi_overbought=settings.rsi_overbought,
                    bollinger_period=settings.bollinger_period,
                    bollinger_std=settings.bollinger_std,
                    master_storage=self.master_storage,
                )
                result = run_long_only_signal_backtest(
                    data=analyzed,
                    initial_equity=settings.initial_capital,
                    fee_bps=settings.fee_bps,
                    slippage_bps=settings.slippage_bps,
                    compound=settings.compound,
                    trade_mode=trade_mode_for_strategy(strategy),
                )

            alpha = result.total_return_pct - result.buy_and_hold_return_pct
            exposure = _build_exposure_metrics(window_data, result)
            rows.append(
                {
                    "window": label,
                    "status": "OK",
                    "rows": len(window_data),
                    "strategy_gain": _format_pct(result.total_return_pct),
                    "buy_hold": _format_pct(result.buy_and_hold_return_pct),
                    "alpha": _format_pct(alpha),
                    "drawdown": _format_pct(result.max_drawdown_pct),
                    "trades": result.trades,
                    "profit_factor": _format_ratio(result.profit_factor),
                    "exposure": exposure["exposure_pct"],
                    "avg_trade_days": exposure["avg_trade_days"],
                    "verdict": _build_validation_verdict(
                        validation_return_pct=result.total_return_pct,
                        validation_alpha_pct=alpha,
                        validation_profit_factor=result.profit_factor,
                    ),
                }
            )
            window_start = window_end

        self._set_cached(cache_key, fingerprint, rows)
        return rows

    def build_strategy_comparison(
        self,
        exchange: str,
        symbol: str,
        timeframes: list[str],
        settings: BacktestAnalysisSettings,
    ) -> list[dict[str, object]]:
        cache_key = ("strategies", exchange, symbol, tuple(timeframes), settings)
        fingerprint = self._dataset_fingerprint(exchange, symbol, timeframes)
        cached = self._get_cached(cache_key, fingerprint)
        if cached is not None:
            return cast(list[dict[str, object]], cached)

        groups: list[dict[str, object]] = []

        for candidate_timeframe in timeframes:
            dataset = self.storage.load(exchange, symbol, candidate_timeframe)
            rows: list[dict[str, str | int]] = []
            group: dict[str, object] = {
                "timeframe": candidate_timeframe,
                "status": "Missing",
                "rows": 0,
                "from": "-",
                "to": "-",
                "strategies": rows,
            }

            if dataset.empty:
                groups.append(group)
                continue

            filtered = cast(
                pd.DataFrame,
                _filter_by_date_range(
                dataset, settings.start_date, settings.end_date
                ),
            )
            if filtered.empty:
                group["status"] = "No rows in date range"
                groups.append(group)
                continue

            group.update(
                {
                    "status": "OK",
                    "rows": len(filtered),
                    "from": str(filtered.iloc[0]["datetime"]),
                    "to": str(filtered.iloc[-1]["datetime"]),
                }
            )
            base_data = add_default_moving_averages(filtered)

            for option in module_strategy_options():
                candidate_strategy = option["value"]
                if candidate_strategy == "none":
                    continue

                try:
                    analyzed = apply_strategy(
                        data=base_data,
                        strategy=candidate_strategy,
                        fast_period=settings.fast_period,
                        slow_period=settings.slow_period,
                        rsi_period=settings.rsi_period,
                        rsi_oversold=settings.rsi_oversold,
                        rsi_overbought=settings.rsi_overbought,
                        bollinger_period=settings.bollinger_period,
                        bollinger_std=settings.bollinger_std,
                        master_storage=self.master_storage,
                    )
                    trade_mode = trade_mode_for_strategy(candidate_strategy)
                    result = run_long_only_signal_backtest(
                        data=analyzed,
                        initial_equity=settings.initial_capital,
                        fee_bps=settings.fee_bps,
                        slippage_bps=settings.slippage_bps,
                        compound=settings.compound,
                        trade_mode=trade_mode,
                    )
                    split = self.build_split_metrics(
                        data=filtered,
                        strategy=candidate_strategy,
                        settings=settings,
                        trade_mode=trade_mode,
                    )
                    exposure = _build_exposure_metrics(filtered, result)
                except Exception as exc:
                    rows.append(
                        {
                            "strategy": option["label"],
                            "status": f"Error: {exc}",
                            "trades": "-",
                            "strategy_gain": "-",
                            "buy_hold": "-",
                            "alpha": "-",
                            "drawdown": "-",
                            "profit_factor": "-",
                            "exposure": "-",
                            "avg_trade_days": "-",
                            "validation_gain": "-",
                            "validation_alpha": "-",
                            "validation_profit_factor": "-",
                            "verdict": "-",
                        }
                    )
                    continue

                rows.append(
                    {
                        "strategy": option["label"],
                        "status": "OK",
                        "trades": result.trades,
                        "strategy_gain": _format_pct(result.total_return_pct),
                        "buy_hold": _format_pct(result.buy_and_hold_return_pct),
                        "alpha": _format_pct(
                            result.total_return_pct - result.buy_and_hold_return_pct
                        ),
                        "drawdown": _format_pct(result.max_drawdown_pct),
                        "profit_factor": _format_ratio(result.profit_factor),
                        "exposure": exposure["exposure_pct"],
                        "avg_trade_days": exposure["avg_trade_days"],
                        "validation_gain": split["validation_return_pct"],
                        "validation_alpha": split["validation_alpha_pct"],
                        "validation_profit_factor": split["validation_profit_factor"],
                        "verdict": split["validation_verdict"],
                    }
                )

            group["strategies"] = sorted(rows, key=_strategy_sort_key, reverse=True)
            groups.append(group)

        self._set_cached(cache_key, fingerprint, groups)
        return groups

    def build_strategy_comparison_summary(
        self,
        groups: list[dict[str, object]],
    ) -> list[dict[str, str | int]]:
        by_strategy: dict[str, dict[str, object]] = {}

        for group in groups:
            timeframe = str(group["timeframe"])
            strategy_rows = cast(list[dict[str, object]], group.get("strategies", []))
            for row in strategy_rows:
                if not isinstance(row, dict) or row.get("status") != "OK":
                    continue

                strategy = str(row["strategy"])
                stats = by_strategy.setdefault(
                    strategy,
                    {
                        "strategy": strategy,
                        "tested": 0,
                        "promising": 0,
                        "validation_alphas": [],
                        "drawdowns": [],
                        "exposures": [],
                        "best_timeframe": "-",
                        "best_validation_alpha": -999999.0,
                        "best_gain": "-",
                        "best_profit_factor": "-",
                    },
                )
                validation_alpha = _parse_pct(row.get("validation_alpha"))
                drawdown = _parse_pct(row.get("drawdown"))
                exposure = _parse_pct(row.get("exposure"))
                validation_alphas = cast(list[float], stats["validation_alphas"])
                drawdowns = cast(list[float], stats["drawdowns"])
                exposures = cast(list[float], stats["exposures"])

                stats["tested"] = int(cast(int, stats["tested"])) + 1
                if row.get("verdict") == "Promising":
                    stats["promising"] = int(cast(int, stats["promising"])) + 1
                if validation_alpha is not None:
                    validation_alphas.append(validation_alpha)
                    best_validation_alpha = float(
                        cast(float, stats["best_validation_alpha"])
                    )
                    if validation_alpha > best_validation_alpha:
                        stats["best_validation_alpha"] = validation_alpha
                        stats["best_timeframe"] = timeframe
                        stats["best_gain"] = str(row.get("validation_gain", "-"))
                        stats["best_profit_factor"] = str(
                            row.get("validation_profit_factor", "-")
                        )
                if drawdown is not None:
                    drawdowns.append(drawdown)
                if exposure is not None:
                    exposures.append(exposure)

        summary: list[dict[str, str | int]] = []
        for stats in by_strategy.values():
            validation_alphas = cast(list[float], stats["validation_alphas"])
            drawdowns = cast(list[float], stats["drawdowns"])
            exposures = cast(list[float], stats["exposures"])
            avg_validation_alpha = (
                sum(validation_alphas) / len(validation_alphas)
                if validation_alphas
                else 0.0
            )
            worst_drawdown = min(drawdowns) if drawdowns else 0.0
            avg_exposure = sum(exposures) / len(exposures) if exposures else 0.0
            summary.append(
                {
                    "strategy": str(stats["strategy"]),
                    "best_timeframe": str(stats["best_timeframe"]),
                    "tested": int(cast(int, stats["tested"])),
                    "promising": int(cast(int, stats["promising"])),
                    "avg_validation_alpha": _format_pct(avg_validation_alpha),
                    "worst_drawdown": _format_pct(worst_drawdown),
                    "avg_exposure": _format_pct(avg_exposure),
                    "best_validation_gain": str(stats["best_gain"]),
                    "best_validation_pf": str(stats["best_profit_factor"]),
                }
            )

        def sort_key(row: dict[str, str | int]) -> tuple[int, float]:
            return (
                int(row["promising"]),
                _parse_pct(row["avg_validation_alpha"]) or -999999.0,
            )

        return sorted(summary, key=sort_key, reverse=True)

    def build_split_metrics(
        self,
        data: pd.DataFrame,
        strategy: str,
        settings: BacktestAnalysisSettings,
        trade_mode: str,
    ) -> dict[str, str | int]:
        if data.empty:
            return _empty_split_metrics()

        validation_rows = max(1, int(len(data) * settings.validation_pct / 100))
        split_index = max(1, len(data) - validation_rows)
        train_data = data.iloc[:split_index].reset_index(drop=True)
        validation_data = data.iloc[split_index:].reset_index(drop=True)
        train_data = apply_strategy(
            data=add_default_moving_averages(train_data),
            strategy=strategy,
            fast_period=settings.fast_period,
            slow_period=settings.slow_period,
            rsi_period=settings.rsi_period,
            rsi_oversold=settings.rsi_oversold,
            rsi_overbought=settings.rsi_overbought,
            bollinger_period=settings.bollinger_period,
            bollinger_std=settings.bollinger_std,
            master_storage=self.master_storage,
        )
        validation_data = apply_strategy(
            data=add_default_moving_averages(validation_data),
            strategy=strategy,
            fast_period=settings.fast_period,
            slow_period=settings.slow_period,
            rsi_period=settings.rsi_period,
            rsi_oversold=settings.rsi_oversold,
            rsi_overbought=settings.rsi_overbought,
            bollinger_period=settings.bollinger_period,
            bollinger_std=settings.bollinger_std,
            master_storage=self.master_storage,
        )

        train_result = run_long_only_signal_backtest(
            data=train_data,
            initial_equity=settings.initial_capital,
            fee_bps=settings.fee_bps,
            slippage_bps=settings.slippage_bps,
            compound=settings.compound,
            trade_mode=trade_mode,
        )
        validation_result = run_long_only_signal_backtest(
            data=validation_data,
            initial_equity=settings.initial_capital,
            fee_bps=settings.fee_bps,
            slippage_bps=settings.slippage_bps,
            compound=settings.compound,
            trade_mode=trade_mode,
        )

        validation_alpha = (
            validation_result.total_return_pct
            - validation_result.buy_and_hold_return_pct
        )
        return {
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "validation_from": validation_data.iloc[0]["datetime"]
            if not validation_data.empty
            else "-",
            "train_return_pct": _format_pct(train_result.total_return_pct),
            "train_drawdown_pct": _format_pct(train_result.max_drawdown_pct),
            "train_profit_factor": _format_ratio(train_result.profit_factor),
            "validation_return_pct": _format_pct(validation_result.total_return_pct),
            "validation_buy_and_hold_pct": _format_pct(
                validation_result.buy_and_hold_return_pct
            ),
            "validation_alpha_pct": _format_pct(validation_alpha),
            "validation_drawdown_pct": _format_pct(
                validation_result.max_drawdown_pct
            ),
            "validation_profit_factor": _format_ratio(
                validation_result.profit_factor
            ),
            "validation_return_drawdown_ratio": _format_ratio(
                _calculate_return_drawdown_ratio(
                    validation_result.total_return_pct,
                    validation_result.max_drawdown_pct,
                )
            ),
            "validation_verdict": _build_validation_verdict(
                validation_return_pct=validation_result.total_return_pct,
                validation_alpha_pct=validation_alpha,
                validation_profit_factor=validation_result.profit_factor,
            ),
        }

    def _run_strategy_or_master(
        self,
        data: pd.DataFrame,
        symbol: str,
        strategy: str,
        timeframe: str,
        settings: BacktestAnalysisSettings,
    ) -> tuple[BacktestResult, dict[str, str | int]]:
        if strategy.startswith("master:"):
            master = self.master_storage.get(strategy.removeprefix("master:"))
            if master is None:
                raise ValueError(f"Master strategy not found: {strategy}")
            candidate_master = MasterConfig(
                name=master.name,
                symbol=symbol,
                main_strategy=master.main_strategy,
                main_timeframe=timeframe,
                main_allocation_pct=master.main_allocation_pct,
                rebound_strategy=master.rebound_strategy,
                rebound_allocation_pct=master.rebound_allocation_pct,
                rebound_flags=master.rebound_flags,
                sideways_strategy=master.sideways_strategy,
                sideways_allocation_pct=master.sideways_allocation_pct,
                sideways_flags=master.sideways_flags,
                defensive_strategy=master.defensive_strategy,
                defensive_cash_pct=master.defensive_cash_pct,
                defensive_flags=master.defensive_flags,
            )
            result = run_master_system_backtest(
                data=data,
                master=candidate_master,
                initial_equity=settings.initial_capital,
                fee_bps=settings.fee_bps,
                slippage_bps=settings.slippage_bps,
                compound=settings.compound,
            ).backtest
            split = _build_master_split_metrics(
                data=data,
                master=candidate_master,
                settings=settings,
            )
            return result, split

        analyzed = apply_strategy(
            data=add_default_moving_averages(data),
            strategy=strategy,
            fast_period=settings.fast_period,
            slow_period=settings.slow_period,
            rsi_period=settings.rsi_period,
            rsi_oversold=settings.rsi_oversold,
            rsi_overbought=settings.rsi_overbought,
            bollinger_period=settings.bollinger_period,
            bollinger_std=settings.bollinger_std,
            master_storage=self.master_storage,
        )
        trade_mode = trade_mode_for_strategy(strategy)
        result = run_long_only_signal_backtest(
            data=analyzed,
            initial_equity=settings.initial_capital,
            fee_bps=settings.fee_bps,
            slippage_bps=settings.slippage_bps,
            compound=settings.compound,
            trade_mode=trade_mode,
        )
        split = self.build_split_metrics(
            data=data,
            strategy=strategy,
            settings=settings,
            trade_mode=trade_mode,
        )
        return result, split

    def _dataset_fingerprint(
        self,
        exchange: str,
        symbol: str,
        timeframes: list[str],
    ) -> tuple[object, ...]:
        parts: list[object] = []
        for timeframe in timeframes:
            path = self.storage.get_path(exchange, symbol, timeframe)
            if path.exists():
                stat = path.stat()
                parts.append((timeframe, stat.st_mtime_ns, stat.st_size))
            else:
                parts.append((timeframe, None, 0))
        return tuple(parts)

    def _get_cached(
        self,
        cache_key: tuple[object, ...],
        fingerprint: tuple[object, ...],
    ) -> object | None:
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return copy.deepcopy(cached[1])
        disk_value = self._get_disk_cached(cache_key, fingerprint)
        if disk_value is not None:
            self._cache[cache_key] = (fingerprint, copy.deepcopy(disk_value))
            return disk_value
        return None

    def _set_cached(
        self,
        cache_key: tuple[object, ...],
        fingerprint: tuple[object, ...],
        value: object,
    ) -> None:
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[cache_key] = (fingerprint, copy.deepcopy(value))
        self._set_disk_cached(cache_key, fingerprint, value)

    def _cache_path(self, cache_key: tuple[object, ...]) -> Path:
        raw_key = json.dumps(_json_safe(cache_key), sort_keys=True)
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _get_disk_cached(
        self,
        cache_key: tuple[object, ...],
        fingerprint: tuple[object, ...],
    ) -> object | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("fingerprint") != _json_safe(fingerprint):
            return None
        return payload.get("value")

    def _set_disk_cached(
        self,
        cache_key: tuple[object, ...],
        fingerprint: tuple[object, ...],
        value: object,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": _json_safe(fingerprint),
            "value": _json_safe(value),
        }
        self._cache_path(cache_key).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def _empty_timeframe_row(timeframe: str, status: str) -> dict[str, str | int]:
    return {
        "timeframe": timeframe,
        "status": status,
        "rows": 0,
        "from": "-",
        "to": "-",
        "strategy_gain": "-",
        "buy_hold": "-",
        "alpha": "-",
        "drawdown": "-",
        "trades": "-",
        "profit_factor": "-",
        "validation_gain": "-",
        "validation_alpha": "-",
        "validation_profit_factor": "-",
        "verdict": "-",
    }


def _empty_split_metrics() -> dict[str, str | int]:
    return {
        "train_rows": 0,
        "validation_rows": 0,
        "validation_from": "-",
        "train_return_pct": "0.00%",
        "train_drawdown_pct": "0.00%",
        "train_profit_factor": "0.00",
        "validation_return_pct": "0.00%",
        "validation_buy_and_hold_pct": "0.00%",
        "validation_alpha_pct": "0.00%",
        "validation_drawdown_pct": "0.00%",
        "validation_profit_factor": "0.00",
        "validation_return_drawdown_ratio": "0.00",
        "validation_verdict": "No data",
    }


def _build_master_split_metrics(
    data: pd.DataFrame,
    master: MasterConfig,
    settings: BacktestAnalysisSettings,
) -> dict[str, str | int]:
    if data.empty:
        return _empty_split_metrics()

    validation_rows = max(1, int(len(data) * settings.validation_pct / 100))
    split_index = max(1, len(data) - validation_rows)
    train_data = data.iloc[:split_index].reset_index(drop=True)
    validation_data = data.iloc[split_index:].reset_index(drop=True)
    train_result = run_master_system_backtest(
        data=train_data,
        master=master,
        initial_equity=settings.initial_capital,
        fee_bps=settings.fee_bps,
        slippage_bps=settings.slippage_bps,
        compound=settings.compound,
    ).backtest
    validation_result = run_master_system_backtest(
        data=validation_data,
        master=master,
        initial_equity=settings.initial_capital,
        fee_bps=settings.fee_bps,
        slippage_bps=settings.slippage_bps,
        compound=settings.compound,
    ).backtest
    validation_alpha = (
        validation_result.total_return_pct
        - validation_result.buy_and_hold_return_pct
    )

    return {
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "validation_from": validation_data.iloc[0]["datetime"]
        if not validation_data.empty
        else "-",
        "train_return_pct": _format_pct(train_result.total_return_pct),
        "train_drawdown_pct": _format_pct(train_result.max_drawdown_pct),
        "train_profit_factor": _format_ratio(train_result.profit_factor),
        "validation_return_pct": _format_pct(validation_result.total_return_pct),
        "validation_buy_and_hold_pct": _format_pct(
            validation_result.buy_and_hold_return_pct
        ),
        "validation_alpha_pct": _format_pct(validation_alpha),
        "validation_drawdown_pct": _format_pct(validation_result.max_drawdown_pct),
        "validation_profit_factor": _format_ratio(validation_result.profit_factor),
        "validation_return_drawdown_ratio": _format_ratio(
            _calculate_return_drawdown_ratio(
                validation_result.total_return_pct,
                validation_result.max_drawdown_pct,
            )
        ),
        "validation_verdict": _build_validation_verdict(
            validation_return_pct=validation_result.total_return_pct,
            validation_alpha_pct=validation_alpha,
            validation_profit_factor=validation_result.profit_factor,
        ),
    }


def _build_exposure_metrics(data: pd.DataFrame, result: BacktestResult) -> dict[str, str]:
    if data.empty or result.trade_log.empty:
        return {"exposure_pct": "0.00%", "avg_trade_days": "0.00"}

    start = pd.to_datetime(data.iloc[0]["datetime"])
    end = pd.to_datetime(data.iloc[-1]["datetime"])
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return {"exposure_pct": "0.00%", "avg_trade_days": "0.00"}

    entries = pd.to_datetime(result.trade_log["entry_datetime"])
    exits = pd.to_datetime(result.trade_log["exit_datetime"])
    durations = (exits - entries).dt.total_seconds().clip(lower=0)
    exposure_pct = float(durations.sum() / total_seconds * 100)
    avg_trade_days = float(durations.mean() / 86_400) if not durations.empty else 0.0

    return {
        "exposure_pct": _format_pct(exposure_pct),
        "avg_trade_days": f"{avg_trade_days:.2f}",
    }


def _filter_by_date_range(
    data: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    result = data.copy()
    if start_date.strip():
        start = _parse_date_filter(start_date, "start_date")
        result = cast(pd.DataFrame, result[result["datetime"] >= start])
    if end_date.strip():
        end = _parse_date_filter(end_date, "end_date")
        result = cast(pd.DataFrame, result[result["datetime"] <= end])

    return cast(pd.DataFrame, result.reset_index(drop=True))


def _parse_date_filter(value: str, field_name: str):
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} must be a valid date")

    return parsed


def _strategy_sort_key(row: dict[str, str | int]) -> float:
    value = str(row["validation_alpha"]).replace("%", "")
    try:
        return float(value)
    except ValueError:
        return -999999.0


def _parse_pct(value: object) -> float | None:
    try:
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _calculate_return_drawdown_ratio(return_pct: float, drawdown_pct: float) -> float:
    if drawdown_pct == 0:
        return 0.0

    return return_pct / abs(drawdown_pct)


def _build_validation_verdict(
    validation_return_pct: float,
    validation_alpha_pct: float,
    validation_profit_factor: float,
) -> str:
    if (
        validation_return_pct > 0
        and validation_alpha_pct > 0
        and validation_profit_factor > 1
    ):
        return "Promising"
    if validation_alpha_pct > 0:
        return "Defensive"
    if validation_return_pct > 0:
        return "Profitable but weak"

    return "Weak validation"


def _format_ratio(value: float) -> str:
    if value == float("inf"):
        return "inf"

    return f"{value:.2f}"


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _json_safe(value: object) -> object:
    if isinstance(value, BacktestAnalysisSettings):
        return {
            "start_date": value.start_date,
            "end_date": value.end_date,
            "fast_period": value.fast_period,
            "slow_period": value.slow_period,
            "rsi_period": value.rsi_period,
            "rsi_oversold": value.rsi_oversold,
            "rsi_overbought": value.rsi_overbought,
            "bollinger_period": value.bollinger_period,
            "bollinger_std": value.bollinger_std,
            "initial_capital": value.initial_capital,
            "fee_bps": value.fee_bps,
            "slippage_bps": value.slippage_bps,
            "compound": value.compound,
            "validation_pct": value.validation_pct,
        }
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
