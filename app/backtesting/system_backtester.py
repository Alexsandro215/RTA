from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.backtesting.simple_backtester import BacktestResult, run_long_only_signal_backtest
from app.indicators.momentum import add_rsi
from app.indicators.moving_averages import add_default_moving_averages, add_ema
from app.indicators.volatility import add_bollinger_bands
from app.systems.master_config import MasterConfig


@dataclass(frozen=True)
class SystemBacktestResult:
    analyzed_data: pd.DataFrame
    backtest: BacktestResult


def run_master_system_backtest(
    data: pd.DataFrame,
    master: MasterConfig,
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
    compound: bool,
) -> SystemBacktestResult:
    """Run a first-pass master system backtest.

    v1 routing:
    - Main module controls normal exposure.
    - Rebound module can replace main when panic/reclaim flags are active.
    - Sideways module can replace main when sideways flag is active.
    - Defensive module/cash blocks long exposure when bear/breakdown flags are active.

    Allocation is tracked as metadata for now; the simple backtester still uses one
    active signal stream. This keeps the system test deterministic while we prepare
    a multi-allocation equity engine.
    """
    analyzed = _prepare_system_indicators(data)
    analyzed = _add_regime_flags(analyzed)
    analyzed = _apply_module_signals(analyzed, master)
    analyzed = _route_master_signals(analyzed, master)
    backtest = run_long_only_signal_backtest(
        data=analyzed,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        compound=compound,
    )
    return SystemBacktestResult(analyzed_data=analyzed, backtest=backtest)


def _prepare_system_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = add_default_moving_averages(data)
    for period in [50, 80, 300]:
        column = f"ema_{period}"
        if column not in result.columns:
            result = add_ema(result, period=period)
    result = add_bollinger_bands(result, period=40, std_dev=2.0)
    result = add_rsi(result, period=14)
    return result


def _add_regime_flags(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["flag_bull_regime"] = result["ema_80"] > result["ema_300"]
    result["flag_bear_regime"] = result["ema_80"] < result["ema_300"]

    ema_spread = ((result["ema_80"] - result["ema_300"]).abs() / result["close"]).fillna(0)
    recent_range = (
        (result["high"].rolling(40).max() - result["low"].rolling(40).min())
        / result["close"]
    ).fillna(0)
    result["flag_sideways_regime"] = (ema_spread < 0.03) & (recent_range < 0.20)

    result["flag_panic_flush"] = (
        (result["close"].shift(1) < result["bb_lower_40"].shift(1))
        | (result["low"].shift(1) < result["bb_lower_40"].shift(1))
    )
    result["flag_base_reclaim"] = result["close"] >= result["bb_lower_40"]
    result["flag_breakdown_guard"] = result["close"] < result["ema_300"]
    return result


def _apply_module_signals(data: pd.DataFrame, master: MasterConfig) -> pd.DataFrame:
    from app.web.routes import _apply_strategy

    result = data.copy()
    module_specs = {
        "main": master.main_strategy,
        "rebound": master.rebound_strategy,
        "sideways": master.sideways_strategy,
        "defensive": master.defensive_strategy,
    }
    for module_name, strategy_name in module_specs.items():
        module_data = _apply_strategy(
            data=result,
            strategy=strategy_name,
            fast_period=20,
            slow_period=50,
            rsi_period=14,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            bollinger_period=40,
            bollinger_std=2.0,
        )
        result[f"{module_name}_signal"] = module_data["signal"]

    return result


def _route_master_signals(data: pd.DataFrame, master: MasterConfig) -> pd.DataFrame:
    result = data.copy()
    signals: list[int] = []
    active_modules: list[str] = []
    reasons: list[str] = []
    allocation: list[float] = []
    has_position = False

    for row in result.itertuples(index=False):
        defensive_active = _defensive_is_active(row, master)
        rebound_active = _rebound_is_active(row, master)
        sideways_active = _sideways_is_active(row, master)

        module = "main"
        reason = "main strategy"
        module_allocation = master.main_allocation_pct
        raw_signal = int(getattr(row, "main_signal"))

        if defensive_active:
            module = "defensive"
            reason = "defensive flags active"
            module_allocation = master.defensive_cash_pct
            raw_signal = 0 if master.defensive_strategy == "none" else int(getattr(row, "defensive_signal"))
        elif rebound_active:
            module = "rebound"
            reason = "panic flush + base reclaim"
            module_allocation = master.rebound_allocation_pct
            raw_signal = int(getattr(row, "rebound_signal"))
        elif sideways_active:
            module = "sideways"
            reason = "sideways regime"
            module_allocation = master.sideways_allocation_pct
            raw_signal = int(getattr(row, "sideways_signal"))

        signal = 0
        if not has_position and raw_signal == 1:
            signal = 1
            has_position = True
        elif has_position and (raw_signal == -1 or defensive_active):
            signal = -1
            has_position = False

        signals.append(signal)
        active_modules.append(module)
        reasons.append(reason)
        allocation.append(module_allocation)

    result["signal"] = signals
    result["active_module"] = active_modules
    result["signal_reason"] = reasons
    result["module_allocation_pct"] = allocation
    result["strategy"] = f"Master System: {master.name}"
    return result


def _rebound_is_active(row, master: MasterConfig) -> bool:
    flags = set(master.rebound_flags)
    checks = {
        "panic_flush": bool(getattr(row, "flag_panic_flush")),
        "base_reclaim": bool(getattr(row, "flag_base_reclaim")),
    }
    return bool(flags) and all(checks[name] for name in flags)


def _sideways_is_active(row, master: MasterConfig) -> bool:
    flags = set(master.sideways_flags)
    checks = {"sideways_regime": bool(getattr(row, "flag_sideways_regime"))}
    return bool(flags) and all(checks[name] for name in flags)


def _defensive_is_active(row, master: MasterConfig) -> bool:
    flags = set(master.defensive_flags)
    checks = {
        "bear_regime": bool(getattr(row, "flag_bear_regime")),
        "breakdown_guard": bool(getattr(row, "flag_breakdown_guard")),
    }
    return bool(flags) and all(checks[name] for name in flags)
