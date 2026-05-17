from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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

    The routed module allocation is used as the active stake size. For example,
    a main module with 70% allocation can only deploy 70% of current equity on
    entry, while the rest remains in cash.
    """
    analyzed = _prepare_system_indicators(data)
    analyzed = _add_regime_flags(analyzed)
    analyzed = _apply_module_signals(analyzed, master)
    analyzed = _route_master_signals(analyzed, master)
    backtest = run_allocated_master_signal_backtest(
        data=analyzed,
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        compound=compound,
    )
    return SystemBacktestResult(analyzed_data=analyzed, backtest=backtest)


def run_allocated_master_signal_backtest(
    data: pd.DataFrame,
    initial_equity: float = 1.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    compound: bool = True,
) -> BacktestResult:
    if data.empty or "signal" not in data.columns:
        return run_long_only_signal_backtest(
            data=data,
            initial_equity=initial_equity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            compound=compound,
        )

    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    cash = initial_equity
    units = 0.0
    entry_price: float | None = None
    entry_datetime = None
    entry_value = 0.0
    entry_module = "none"
    entry_reason = ""
    entry_allocation = 0.0
    equity_records: list[dict[str, object]] = []
    trade_records: list[dict[str, object]] = []

    for row in data.itertuples(index=False):
        signal = int(getattr(row, "signal"))
        close = float(getattr(row, "close"))
        datetime = getattr(row, "datetime")
        module = str(getattr(row, "active_module", "main"))
        reason = str(getattr(row, "signal_reason", ""))
        allocation_pct = float(getattr(row, "module_allocation_pct", 100.0))
        mark_price = close * (1 - slippage_rate)
        equity = cash + units * mark_price

        if units > 0 and signal == -1:
            exit_price = close * (1 - slippage_rate)
            gross_exit_value = units * exit_price
            exit_value = gross_exit_value * (1 - fee_rate)
            cash += exit_value
            trade_return = exit_value / entry_value if entry_value > 0 else 1.0
            trade_pl = exit_value - entry_value
            equity_after = cash
            trade_records.append(
                {
                    "entry_datetime": entry_datetime,
                    "exit_datetime": datetime,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": (trade_return - 1) * 100,
                    "pl_usd": trade_pl,
                    "equity_after": equity_after,
                    "module": entry_module,
                    "reason": entry_reason,
                    "allocation_pct": entry_allocation,
                }
            )
            units = 0.0
            entry_price = None
            entry_datetime = None
            entry_value = 0.0
            entry_module = "none"
            entry_reason = ""
            entry_allocation = 0.0
            equity = cash

        if units == 0 and signal == 1:
            base_equity = equity if compound else initial_equity
            target_value = max(0.0, base_equity * min(allocation_pct, 100.0) / 100)
            entry_value = min(cash, target_value)
            if entry_value > 0:
                entry_price = close * (1 + slippage_rate)
                invest_after_fee = entry_value * (1 - fee_rate)
                units = invest_after_fee / entry_price
                cash -= entry_value
                entry_datetime = datetime
                entry_module = module
                entry_reason = reason
                entry_allocation = allocation_pct
                equity = cash + units * close

        mark_value = units * close * (1 - slippage_rate)
        equity_records.append({"datetime": datetime, "equity": cash + mark_value})

    if units > 0 and not data.empty:
        last = data.iloc[-1]
        exit_price = float(last["close"]) * (1 - slippage_rate)
        exit_value = units * exit_price * (1 - fee_rate)
        cash += exit_value
        trade_return = exit_value / entry_value if entry_value > 0 else 1.0
        trade_records.append(
            {
                "entry_datetime": entry_datetime,
                "exit_datetime": last["datetime"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (trade_return - 1) * 100,
                "pl_usd": exit_value - entry_value,
                "equity_after": cash,
                "module": entry_module,
                "reason": entry_reason,
                "allocation_pct": entry_allocation,
            }
        )
        if equity_records:
            equity_records[-1]["equity"] = cash

    equity_frame = pd.DataFrame(equity_records)
    trade_log = pd.DataFrame(trade_records)
    final_equity = cash
    total_return_pct = (final_equity / initial_equity - 1) * 100
    buy_and_hold_return_pct = _buy_and_hold_return_pct(data, fee_rate, slippage_rate)
    max_drawdown_pct = _max_drawdown_pct(
        equity_frame["equity"].tolist() if not equity_frame.empty else [initial_equity]
    )
    stats = _trade_stats(trade_log)
    trades = len(trade_log)
    winning_trades = int((trade_log["pl_usd"] > 0).sum()) if trades else 0
    losing_trades = int((trade_log["pl_usd"] < 0).sum()) if trades else 0

    return BacktestResult(
        total_return_pct=total_return_pct,
        buy_and_hold_return_pct=buy_and_hold_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        trades=trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=(winning_trades / trades * 100) if trades else 0.0,
        profit_factor=stats["profit_factor"],
        best_trade_pct=stats["best_trade_pct"],
        worst_trade_pct=stats["worst_trade_pct"],
        average_trade_pct=stats["average_trade_pct"],
        max_winning_streak=int(stats["max_winning_streak"]),
        max_losing_streak=int(stats["max_losing_streak"]),
        final_equity=final_equity,
        equity_curve=equity_frame,
        trade_log=trade_log,
    )


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


def _buy_and_hold_return_pct(
    data: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    if data.empty:
        return 0.0
    first_close = float(data.iloc[0]["close"]) * (1 + slippage_rate)
    last_close = float(data.iloc[-1]["close"]) * (1 - slippage_rate)
    if first_close <= 0:
        return 0.0
    return ((last_close / first_close) * (1 - fee_rate) ** 2 - 1) * 100


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, (equity / peak - 1) * 100)
    return max_drawdown


def _trade_stats(trade_log: pd.DataFrame) -> dict[str, float | int]:
    if trade_log.empty:
        return {
            "profit_factor": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "average_trade_pct": 0.0,
            "max_winning_streak": 0,
            "max_losing_streak": 0,
        }

    gross_profit = trade_log.loc[trade_log["pl_usd"] > 0, "pl_usd"].sum()
    gross_loss = abs(trade_log.loc[trade_log["pl_usd"] < 0, "pl_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    pl_usd = cast(pd.Series, trade_log["pl_usd"])
    return_pct = cast(pd.Series, trade_log["return_pct"])
    max_winning_streak, max_losing_streak = _streaks(pl_usd)
    return {
        "profit_factor": float(profit_factor),
        "best_trade_pct": float(return_pct.max()),
        "worst_trade_pct": float(return_pct.min()),
        "average_trade_pct": float(return_pct.mean()),
        "max_winning_streak": max_winning_streak,
        "max_losing_streak": max_losing_streak,
    }


def _streaks(trade_pl_series: pd.Series) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    wins = 0
    losses = 0
    for trade_pl in trade_pl_series:
        if trade_pl > 0:
            wins += 1
            losses = 0
        elif trade_pl < 0:
            losses += 1
            wins = 0
        else:
            wins = 0
            losses = 0
        max_wins = max(max_wins, wins)
        max_losses = max(max_losses, losses)
    return max_wins, max_losses
