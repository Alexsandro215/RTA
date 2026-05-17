from dataclasses import dataclass
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    total_return_pct: float
    buy_and_hold_return_pct: float
    max_drawdown_pct: float
    trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    best_trade_pct: float
    worst_trade_pct: float
    average_trade_pct: float
    max_winning_streak: int
    max_losing_streak: int
    final_equity: float
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame


def run_long_only_signal_backtest(
    data: pd.DataFrame,
    initial_equity: float = 1.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    compound: bool = True,
    trade_mode: str = "long_only",
) -> BacktestResult:
    """Run a simple signal backtest.

    long_only: signal=1 opens long and signal=-1 closes long.
    short_only: signal=-1 opens short and signal=1 closes short.
    """
    if "signal" not in data.columns:
        return _empty_result(data, initial_equity, fee_bps, slippage_bps)
    if fee_bps < 0:
        raise ValueError("fee_bps must be greater than or equal to 0")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be greater than or equal to 0")
    if trade_mode not in {"long_only", "short_only", "bidirectional"}:
        raise ValueError("trade_mode must be 'long_only', 'short_only' or 'bidirectional'")

    equity = initial_equity
    entry_price: float | None = None
    entry_datetime: object | None = None
    entry_equity: float | None = None
    current_side: int = 0  # 1 for Long, -1 for Short, 0 for None
    trades = 0
    winning_trades = 0
    losing_trades = 0
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    equity_curve: list[float] = [equity]
    equity_records: list[dict[str, float | object]] = []
    trade_records: list[dict[str, float | object]] = []

    for row in data.itertuples(index=False):
        signal = int(getattr(row, "signal"))
        close = float(getattr(row, "close"))
        datetime = getattr(row, "datetime")

        # --- EXIT LOGIC ---
        should_exit = False
        if entry_price is not None:
            if trade_mode == "long_only" and signal == -1:
                should_exit = True
            elif trade_mode == "short_only" and signal == 1:
                should_exit = True
            elif trade_mode == "bidirectional" and signal != 0 and signal != current_side:
                should_exit = True

        if should_exit:
            assert entry_price is not None
            exit_price = _apply_exit_slippage(
                close,
                slippage_rate,
                "short_only" if current_side == -1 else "long_only",
            )
            gross_trade_return = _calculate_gross_trade_return(
                entry_price=entry_price,
                exit_price=exit_price,
                trade_mode="short_only" if current_side == -1 else "long_only",
            )
            net_trade_return = gross_trade_return * (1 - fee_rate) ** 2
            trade_pl = (entry_equity or initial_equity) * (net_trade_return - 1)
            
            if compound:
                equity *= gross_trade_return * (1 - fee_rate)
            else:
                equity += trade_pl
                
            equity_curve.append(equity)
            trades += 1
            if net_trade_return > 1:
                winning_trades += 1
            else:
                losing_trades += 1

            trade_records.append(
                _build_trade_record(
                    entry_datetime=entry_datetime,
                    exit_datetime=datetime,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    trade_return=net_trade_return,
                    trade_pl=trade_pl,
                    equity_after=equity,
                )
            )
            entry_price = None
            entry_datetime = None
            entry_equity = None
            current_side = 0

        # --- ENTRY LOGIC ---
        should_enter = False
        entry_side = 0
        if entry_price is None:
            if trade_mode == "long_only" and signal == 1:
                should_enter = True
                entry_side = 1
            elif trade_mode == "short_only" and signal == -1:
                should_enter = True
                entry_side = -1
            elif trade_mode == "bidirectional" and signal != 0:
                should_enter = True
                entry_side = signal

        if should_enter:
            current_side = entry_side
            entry_price = _apply_entry_slippage(
                close,
                slippage_rate,
                "short_only" if current_side == -1 else "long_only",
            )
            entry_datetime = datetime
            entry_equity = equity if compound else initial_equity
            if compound:
                equity *= 1 - fee_rate
            equity_curve.append(equity)

        # --- MARK TO MARKET ---
        mark_to_market_equity = _calculate_mark_to_market_equity(
            equity=equity,
            entry_price=entry_price,
            entry_equity=entry_equity,
            current_close=close,
            initial_equity=initial_equity,
            compound=compound,
            trade_mode="short_only" if current_side == -1 else "long_only",
        )
        equity_records.append({"datetime": datetime, "equity": mark_to_market_equity})

    # Close open position at end
    if entry_price is not None and not data.empty:
        last_close = float(data.iloc[-1]["close"])
        last_datetime = data.iloc[-1]["datetime"]
        exit_price = _apply_exit_slippage(
            last_close,
            slippage_rate,
            "short_only" if current_side == -1 else "long_only",
        )
        gross_trade_return = _calculate_gross_trade_return(
            entry_price=entry_price,
            exit_price=exit_price,
            trade_mode="short_only" if current_side == -1 else "long_only",
        )
        net_trade_return = gross_trade_return * (1 - fee_rate) ** 2
        trade_pl = (entry_equity or initial_equity) * (net_trade_return - 1)
        
        if compound:
            equity *= gross_trade_return * (1 - fee_rate)
        else:
            equity += trade_pl
        
        trades += 1
        if net_trade_return > 1:
            winning_trades += 1
        else:
            losing_trades += 1
        
        if equity_records:
            equity_records[-1]["equity"] = equity
        trade_records.append(
            _build_trade_record(
                entry_datetime=entry_datetime,
                exit_datetime=last_datetime,
                entry_price=entry_price,
                exit_price=exit_price,
                trade_return=net_trade_return,
                trade_pl=trade_pl,
                equity_after=equity,
            )
        )

    total_return_pct = (equity / initial_equity - 1) * 100
    equity_frame = pd.DataFrame(equity_records)
    trade_log = pd.DataFrame(trade_records)
    buy_and_hold_return_pct = _calculate_buy_and_hold_return_pct(
        data=data,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    max_drawdown_pct = _calculate_max_drawdown_pct(
        equity_frame["equity"].tolist() if not equity_frame.empty else equity_curve
    )
    win_rate_pct = (winning_trades / trades * 100) if trades else 0.0
    trade_stats = _calculate_trade_stats(trade_log)

    return BacktestResult(
        total_return_pct=total_return_pct,
        buy_and_hold_return_pct=buy_and_hold_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        trades=trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=trade_stats["profit_factor"],
        best_trade_pct=trade_stats["best_trade_pct"],
        worst_trade_pct=trade_stats["worst_trade_pct"],
        average_trade_pct=trade_stats["average_trade_pct"],
        max_winning_streak=int(trade_stats["max_winning_streak"]),
        max_losing_streak=int(trade_stats["max_losing_streak"]),
        final_equity=equity,
        equity_curve=equity_frame,
        trade_log=trade_log,
    )


def _empty_result(
    data: pd.DataFrame,
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
) -> BacktestResult:
    fee_rate = fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    return BacktestResult(
        total_return_pct=0.0,
        buy_and_hold_return_pct=_calculate_buy_and_hold_return_pct(
            data,
            fee_rate,
            slippage_rate,
        ),
        max_drawdown_pct=0.0,
        trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate_pct=0.0,
        profit_factor=0.0,
        best_trade_pct=0.0,
        worst_trade_pct=0.0,
        average_trade_pct=0.0,
        max_winning_streak=0,
        max_losing_streak=0,
        final_equity=initial_equity,
        equity_curve=_build_flat_equity_curve(data, initial_equity),
        trade_log=_build_empty_trade_log(),
    )


def _calculate_trade_stats(trade_log: pd.DataFrame) -> dict[str, float | int]:
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
    max_winning_streak, max_losing_streak = _calculate_streaks(pl_usd)

    return {
        "profit_factor": float(profit_factor),
        "best_trade_pct": float(return_pct.max()),
        "worst_trade_pct": float(return_pct.min()),
        "average_trade_pct": float(return_pct.mean()),
        "max_winning_streak": max_winning_streak,
        "max_losing_streak": max_losing_streak,
    }


def _calculate_streaks(trade_pl_series: pd.Series) -> tuple[int, int]:
    max_winning_streak = 0
    max_losing_streak = 0
    current_winning_streak = 0
    current_losing_streak = 0

    for trade_pl in trade_pl_series:
        if trade_pl > 0:
            current_winning_streak += 1
            current_losing_streak = 0
        elif trade_pl < 0:
            current_losing_streak += 1
            current_winning_streak = 0
        else:
            current_winning_streak = 0
            current_losing_streak = 0

        max_winning_streak = max(max_winning_streak, current_winning_streak)
        max_losing_streak = max(max_losing_streak, current_losing_streak)

    return max_winning_streak, max_losing_streak


def _apply_buy_slippage(price: float, slippage_rate: float) -> float:
    return price * (1 + slippage_rate)


def _apply_sell_slippage(price: float, slippage_rate: float) -> float:
    return price * (1 - slippage_rate)


def _apply_entry_slippage(
    price: float,
    slippage_rate: float,
    trade_mode: str,
) -> float:
    if trade_mode == "short_only":
        return _apply_sell_slippage(price, slippage_rate)

    return _apply_buy_slippage(price, slippage_rate)


def _apply_exit_slippage(
    price: float,
    slippage_rate: float,
    trade_mode: str,
) -> float:
    if trade_mode == "short_only":
        return _apply_buy_slippage(price, slippage_rate)

    return _apply_sell_slippage(price, slippage_rate)


def _calculate_gross_trade_return(
    entry_price: float,
    exit_price: float,
    trade_mode: str,
) -> float:
    if trade_mode == "short_only":
        return entry_price / exit_price

    return exit_price / entry_price


def _is_entry_signal(
    signal: int,
    entry_price: float | None,
    trade_mode: str,
) -> bool:
    if entry_price is not None:
        return False
    if trade_mode == "short_only":
        return signal == -1

    return signal == 1


def _is_exit_signal(
    signal: int,
    entry_price: float | None,
    trade_mode: str,
) -> bool:
    if entry_price is None:
        return False
    if trade_mode == "short_only":
        return signal == 1

    return signal == -1


def _build_trade_record(
    entry_datetime: object | None,
    exit_datetime: object,
    entry_price: float,
    exit_price: float,
    trade_return: float,
    trade_pl: float,
    equity_after: float,
) -> dict[str, float | object]:
    return {
        "entry_datetime": entry_datetime,
        "exit_datetime": exit_datetime,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": (trade_return - 1) * 100,
        "pl_usd": trade_pl,
        "equity_after": equity_after,
    }


def _build_empty_trade_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entry_datetime",
            "exit_datetime",
            "entry_price",
            "exit_price",
            "return_pct",
            "pl_usd",
            "equity_after",
        ]
    )


def _calculate_mark_to_market_equity(
    equity: float,
    entry_price: float | None,
    entry_equity: float | None,
    current_close: float,
    initial_equity: float,
    compound: bool,
    trade_mode: str,
) -> float:
    if entry_price is None:
        return equity

    gross_trade_return = _calculate_gross_trade_return(
        entry_price=entry_price,
        exit_price=current_close,
        trade_mode=trade_mode,
    )
    if compound:
        return equity * gross_trade_return

    stake = entry_equity or initial_equity
    return equity + stake * (gross_trade_return - 1)


def _build_flat_equity_curve(
    data: pd.DataFrame,
    initial_equity: float,
) -> pd.DataFrame:
    if data.empty or "datetime" not in data.columns:
        return pd.DataFrame(columns=["datetime", "equity"])

    return pd.DataFrame(
        {
            "datetime": data["datetime"],
            "equity": initial_equity,
        }
    )


def _calculate_buy_and_hold_return_pct(
    data: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    if data.empty:
        return 0.0

    first_close = _apply_buy_slippage(float(data.iloc[0]["close"]), slippage_rate)
    last_close = _apply_sell_slippage(float(data.iloc[-1]["close"]), slippage_rate)
    if first_close <= 0:
        return 0.0

    return ((last_close / first_close) * (1 - fee_rate) ** 2 - 1) * 100


def _calculate_max_drawdown_pct(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_drawdown = 0.0

    for equity in equity_curve:
        peak = max(peak, equity)
        if peak <= 0:
            continue

        drawdown = (equity / peak - 1) * 100
        max_drawdown = min(max_drawdown, drawdown)

    return max_drawdown
