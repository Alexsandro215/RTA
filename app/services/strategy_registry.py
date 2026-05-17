import pandas as pd

from app.backtesting.system_backtester import run_master_system_backtest
from app.strategies.bollinger_mean_reversion import (
    apply_bollinger_mean_reversion_strategy,
)
from app.strategies.bollinger_trend_reversion import (
    apply_bollinger_trend_reversion_strategy,
)
from app.strategies.custom_loader import apply_custom_strategy, list_custom_strategies
from app.strategies.ema_crossover import apply_ema_crossover_strategy
from app.strategies.rsi_mean_reversion import apply_rsi_mean_reversion_strategy
from app.strategies.sma_crossover import apply_sma_crossover_strategy
from app.strategies.trend_capture import (
    apply_adaptive_trend_guard_strategy,
    apply_bear_trend_rider_strategy,
    apply_bull_trend_rider_strategy,
    apply_ema_donchian_trend_strategy,
    apply_ema_trend_capture_strategy,
    apply_panic_dip_buyer_strategy,
)
from app.strategies.ultimate_hybrid import apply_ultimate_hybrid_strategy
from app.systems.master_config import MasterConfigStorage


def strategy_options(master_storage: MasterConfigStorage) -> list[dict[str, str]]:
    options = module_strategy_options()
    options.extend(
        {"value": f"master:{master.name}", "label": f"Master: {master.name}"}
        for master in master_storage.list()
    )
    return options


def module_strategy_options() -> list[dict[str, str]]:
    options = base_strategy_options()
    options.extend(
        {"value": item.key, "label": f"Custom: {item.label}"}
        for item in list_custom_strategies()
    )
    return options


def base_strategy_options() -> list[dict[str, str]]:
    return [
        {"value": "none", "label": "None"},
        {"value": "sma_crossover", "label": "SMA 20/50 Crossover"},
        {"value": "ema_crossover", "label": "EMA 20/50 Crossover"},
        {"value": "rsi_mean_reversion", "label": "RSI 14 Mean Reversion"},
        {"value": "bollinger_mean_reversion", "label": "Bollinger Mean Reversion"},
        {"value": "bollinger_trend_reversion", "label": "Bollinger Trend Filtered"},
        {"value": "ema_trend_capture", "label": "EMA Trend Capture 80/300"},
        {"value": "ema_donchian_trend", "label": "EMA Donchian Trend 50/200"},
        {"value": "adaptive_trend_guard", "label": "Adaptive Trend Guard 80/300"},
        {"value": "bull_trend_rider", "label": "Bull Trend Rider 80/300"},
        {"value": "bear_trend_rider", "label": "Bear Trend Rider 80/300"},
        {"value": "panic_dip_buyer", "label": "Panic Dip Buyer"},
        {"value": "ultimate_hybrid", "label": "Ultimate Hybrid System"},
    ]


def apply_strategy(
    data: pd.DataFrame,
    strategy: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
    bollinger_period: int,
    bollinger_std: float,
    master_storage: MasterConfigStorage,
) -> pd.DataFrame:
    if strategy.startswith("master:"):
        master = master_storage.get(strategy.removeprefix("master:"))
        if master is not None:
            return run_master_system_backtest(
                data=data,
                master=master,
                initial_equity=1.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                compound=True,
            ).analyzed_data
    if strategy.startswith("custom:"):
        return apply_custom_strategy(data=data, key=strategy)
    if strategy == "sma_crossover":
        return apply_sma_crossover_strategy(
            data=data,
            fast_period=fast_period,
            slow_period=slow_period,
        )
    if strategy == "ema_crossover":
        return apply_ema_crossover_strategy(
            data=data,
            fast_period=fast_period,
            slow_period=slow_period,
        )
    if strategy == "rsi_mean_reversion":
        return apply_rsi_mean_reversion_strategy(
            data=data,
            period=rsi_period,
            oversold=rsi_oversold,
            overbought=rsi_overbought,
        )
    if strategy == "bollinger_mean_reversion":
        return apply_bollinger_mean_reversion_strategy(
            data=data,
            period=bollinger_period,
            std_dev=bollinger_std,
        )
    if strategy == "bollinger_trend_reversion":
        return apply_bollinger_trend_reversion_strategy(
            data=data,
            period=bollinger_period,
            std_dev=bollinger_std,
            rsi_period=rsi_period,
        )
    if strategy == "ema_trend_capture":
        return apply_ema_trend_capture_strategy(data=data)
    if strategy == "ema_donchian_trend":
        return apply_ema_donchian_trend_strategy(data=data)
    if strategy == "adaptive_trend_guard":
        return apply_adaptive_trend_guard_strategy(data=data)
    if strategy == "bull_trend_rider":
        return apply_bull_trend_rider_strategy(data=data)
    if strategy == "bear_trend_rider":
        return apply_bear_trend_rider_strategy(data=data)
    if strategy == "panic_dip_buyer":
        return apply_panic_dip_buyer_strategy(data=data)
    if strategy == "ultimate_hybrid":
        return apply_ultimate_hybrid_strategy(
            data=data,
            fast_ema=fast_period,
            slow_ema=slow_period,
            rsi_period=rsi_period,
            bb_period=bollinger_period,
            bb_std=bollinger_std,
        )

    result = data.copy()
    result["signal"] = 0
    result["strategy"] = "None"
    return result


def strategy_label(
    strategy: str,
    fast_period: int = 20,
    slow_period: int = 50,
    rsi_period: int = 14,
    bollinger_period: int = 40,
    bollinger_std: float = 2.2,
) -> str:
    if strategy.startswith("master:"):
        return f"Master: {strategy.removeprefix('master:')}"
    if strategy.startswith("custom:"):
        custom_strategy = next(
            (item for item in list_custom_strategies() if item.key == strategy),
            None,
        )
        if custom_strategy is not None:
            return custom_strategy.label
        return strategy.removeprefix("custom:").replace("_", " ").title()
    if strategy == "sma_crossover":
        return f"SMA {fast_period}/{slow_period} Crossover"
    if strategy == "ema_crossover":
        return f"EMA {fast_period}/{slow_period} Crossover"
    if strategy == "rsi_mean_reversion":
        return f"RSI {rsi_period} Mean Reversion"
    if strategy == "bollinger_mean_reversion":
        return f"Bollinger {bollinger_period}/{bollinger_std:g} Mean Reversion"
    if strategy == "bollinger_trend_reversion":
        return f"Bollinger Trend {bollinger_period}/{bollinger_std:g}"
    if strategy == "ema_trend_capture":
        return "EMA Trend Capture 80/300"
    if strategy == "ema_donchian_trend":
        return "EMA Donchian Trend 50/200 24/72"
    if strategy == "adaptive_trend_guard":
        return "Adaptive Trend Guard 80/300 24/96"
    if strategy == "bull_trend_rider":
        return "Bull Trend Rider 80/300 48/240"
    if strategy == "bear_trend_rider":
        return "Bear Trend Rider 80/300 48/240"
    if strategy == "panic_dip_buyer":
        return "Panic Dip Buyer 40/2 RSI"
    if strategy == "ultimate_hybrid":
        return "Ultimate Hybrid System"

    return "None"


def trade_mode_for_strategy(strategy: str) -> str:
    if strategy == "bear_trend_rider":
        return "short_only"
    if strategy == "ultimate_hybrid":
        return "bidirectional"

    return "long_only"
