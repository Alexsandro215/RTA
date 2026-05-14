import pandas as pd

from app.indicators.momentum import add_rsi
from app.indicators.moving_averages import add_ema
from app.indicators.volatility import add_bollinger_bands


def apply_ema_trend_capture_strategy(
    data: pd.DataFrame,
    fast_period: int = 80,
    slow_period: int = 300,
) -> pd.DataFrame:
    """Capture long BTC trends with a slower EMA crossover."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("EMA periods must be greater than 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be lower than slow_period")

    result = data.copy()
    fast_column = f"ema_{fast_period}"
    slow_column = f"ema_{slow_period}"

    if fast_column not in result.columns:
        result = add_ema(result, period=fast_period)
    if slow_column not in result.columns:
        result = add_ema(result, period=slow_period)

    fast_above_slow = result[fast_column] > result[slow_column]
    previous_fast_above_slow = fast_above_slow.shift(1)

    result["signal"] = 0
    result.loc[fast_above_slow & (previous_fast_above_slow == False), "signal"] = 1
    result.loc[(~fast_above_slow) & (previous_fast_above_slow == True), "signal"] = -1
    result["strategy"] = f"EMA Trend Capture {fast_period}/{slow_period}"
    return result


def apply_ema_donchian_trend_strategy(
    data: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200,
    breakout_period: int = 24,
    exit_period: int = 72,
) -> pd.DataFrame:
    """Enter breakouts only when EMA trend is aligned, then exit on trend loss or channel break."""
    if min(fast_period, slow_period, breakout_period, exit_period) <= 0:
        raise ValueError("periods must be greater than 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be lower than slow_period")

    result = data.copy()
    fast_column = f"ema_{fast_period}"
    slow_column = f"ema_{slow_period}"
    breakout_column = f"donchian_high_{breakout_period}"
    exit_column = f"donchian_low_{exit_period}"

    if fast_column not in result.columns:
        result = add_ema(result, period=fast_period)
    if slow_column not in result.columns:
        result = add_ema(result, period=slow_period)

    result[breakout_column] = result["high"].rolling(breakout_period).max().shift(1)
    result[exit_column] = result["low"].rolling(exit_period).min().shift(1)

    trend_is_up = result[fast_column] > result[slow_column]
    breakout = result["close"] > result[breakout_column]
    channel_exit = result["close"] < result[exit_column]
    trend_exit = result[fast_column] < result[slow_column]

    result["signal"] = _build_position_aware_signals(
        buy_condition=breakout & trend_is_up,
        sell_condition=channel_exit | trend_exit,
    )
    result["strategy"] = "EMA Donchian Trend 50/200 24/72"
    return result


def apply_adaptive_trend_guard_strategy(
    data: pd.DataFrame,
    fast_period: int = 80,
    slow_period: int = 300,
    breakout_period: int = 24,
    exit_period: int = 96,
) -> pd.DataFrame:
    """
    Follow slower confirmed trends with a Donchian breakout/exit guard.

    Entry:
    - EMA fast above EMA slow.
    - Close breaks above the previous Donchian high.

    Exit:
    - Close breaks below the Donchian exit channel.
    - EMA fast loses EMA slow.
    """
    if min(fast_period, slow_period, breakout_period, exit_period) <= 0:
        raise ValueError("periods must be greater than 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be lower than slow_period")

    result = data.copy()
    fast_column = f"ema_{fast_period}"
    slow_column = f"ema_{slow_period}"
    breakout_column = f"donchian_high_{breakout_period}"
    exit_column = f"donchian_low_{exit_period}"

    if fast_column not in result.columns:
        result = add_ema(result, period=fast_period)
    if slow_column not in result.columns:
        result = add_ema(result, period=slow_period)

    result[breakout_column] = result["high"].rolling(breakout_period).max().shift(1)
    result[exit_column] = result["low"].rolling(exit_period).min().shift(1)

    trend_is_up = result[fast_column] > result[slow_column]
    breakout = result["close"] > result[breakout_column]
    trend_exit = result[fast_column] < result[slow_column]
    channel_exit = result["close"] < result[exit_column]

    result["signal"] = _build_position_aware_signals(
        buy_condition=breakout & trend_is_up,
        sell_condition=channel_exit | trend_exit,
    )
    result["strategy"] = "Adaptive Trend Guard 80/300 24/96"
    return result


def apply_bull_trend_rider_strategy(
    data: pd.DataFrame,
    fast_period: int = 80,
    slow_period: int = 300,
    breakout_period: int = 48,
    exit_period: int = 240,
) -> pd.DataFrame:
    """
    Ride long bull trends with a slower exit than Adaptive Trend Guard.

    This variant is designed for higher timeframes such as 4h, where the main
    failure mode is exiting too early during strong BTC expansions.
    """
    if min(fast_period, slow_period, breakout_period, exit_period) <= 0:
        raise ValueError("periods must be greater than 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be lower than slow_period")

    result = data.copy()
    fast_column = f"ema_{fast_period}"
    slow_column = f"ema_{slow_period}"
    breakout_column = f"donchian_high_{breakout_period}"
    exit_column = f"donchian_low_{exit_period}"

    if fast_column not in result.columns:
        result = add_ema(result, period=fast_period)
    if slow_column not in result.columns:
        result = add_ema(result, period=slow_period)

    result[breakout_column] = result["high"].rolling(breakout_period).max().shift(1)
    result[exit_column] = result["low"].rolling(exit_period).min().shift(1)

    buy_condition = (
        (result["close"] > result[breakout_column])
        & (result[fast_column] > result[slow_column])
    )
    sell_condition = result["close"] < result[exit_column]

    result["signal"] = _build_position_aware_signals(
        buy_condition=buy_condition,
        sell_condition=sell_condition,
    )
    result["strategy"] = "Bull Trend Rider 80/300 48/240"
    return result


def apply_bear_trend_rider_strategy(
    data: pd.DataFrame,
    fast_period: int = 80,
    slow_period: int = 300,
    breakdown_period: int = 48,
    exit_period: int = 240,
) -> pd.DataFrame:
    """
    Ride bearish trends with short signals.

    Signal semantics for the short backtester:
    - -1 opens a short when price breaks below the Donchian low in a downtrend.
    - 1 closes the short when price breaks above the Donchian high exit channel.
    """
    if min(fast_period, slow_period, breakdown_period, exit_period) <= 0:
        raise ValueError("periods must be greater than 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be lower than slow_period")

    result = data.copy()
    fast_column = f"ema_{fast_period}"
    slow_column = f"ema_{slow_period}"
    breakdown_column = f"donchian_low_{breakdown_period}"
    exit_column = f"donchian_high_{exit_period}"

    if fast_column not in result.columns:
        result = add_ema(result, period=fast_period)
    if slow_column not in result.columns:
        result = add_ema(result, period=slow_period)

    result[breakdown_column] = result["low"].rolling(breakdown_period).min().shift(1)
    result[exit_column] = result["high"].rolling(exit_period).max().shift(1)

    short_condition = (
        (result["close"] < result[breakdown_column])
        & (result[fast_column] < result[slow_column])
    )
    cover_condition = result["close"] > result[exit_column]

    result["signal"] = _build_short_signals(
        short_condition=short_condition,
        cover_condition=cover_condition,
    )
    result["strategy"] = "Bear Trend Rider 80/300 48/240"
    return result


def apply_panic_dip_buyer_strategy(
    data: pd.DataFrame,
    trend_period: int = 300,
    bollinger_period: int = 40,
    bollinger_std: float = 2.0,
    rsi_period: int = 14,
    max_entry_rsi: float = 40.0,
    exit_ema_period: int = 50,
) -> pd.DataFrame:
    """
    Buy confirmed bearish flush recoveries and exit on mean recovery.

    This is a long-only contrarian strategy. It looks for price stretched below
    the lower Bollinger band with oversold RSI, then waits for price to recover
    back above the lower band before entering. That avoids catching a falling
    knife while the flush is still expanding.
    """
    if min(trend_period, bollinger_period, rsi_period, exit_ema_period) <= 0:
        raise ValueError("periods must be greater than 0")
    if bollinger_std <= 0:
        raise ValueError("bollinger_std must be greater than 0")

    result = data.copy()
    trend_column = f"ema_{trend_period}"
    exit_ema_column = f"ema_{exit_ema_period}"
    lower_column = f"bb_lower_{bollinger_period}"
    middle_column = f"bb_middle_{bollinger_period}"
    rsi_column = f"rsi_{rsi_period}"

    if trend_column not in result.columns:
        result = add_ema(result, period=trend_period)
    if exit_ema_column not in result.columns:
        result = add_ema(result, period=exit_ema_period)
    if lower_column not in result.columns or middle_column not in result.columns:
        result = add_bollinger_bands(
            result,
            period=bollinger_period,
            std_dev=bollinger_std,
        )
    if rsi_column not in result.columns:
        result = add_rsi(result, period=rsi_period)

    previous_flush = (
        (result["close"].shift(1) < result[lower_column].shift(1))
        | (result["low"].shift(1) < result[lower_column].shift(1))
    )
    reclaims_lower_band = result["close"] >= result[lower_column]
    oversold = result[rsi_column] <= max_entry_rsi
    higher_timeframe_trend_is_up = result[exit_ema_column] > result[trend_column]
    buy_condition = (
        previous_flush
        & reclaims_lower_band
        & oversold
        & higher_timeframe_trend_is_up
    )

    exit_condition = (
        (result["close"] > result[exit_ema_column])
        | (result["close"] > result[middle_column])
        | (result["close"] < result[lower_column])
    )

    result["signal"] = _build_position_aware_signals(
        buy_condition=buy_condition,
        sell_condition=exit_condition,
    )
    result["strategy"] = "Panic Dip Buyer 40/2 RSI"
    return result


def _build_short_signals(
    short_condition: pd.Series,
    cover_condition: pd.Series,
) -> list[int]:
    signals: list[int] = []
    has_short = False

    for should_short, should_cover in zip(
        short_condition.fillna(False),
        cover_condition.fillna(False),
    ):
        if not has_short and should_short:
            signals.append(-1)
            has_short = True
        elif has_short and should_cover:
            signals.append(1)
            has_short = False
        else:
            signals.append(0)

    return signals


def _build_position_aware_signals(
    buy_condition: pd.Series,
    sell_condition: pd.Series,
) -> list[int]:
    signals: list[int] = []
    has_position = False

    for should_buy, should_sell in zip(
        buy_condition.fillna(False),
        sell_condition.fillna(False),
    ):
        if not has_position and should_buy:
            signals.append(1)
            has_position = True
        elif has_position and should_sell:
            signals.append(-1)
            has_position = False
        else:
            signals.append(0)

    return signals
