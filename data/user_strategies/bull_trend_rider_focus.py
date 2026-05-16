import pandas as pd

from app.indicators.moving_averages import add_ema


def apply_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Bull Trend Rider Focus.

    This is a stricter version of Bull Trend Rider Guarded. It only trades the
    timeframes that looked useful in TestBack:
    - 4h: main execution timeframe.
    - 1d / 3d: slower confirmation style trades.

    It intentionally does not trade 15m, 30m, 1h, 12h, or 1w because those
    timeframes either lost in validation or had poor evidence.
    """
    timeframe_minutes = _infer_timeframe_minutes(data)

    if 230 <= timeframe_minutes <= 250:
        return _apply_rider(
            data=data,
            breakout_period=48,
            exit_period=160,
            exit_ema_period=200,
            label="Bull Trend Rider Focus 4h",
        )

    if 23 * 60 <= timeframe_minutes <= 25 * 60:
        return _apply_rider(
            data=data,
            breakout_period=48,
            exit_period=120,
            exit_ema_period=120,
            label="Bull Trend Rider Focus 1d",
        )

    if 3 * 23 * 60 <= timeframe_minutes <= 3 * 25 * 60:
        return _apply_rider(
            data=data,
            breakout_period=48,
            exit_period=160,
            exit_ema_period=120,
            trailing_stop_pct=0.80,
            label="Bull Trend Rider Focus 3d",
        )

    return _no_trade(data, "Bull Trend Rider Focus - No Trade Timeframe")


def _apply_rider(
    data: pd.DataFrame,
    breakout_period: int,
    exit_period: int,
    exit_ema_period: int,
    label: str,
    trailing_stop_pct: float = 0.0,
    fast_period: int = 80,
    slow_period: int = 300,
) -> pd.DataFrame:
    result = _with_emas(data, [fast_period, slow_period, exit_ema_period])
    fast_column = f"ema_{fast_period}"
    slow_column = f"ema_{slow_period}"
    exit_ema_column = f"ema_{exit_ema_period}"

    breakout_line = result["high"].rolling(breakout_period).max().shift(1)
    exit_line = result["low"].rolling(exit_period).min().shift(1)

    buy_condition = (
        (result["close"] > breakout_line)
        & (result[fast_column] > result[slow_column])
    )
    sell_condition = (
        (result["close"] < exit_line)
        | (result["close"] < result[exit_ema_column])
        | (result[fast_column] < result[slow_column])
    )

    if trailing_stop_pct > 0:
        rolling_high = result["high"].rolling(exit_period).max().shift(1)
        sell_condition = sell_condition | (
            result["close"] < rolling_high * trailing_stop_pct
        )

    result["signal"] = _build_position_aware_signals(
        buy_condition=buy_condition,
        sell_condition=sell_condition,
    )
    result["strategy"] = label
    return result


def _with_emas(data: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    result = data.copy()
    for period in sorted(set(periods)):
        column = f"ema_{period}"
        if column not in result.columns:
            result = add_ema(result, period=period)
    return result


def _infer_timeframe_minutes(data: pd.DataFrame) -> float:
    if len(data) < 2 or "timestamp" not in data.columns:
        return 60.0

    spacing_ms = data["timestamp"].diff().dropna().median()
    if pd.isna(spacing_ms) or spacing_ms <= 0:
        return 60.0

    return float(spacing_ms) / 60_000


def _no_trade(data: pd.DataFrame, label: str) -> pd.DataFrame:
    result = data.copy()
    result["signal"] = 0
    result["strategy"] = label
    return result


def _build_position_aware_signals(
    buy_condition: pd.Series,
    sell_condition: pd.Series,
) -> list[int]:
    signals = []
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
