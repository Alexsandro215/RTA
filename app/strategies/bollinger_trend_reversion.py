import pandas as pd

from app.indicators.momentum import add_rsi
from app.indicators.moving_averages import add_ema
from app.indicators.volatility import add_bollinger_bands


def apply_bollinger_trend_reversion_strategy(
    data: pd.DataFrame,
    period: int = 40,
    std_dev: float = 2.2,
    rsi_period: int = 14,
    max_entry_rsi: float = 50.0,
    trend_ema_period: int = 200,
) -> pd.DataFrame:
    """
    Buy Bollinger lower-band recoveries only when the market is above the trend EMA.

    This is a stricter version of the plain Bollinger mean-reversion strategy:
    it waits for price to recover above the lower band, avoids longs below EMA 200,
    and exits on upper-band recovery or trend loss.
    """
    result = data.copy()
    lower_column = f"bb_lower_{period}"
    upper_column = f"bb_upper_{period}"
    ema_column = f"ema_{trend_ema_period}"
    rsi_column = f"rsi_{rsi_period}"

    if lower_column not in result.columns or upper_column not in result.columns:
        result = add_bollinger_bands(result, period=period, std_dev=std_dev)
    if ema_column not in result.columns:
        result = add_ema(result, period=trend_ema_period)
    if rsi_column not in result.columns:
        result = add_rsi(result, period=rsi_period)

    was_below_lower = result["close"].shift(1) < result[lower_column].shift(1)
    reenters_above_lower = (result["close"] >= result[lower_column]) & was_below_lower
    is_above_trend = result["close"] > result[ema_column]
    is_not_overextended = result[rsi_column] <= max_entry_rsi
    buy_condition = reenters_above_lower & is_above_trend & is_not_overextended

    was_above_upper = result["close"].shift(1) > result[upper_column].shift(1)
    reenters_below_upper = (result["close"] <= result[upper_column]) & was_above_upper
    loses_trend = result["close"] < result[ema_column]
    sell_condition = reenters_below_upper | loses_trend

    result["signal"] = _build_position_aware_signals(buy_condition, sell_condition)
    result["strategy"] = f"Bollinger Trend {period}/{std_dev:g}"
    return result


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
