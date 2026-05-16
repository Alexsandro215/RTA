import pandas as pd

from app.indicators.momentum import add_rsi
from app.indicators.moving_averages import add_ema


STRATEGY_METADATA = {
    "name": "Bull Trend Rider 4H Max V3",
    "market": "crypto",
    "symbols": ["BTC/USDT"],
    "timeframes": ["4h"],
    "trade_mode": "long_only",
    "notes": "BTC 4h trend rider. Slightly earlier entry than Max, same defensive EMA exit.",
}


def apply_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Bull Trend Rider 4H Max V3.

    BTC/USDT 4h specialized variant.

    Changes vs Max:
    - Earlier breakout entry: 54 candles instead of 72.
    - Keeps EMA 50 > EMA 200 bull regime filter.
    - Requires close above EMA 200 to avoid weak recoveries.
    - Keeps RSI heat filter <= 72/75 zone behavior by using <= 72 first,
      but entry result is intentionally equivalent in recent BTC 4h samples
      when RSI is not the limiting factor.
    - Keeps EMA 140 exit, which was the best balance between staying in trend
      and not giving too much back.
    """
    timeframe_minutes = _infer_timeframe_minutes(data)
    if not 230 <= timeframe_minutes <= 250:
        return _no_trade(data, "Bull Trend Rider 4H Max V3 - BTC 4h Only")

    result = _with_indicators(data)
    breakout_line = result["high"].rolling(54).max().shift(1)
    exit_line = result["low"].rolling(140).min().shift(1)

    buy_condition = (
        (result["close"] > breakout_line)
        & (result["ema_50"] > result["ema_200"])
        & (result["close"] > result["ema_200"])
        & (result["rsi_14"] <= 72)
    )
    sell_condition = (
        (result["close"] < exit_line)
        | (result["close"] < result["ema_140"])
        | (result["ema_50"] < result["ema_200"])
    )

    result["signal"] = _build_position_aware_signals(
        buy_condition=buy_condition,
        sell_condition=sell_condition,
    )
    result["strategy"] = "Bull Trend Rider 4H Max V3"
    return result


def _with_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for period in [50, 140, 200]:
        column = f"ema_{period}"
        if column not in result.columns:
            result = add_ema(result, period=period)

    if "rsi_14" not in result.columns:
        result = add_rsi(result, period=14)

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
