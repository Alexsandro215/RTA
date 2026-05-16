from __future__ import annotations

import pandas as pd


def apply_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    BTC 1m Momentum Pulse.

    Long-only minute strategy for paper trading experiments:
    - Enters only when short momentum agrees with the 1m trend.
    - Avoids weak volume and very flat candles.
    - Exits with take-profit, stop-loss, trend failure, or time stop.

    This is an experiment, not a promise of profitability.
    """
    result = data.copy()
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["volume"].astype(float)

    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    ema_800 = close.ewm(span=800, adjust=False).mean()
    rsi = _rsi(close, period=14)
    atr_pct = _atr_pct(high=high, low=low, close=close, period=14)
    volume_ma = volume.rolling(30).mean()
    prior_high = high.rolling(45).max().shift(1)

    trend_ok = (
        (ema_9 > ema_21)
        & (ema_21 > ema_55)
        & (ema_55 > ema_200)
        & (ema_200 > ema_800)
        & (close > ema_200)
    )
    breakout = close > prior_high
    momentum_ok = rsi.between(54, 74)
    volume_ok = volume > (volume_ma * 1.05)
    volatility_ok = atr_pct.between(0.00018, 0.0045)
    buy_condition = trend_ok & breakout & momentum_ok & volume_ok & volatility_ok

    signals: list[int] = []
    has_position = False
    entry_price = 0.0
    bars_in_trade = 0
    highest_close = 0.0

    for i, should_buy in enumerate(buy_condition.fillna(False)):
        price = float(close.iloc[i])
        current_ema_21 = float(ema_21.iloc[i]) if not pd.isna(ema_21.iloc[i]) else price
        current_rsi = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50.0

        if not has_position and should_buy:
            signals.append(1)
            has_position = True
            entry_price = price
            highest_close = price
            bars_in_trade = 0
            continue

        if has_position:
            bars_in_trade += 1
            highest_close = max(highest_close, price)
            pnl_pct = price / entry_price - 1 if entry_price > 0 else 0.0
            trail_pct = price / highest_close - 1 if highest_close > 0 else 0.0
            take_profit = pnl_pct >= 0.008
            stop_loss = pnl_pct <= -0.0028
            trail_stop = bars_in_trade >= 8 and trail_pct <= -0.0025
            trend_fail = price < current_ema_21 and current_rsi < 48
            time_stop = bars_in_trade >= 45 and pnl_pct < 0.002

            if take_profit or stop_loss or trail_stop or trend_fail or time_stop:
                signals.append(-1)
                has_position = False
                entry_price = 0.0
                bars_in_trade = 0
                highest_close = 0.0
                continue

        signals.append(0)

    result["signal"] = signals
    result["strategy"] = "BTC 1m Momentum Pulse"
    return result


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr / close
