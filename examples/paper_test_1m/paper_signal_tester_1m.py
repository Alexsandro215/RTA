from __future__ import annotations

import pandas as pd


def apply_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Paper trading smoke-test strategy for 1m candles.

    It is intentionally simple and not meant to be profitable:
    - BUY near minutes divisible by 6.
    - SELL near minutes divisible by 6 plus 3.

    Use it only to verify that IRT/paper trading opens and closes fictional
    positions automatically.
    """
    result = data.copy()
    datetimes = pd.to_datetime(result["datetime"], utc=True, errors="coerce")
    minute = datetimes.dt.minute
    buy_condition = minute.mod(6).eq(0)
    sell_condition = minute.mod(6).eq(3)

    signals: list[int] = []
    has_position = False
    for should_buy, should_sell in zip(buy_condition.fillna(False), sell_condition.fillna(False)):
        if not has_position and should_buy:
            signals.append(1)
            has_position = True
        elif has_position and should_sell:
            signals.append(-1)
            has_position = False
        else:
            signals.append(0)

    result["signal"] = signals
    result["strategy"] = "Paper Signal Tester 1m"
    return result
