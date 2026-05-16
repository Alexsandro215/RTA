# Bull Trend Rider Examples

Upload these files manually from TestBack.

- `bull_trend_rider_guarded.py`: balanced version. Keeps intraday behavior and improves high-timeframe exits.
- `bull_trend_rider_focus.py`: stricter version. Trades only 4h, 1d, and 3d; skips weaker timeframes.
- `bull_trend_rider_4h_plus.py`: dedicated 4h version. More selective entries and faster exits than Guarded.
- `bull_trend_rider_4h_max.py`: more aggressive dedicated 4h version. EMA 50/200 trend with EMA 140 exit.
- `bull_trend_rider_4h_max_v2.py`: experimental Max variant. Earlier 60-candle breakout with stricter RSI heat filter.

Both files expose `apply_strategy(data)` and return `signal` values:

- `1`: open long
- `-1`: close long
- `0`: wait
