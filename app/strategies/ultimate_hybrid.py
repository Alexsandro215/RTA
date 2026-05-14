import pandas as pd
import numpy as np

from app.indicators.momentum import add_rsi
from app.indicators.moving_averages import add_ema
from app.indicators.volatility import add_bollinger_bands


def apply_ultimate_hybrid_strategy(
    data: pd.DataFrame,
    fast_ema: int = 50,
    slow_ema: int = 200,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    trend_threshold_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Ultimate Hybrid System:
    1. Principal: Mean Reversion (Bollinger + RSI) for sideways markets.
    2. Secondary: Trend Following (EMA Cross + Donchian) for trending markets.
    3. Tertiary: Panic Recovery (Oversold Reclaim) for crashes.
    
    Supports bidirectional signals (1 for Long, -1 for Short).
    """
    result = data.copy()
    
    # 1. Add Indicators
    result = add_ema(result, period=fast_ema)
    result = add_ema(result, period=slow_ema)
    result = add_ema(result, period=20) # Signal EMA
    result = add_rsi(result, period=rsi_period)
    result = add_bollinger_bands(result, period=bb_period, std_dev=bb_std)
    
    f_ema = f"ema_{fast_ema}"
    s_ema = f"ema_{slow_ema}"
    rsi_col = f"rsi_{rsi_period}"
    bb_low = f"bb_lower_{bb_period}"
    bb_high = f"bb_upper_{bb_period}"
    bb_mid = f"bb_middle_{bb_period}"
    
    # 2. Regime Detection
    # Trend strength: distance between EMAs as % of price
    ema_diff_pct = (result[f_ema] - result[s_ema]).abs() / result["close"] * 100
    is_trending = ema_diff_pct > trend_threshold_pct
    is_bullish = (result[f_ema] > result[s_ema]) & is_trending
    is_bearish = (result[f_ema] < result[s_ema]) & is_trending
    is_sideways = ~is_trending
    
    # 3. Signals Generation
    signals = np.zeros(len(result))
    current_pos = 0 # 0: None, 1: Long, -1: Short
    
    # Pre-calculate conditions
    rsi = result[rsi_col]
    close = result["close"]
    
    # Donchian for Trend
    donchian_h = result["high"].rolling(20).max().shift(1)
    donchian_l = result["low"].rolling(20).min().shift(1)
    
    for i in range(1, len(result)):
        # --- TERTIARY: CRASH RECOVERY (Priority) ---
        # Long entry on oversold reclaim
        if rsi.iloc[i-1] < 30 and close.iloc[i] > result[bb_low].iloc[i] and current_pos <= 0:
            signals[i] = 1
            current_pos = 1
            continue
            
        # Short entry on overbought rejection
        if rsi.iloc[i-1] > 70 and close.iloc[i] < result[bb_high].iloc[i] and current_pos >= 0:
            signals[i] = -1
            current_pos = -1
            continue

        # --- SECONDARY: TREND FOLLOWING ---
        if is_bullish.iloc[i]:
            # Enter Long on Donchian Breakout
            if close.iloc[i] > donchian_h.iloc[i] and current_pos <= 0:
                signals[i] = 1
                current_pos = 1
            # Exit Long on Trend Loss
            elif close.iloc[i] < result[f_ema].iloc[i] and current_pos == 1:
                signals[i] = -1
                current_pos = 0
        
        elif is_bearish.iloc[i]:
            # Enter Short on Donchian Breakdown
            if close.iloc[i] < donchian_l.iloc[i] and current_pos >= 0:
                signals[i] = -1
                current_pos = -1
            # Exit Short on Trend Loss
            elif close.iloc[i] > result[f_ema].iloc[i] and current_pos == -1:
                signals[i] = 1
                current_pos = 0
                
        # --- PRINCIPAL: MEAN REVERSION (Sideways) ---
        elif is_sideways.iloc[i]:
            # Long: Price < BB Low + RSI < 40
            if close.iloc[i] < result[bb_low].iloc[i] and rsi.iloc[i] < 40 and current_pos <= 0:
                signals[i] = 1
                current_pos = 1
            # Short: Price > BB High + RSI > 60
            elif close.iloc[i] > result[bb_high].iloc[i] and rsi.iloc[i] > 60 and current_pos >= 0:
                signals[i] = -1
                current_pos = -1
            # Neutral Exit: Touch Middle Band
            elif current_pos == 1 and close.iloc[i] > result[bb_mid].iloc[i]:
                signals[i] = -1
                current_pos = 0
            elif current_pos == -1 and close.iloc[i] < result[bb_mid].iloc[i]:
                signals[i] = 1
                current_pos = 0

    result["signal"] = signals
    result["strategy"] = "Ultimate Hybrid System"
    return result
