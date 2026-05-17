import pandas as pd
import plotly.graph_objects as go


def build_candlestick_chart(
    data,
    symbol: str,
    timeframe: str,
    show_sma_20: bool,
    show_sma_50: bool,
    show_ema_200: bool,
    strategy: str,
    fast_period: int,
    slow_period: int,
    bollinger_period: int,
) -> str:
    plot_data = _with_chart_gaps(data, timeframe)
    if len(data) > 3000:
        figure = go.Figure(
            data=[
                go.Scattergl(
                    x=plot_data["datetime"],
                    y=plot_data["close"],
                    mode="lines",
                    name=f"{symbol} close",
                    line={"color": "#0f766e", "width": 1.5},
                    connectgaps=False,
                )
            ]
        )
        chart_title = f"{symbol} - {timeframe} close history"
    else:
        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=data["datetime"],
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name=symbol,
                    increasing_line_color="#16a34a",
                    decreasing_line_color="#dc2626",
                )
            ]
        )
        chart_title = f"{symbol} - {timeframe} candles"

    if show_sma_20:
        _add_indicator_line(figure, plot_data, "sma_20", "SMA 20", "#2563eb")
    if show_sma_50:
        _add_indicator_line(figure, plot_data, "sma_50", "SMA 50", "#f59e0b")
    if show_ema_200:
        _add_indicator_line(figure, plot_data, "ema_200", "EMA 200", "#7c3aed")
    if strategy == "ema_crossover":
        _add_indicator_line(
            figure,
            plot_data,
            f"ema_{fast_period}",
            f"EMA {fast_period}",
            "#0891b2",
        )
        _add_indicator_line(
            figure,
            plot_data,
            f"ema_{slow_period}",
            f"EMA {slow_period}",
            "#be123c",
        )
    if strategy == "ema_trend_capture":
        _add_indicator_line(figure, plot_data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, plot_data, "ema_300", "EMA 300", "#be123c")
    if strategy == "ema_donchian_trend":
        _add_indicator_line(figure, plot_data, "ema_50", "EMA 50", "#0891b2")
        _add_indicator_line(figure, plot_data, "ema_200", "EMA 200", "#be123c")
    if strategy == "adaptive_trend_guard":
        _add_indicator_line(figure, plot_data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, plot_data, "ema_300", "EMA 300", "#be123c")
    if strategy == "bull_trend_rider":
        _add_indicator_line(figure, plot_data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, plot_data, "ema_300", "EMA 300", "#be123c")
    if strategy == "bear_trend_rider":
        _add_indicator_line(figure, plot_data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, plot_data, "ema_300", "EMA 300", "#be123c")
    if strategy == "panic_dip_buyer":
        _add_indicator_line(figure, plot_data, "ema_50", "EMA 50", "#0891b2")
        _add_indicator_line(figure, plot_data, "ema_300", "EMA 300", "#be123c")
        _add_indicator_line(figure, plot_data, "bb_lower_40", "BB Lower", "#64748b")
        _add_indicator_line(figure, plot_data, "bb_middle_40", "BB Middle", "#94a3b8")
    if strategy == "ultimate_hybrid":
        _add_indicator_line(
            figure,
            plot_data,
            f"ema_{fast_period}",
            f"EMA {fast_period}",
            "#0891b2",
        )
        _add_indicator_line(
            figure,
            plot_data,
            f"ema_{slow_period}",
            f"EMA {slow_period}",
            "#be123c",
        )
        _add_indicator_line(
            figure,
            plot_data,
            f"bb_upper_{bollinger_period}",
            "BB Upper",
            "#64748b",
        )
        _add_indicator_line(
            figure,
            plot_data,
            f"bb_lower_{bollinger_period}",
            "BB Lower",
            "#64748b",
        )
    if strategy in {"bollinger_mean_reversion", "bollinger_trend_reversion"}:
        _add_indicator_line(
            figure,
            plot_data,
            f"bb_upper_{bollinger_period}",
            "BB Upper",
            "#64748b",
        )
        _add_indicator_line(
            figure,
            plot_data,
            f"bb_middle_{bollinger_period}",
            "BB Middle",
            "#94a3b8",
        )
        _add_indicator_line(
            figure,
            plot_data,
            f"bb_lower_{bollinger_period}",
            "BB Lower",
            "#64748b",
        )
    if strategy != "none":
        _add_strategy_markers(figure, data)

    figure.update_layout(
        title=chart_title,
        template="plotly_white",
        height=680,
        margin={"l": 48, "r": 24, "t": 64, "b": 40},
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )


def build_equity_curve_chart(equity_curve, strategy_name: str) -> str:
    if equity_curve.empty:
        return ""

    figure = go.Figure(
        data=[
            go.Scattergl(
                x=equity_curve["datetime"],
                y=equity_curve["equity"],
                mode="lines",
                name="Equity",
                line={"color": "#0f766e", "width": 2},
            )
        ]
    )
    figure.update_layout(
        title=f"Equity Curve - {strategy_name}",
        template="plotly_white",
        height=360,
        margin={"l": 48, "r": 24, "t": 56, "b": 40},
        xaxis_title="Datetime",
        yaxis_title="Equity USD",
    )
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )


def _add_strategy_markers(figure: go.Figure, data) -> None:
    if "signal" not in data.columns:
        return

    buys = data[data["signal"] == 1]
    sells = data[data["signal"] == -1]

    if not buys.empty:
        figure.add_trace(
            go.Scattergl(
                x=buys["datetime"],
                y=buys["close"],
                mode="markers",
                name="BUY",
                marker={
                    "symbol": "triangle-up",
                    "size": 13,
                    "color": "#16a34a",
                    "line": {"color": "#ffffff", "width": 1},
                },
                text=buys["close"],
                hovertemplate="BUY<br>%{x}<br>close=%{y}<extra></extra>",
            )
        )

    if not sells.empty:
        figure.add_trace(
            go.Scattergl(
                x=sells["datetime"],
                y=sells["close"],
                mode="markers",
                name="SELL",
                marker={
                    "symbol": "triangle-down",
                    "size": 13,
                    "color": "#dc2626",
                    "line": {"color": "#ffffff", "width": 1},
                },
                text=sells["close"],
                hovertemplate="SELL<br>%{x}<br>close=%{y}<extra></extra>",
            )
        )


def _add_indicator_line(
    figure: go.Figure,
    data,
    column: str,
    name: str,
    color: str,
) -> None:
    if column not in data.columns or data[column].dropna().empty:
        return

    figure.add_trace(
        go.Scattergl(
            x=data["datetime"],
            y=data[column],
            mode="lines",
            name=name,
            line={"color": color, "width": 1.6},
            connectgaps=False,
        )
    )


def _with_chart_gaps(data, timeframe: str):
    if data.empty or "datetime" not in data.columns:
        return data

    expected_gap = _timeframe_delta(timeframe)
    if expected_gap is None:
        return data

    result = data.copy()
    datetimes = pd.to_datetime(result["datetime"], utc=True, errors="coerce")
    gap_mask = datetimes.diff() > expected_gap * 3
    if not gap_mask.any():
        return result

    columns_to_break = [
        column
        for column in result.columns
        if column not in {"timestamp", "datetime", "signal", "strategy"}
        and pd.api.types.is_numeric_dtype(result[column])
    ]
    result.loc[gap_mask, columns_to_break] = pd.NA
    return result


def _timeframe_delta(timeframe: str):
    value = timeframe.strip().lower()
    unit = value[-1:]
    try:
        amount = int(value[:-1])
    except ValueError:
        return None

    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    if unit == "w":
        return pd.Timedelta(weeks=amount)

    return None
