from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.backtesting.simple_backtester import run_long_only_signal_backtest
from app.backtesting.system_backtester import run_master_system_backtest
from app.data.ccxt_provider import CCXTMarketDataProvider
from app.data.csv_storage import CsvMarketDataStorage
from app.data.market_data_provider import MarketDataError
from app.indicators.moving_averages import add_default_moving_averages
from app.strategies.bollinger_mean_reversion import (
    apply_bollinger_mean_reversion_strategy,
)
from app.strategies.bollinger_trend_reversion import (
    apply_bollinger_trend_reversion_strategy,
)
from app.strategies.ema_crossover import apply_ema_crossover_strategy
from app.strategies.rsi_mean_reversion import apply_rsi_mean_reversion_strategy
from app.strategies.snip_hedge import (
    SnipHedgeConfig,
    preview_snip_hedge,
)
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
from app.systems.master_config import MasterConfig, MasterConfigStorage


router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")
storage = CsvMarketDataStorage()
provider = CCXTMarketDataProvider()
master_storage = MasterConfigStorage()

SUPPORTED_EXCHANGES = ["binance", "coinbase", "kraken", "okx", "bybit"]
SUPPORTED_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
]
SUPPORTED_TIMEFRAMES = ["15m", "30m", "1h", "4h", "12h", "1d", "3d", "1w"]


@router.get("/home", response_class=HTMLResponse)
def home(
    request: Request,
    masters: str = Query(default="BTC/USDT"),
    new_master: str = Query(default=""),
    selected_master: str = Query(default="BTC/USDT"),
    master_name: str = Query(default="BTC Core"),
    save_master: str = Query(default="false"),
    trend_follow: str = Query(default="true"),
    rebound_watch: str = Query(default="true"),
    sideways_reversion: str = Query(default="false"),
    defensive_cash: str = Query(default="true"),
    bull_flag: str = Query(default="true"),
    bear_flag: str = Query(default="true"),
    sideways_flag: str = Query(default="true"),
    panic_flush_flag: str = Query(default="true"),
    reclaim_flag: str = Query(default="true"),
    breakdown_flag: str = Query(default="true"),
    main_strategy: str = Query(default="adaptive_trend_guard"),
    main_timeframe: str = Query(default="1h"),
    secondary_rebound_strategy: str = Query(default="panic_dip_buyer"),
    secondary_sideways_strategy: str = Query(default="bollinger_mean_reversion"),
    secondary_defensive_strategy: str = Query(default="none"),
    main_allocation_pct: float = Query(default=70.0, ge=0.0, le=100.0),
    rebound_allocation_pct: float = Query(default=20.0, ge=0.0, le=100.0),
    sideways_allocation_pct: float = Query(default=15.0, ge=0.0, le=100.0),
    defensive_cash_pct: float = Query(default=100.0, ge=0.0, le=100.0),
) -> HTMLResponse:
    master_list = _parse_master_list(masters)
    if new_master.strip():
        candidate = new_master.strip().upper()
        if "/" not in candidate:
            candidate = f"{candidate}/USDT"
        if candidate not in master_list:
            master_list.append(candidate)
        selected_master = candidate

    if selected_master not in master_list:
        selected_master = master_list[0]

    enabled_modules = {
        "trend_follow": _is_enabled(trend_follow),
        "rebound_watch": _is_enabled(rebound_watch),
        "sideways_reversion": _is_enabled(sideways_reversion),
        "defensive_cash": _is_enabled(defensive_cash),
    }
    enabled_flags = {
        "bull_flag": _is_enabled(bull_flag),
        "bear_flag": _is_enabled(bear_flag),
        "sideways_flag": _is_enabled(sideways_flag),
        "panic_flush_flag": _is_enabled(panic_flush_flag),
        "reclaim_flag": _is_enabled(reclaim_flag),
        "breakdown_flag": _is_enabled(breakdown_flag),
    }

    saved_message = ""
    if _is_enabled(save_master):
        config = MasterConfig(
            name=master_name.strip() or selected_master,
            symbol=selected_master,
            main_strategy=main_strategy,
            main_timeframe=main_timeframe,
            main_allocation_pct=main_allocation_pct,
            rebound_strategy=secondary_rebound_strategy,
            rebound_allocation_pct=rebound_allocation_pct,
            rebound_flags=_active_flag_names(
                {
                    "panic_flush": enabled_flags["panic_flush_flag"],
                    "base_reclaim": enabled_flags["reclaim_flag"],
                }
            ),
            sideways_strategy=secondary_sideways_strategy,
            sideways_allocation_pct=sideways_allocation_pct,
            sideways_flags=_active_flag_names(
                {"sideways_regime": enabled_flags["sideways_flag"]}
            ),
            defensive_strategy=secondary_defensive_strategy,
            defensive_cash_pct=defensive_cash_pct,
            defensive_flags=_active_flag_names(
                {
                    "bear_regime": enabled_flags["bear_flag"],
                    "breakdown_guard": enabled_flags["breakdown_flag"],
                }
            ),
        )
        master_storage.save(config)
        saved_message = f"Saved master '{config.name}'"

    master_rows = []
    for master in master_list:
        dataset = storage.load("binance", master, main_timeframe)
        master_rows.append(
            {
                "symbol": master,
                "safe_symbol": master.replace("/", "_"),
                "timeframe": main_timeframe,
                "rows": len(dataset),
                "from": dataset.iloc[0]["datetime"] if not dataset.empty else "-",
                "to": dataset.iloc[-1]["datetime"] if not dataset.empty else "-",
                "selected": master == selected_master,
            }
        )

    return templates.TemplateResponse(
        request,
        "home.html",
        context={
            "request": request,
            "masters": ",".join(master_list),
            "master_rows": master_rows,
            "selected_master": selected_master,
            "enabled_modules": enabled_modules,
            "enabled_flags": enabled_flags,
            "master_name": master_name,
            "saved_masters": master_storage.list(),
            "saved_message": saved_message,
            "main_strategy": main_strategy,
            "main_timeframe": main_timeframe,
            "secondary_rebound_strategy": secondary_rebound_strategy,
            "secondary_sideways_strategy": secondary_sideways_strategy,
            "secondary_defensive_strategy": secondary_defensive_strategy,
            "main_allocation_pct": main_allocation_pct,
            "rebound_allocation_pct": rebound_allocation_pct,
            "sideways_allocation_pct": sideways_allocation_pct,
            "defensive_cash_pct": defensive_cash_pct,
            "strategy_options": [
                {"value": "adaptive_trend_guard", "label": "Adaptive Trend Guard"},
                {"value": "bull_trend_rider", "label": "Bull Trend Rider"},
                {"value": "ema_donchian_trend", "label": "EMA Donchian Trend"},
                {"value": "panic_dip_buyer", "label": "Panic Dip Buyer"},
                {"value": "bollinger_mean_reversion", "label": "Bollinger Mean Reversion"},
                {"value": "bollinger_trend_reversion", "label": "Bollinger Trend Filtered"},
                {"value": "rsi_mean_reversion", "label": "RSI Mean Reversion"},
                {"value": "ultimate_hybrid", "label": "Ultimate Hybrid System"},
                {"value": "none", "label": "None / Cash"},
            ],
            "timeframe_options": SUPPORTED_TIMEFRAMES,
        },
    )


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTC/USDT"),
    timeframe: str = Query(default="1h"),
    master_config: str = Query(default=""),
    max_rows: int = Query(default=10000, ge=50, le=250000),
    fetch_history: str = Query(default="false"),
    fetch_all_history: str = Query(default="false"),
    compare_timeframes: str = Query(default="false"),
    fetch_since: str = Query(default="2024-01-01"),
    fetch_until: str = Query(default=""),
    fetch_batch_limit: int = Query(default=1000, ge=100, le=1500),
    fetch_max_batches: int = Query(default=0, ge=0, le=1000),
    validation_pct: float = Query(default=30.0, ge=5.0, le=80.0),
    show_sma_20: str = Query(default="true"),
    show_sma_50: str = Query(default="true"),
    show_ema_200: str = Query(default="true"),
    strategy: str = Query(default="none"),
    start_date: str = Query(default=""),
    end_date: str = Query(default=""),
    fast_period: int = Query(default=20, ge=1, le=500),
    slow_period: int = Query(default=50, ge=2, le=1000),
    rsi_period: int = Query(default=14, ge=2, le=200),
    rsi_oversold: float = Query(default=30.0, ge=0.0, le=100.0),
    rsi_overbought: float = Query(default=70.0, ge=0.0, le=100.0),
    bollinger_period: int = Query(default=40, ge=2, le=500),
    bollinger_std: float = Query(default=2.2, gt=0.0, le=10.0),
    fee_bps: float = Query(default=10.0, ge=0.0, le=1000.0),
    slippage_bps: float = Query(default=2.0, ge=0.0, le=1000.0),
    initial_capital: float = Query(default=1000.0, gt=0.0),
    position_mode: str = Query(default="compound"),
    snip_url: str = Query(default=""),
    snip_trigger_up_bid: float = Query(default=0.51, ge=0.0, le=1.0),
    snip_trigger_down_bid: float = Query(default=0.50, ge=0.0, le=1.0),
    snip_limit_up: float = Query(default=0.50, ge=0.0, le=1.0),
    snip_limit_down: float = Query(default=0.48, ge=0.0, le=1.0),
) -> HTMLResponse:
    selected_master_config = None
    if master_config.strip():
        selected_master_config = master_storage.get(master_config)
        if selected_master_config is not None:
            symbol = selected_master_config.symbol
            timeframe = selected_master_config.main_timeframe
            strategy = selected_master_config.main_strategy

    path = storage.get_path(exchange, symbol, timeframe)
    data_message = ""
    data_error = ""
    data_messages = []
    data_errors = []
    if _is_enabled(fetch_all_history):
        data_messages, data_errors = _download_all_listed_datasets(
            exchange=exchange,
            since=fetch_since,
            until=fetch_until or None,
            limit_per_request=fetch_batch_limit,
            max_batches=fetch_max_batches or None,
        )
        data_message = "Downloaded listed datasets." if data_messages else ""
        data_error = "Some datasets failed." if data_errors else ""
    elif _is_enabled(fetch_history):
        try:
            fetched_data = provider.fetch_ohlcv_history(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since=fetch_since,
                until=fetch_until or None,
                limit_per_request=fetch_batch_limit,
                max_batches=fetch_max_batches or None,
            )
            path = storage.save(
                data=fetched_data,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
            )
            saved_data = storage.load(exchange, symbol, timeframe)
            data_message = (
                f"Saved {len(saved_data)} candles for {exchange} {symbol} "
                f"{timeframe} into {path}"
            )
            data_messages = [data_message]
        except (MarketDataError, ValueError) as exc:
            data_error = str(exc)
            data_errors = [data_error]

    data = storage.load(exchange, symbol, timeframe)
    compound = position_mode != "fixed"
    indicators = {
        "show_sma_20": _is_enabled(show_sma_20),
        "show_sma_50": _is_enabled(show_sma_50),
        "show_ema_200": _is_enabled(show_ema_200),
    }

    chart_html = ""
    equity_chart_html = ""
    compare_results = []
    summary = {
        "path": str(path),
        "rows": 0,
        "from": None,
        "to": None,
        "shown_rows": 0,
        "exists": Path(path).exists(),
        "strategy": "None",
        "buy_signals": 0,
        "sell_signals": 0,
        "total_return_pct": "0.00%",
        "buy_and_hold_return_pct": "0.00%",
        "max_drawdown_pct": "0.00%",
        "trades": 0,
        "win_rate_pct": "0.00%",
        "profit_factor": "0.00",
        "best_trade_pct": "0.00%",
        "worst_trade_pct": "0.00%",
        "average_trade_pct": "0.00%",
        "max_winning_streak": 0,
        "max_losing_streak": 0,
        "fee_bps": f"{fee_bps:.2f}",
        "slippage_bps": f"{slippage_bps:.2f}",
        "initial_capital": _format_usd(initial_capital),
        "final_equity": _format_usd(initial_capital),
        "profit_loss": _format_usd(0.0),
        "position_mode": _position_mode_label(position_mode),
        "train_rows": 0,
        "validation_rows": 0,
        "validation_from": "-",
        "train_return_pct": "0.00%",
        "train_drawdown_pct": "0.00%",
        "train_profit_factor": "0.00",
        "validation_return_pct": "0.00%",
        "validation_buy_and_hold_pct": "0.00%",
        "validation_alpha_pct": "0.00%",
        "validation_drawdown_pct": "0.00%",
        "validation_profit_factor": "0.00",
        "alpha_vs_buy_hold_pct": "0.00%",
        "return_drawdown_ratio": "0.00",
        "validation_return_drawdown_ratio": "0.00",
        "validation_verdict": "No strategy",
    }
    trade_log = []
    snip_preview = None
    snip_error = ""

    if snip_url.strip():
        try:
            snip_preview = _format_snip_preview(
                preview_snip_hedge(
                    url_or_slug=snip_url,
                    config=SnipHedgeConfig(
                        trigger_up_bid=snip_trigger_up_bid,
                        trigger_down_bid=snip_trigger_down_bid,
                        limit_up=snip_limit_up,
                        limit_down=snip_limit_down,
                    ),
                )
            )
        except Exception as exc:
            snip_error = str(exc)

    if not data.empty:
        filtered_data = _filter_by_date_range(data, start_date, end_date)
        if selected_master_config is not None:
            system_result = run_master_system_backtest(
                data=filtered_data,
                master=selected_master_config,
                initial_equity=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
            )
            analyzed_data = system_result.analyzed_data
            backtest_result = system_result.backtest
        else:
            analyzed_data = add_default_moving_averages(filtered_data)
            analyzed_data = _apply_strategy(
                data=analyzed_data,
                strategy=strategy,
                fast_period=fast_period,
                slow_period=slow_period,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                bollinger_period=bollinger_period,
                bollinger_std=bollinger_std,
            )
            backtest_result = run_long_only_signal_backtest(
                data=analyzed_data,
                initial_equity=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
                trade_mode=_trade_mode_for_strategy(strategy),
            )
        if selected_master_config is not None:
            split_metrics = _build_master_split_metrics(
                data=filtered_data,
                master=selected_master_config,
                initial_capital=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
                validation_pct=validation_pct,
            )
        else:
            split_metrics = _build_split_metrics(
                data=filtered_data,
                strategy=strategy,
                fast_period=fast_period,
                slow_period=slow_period,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                bollinger_period=bollinger_period,
                bollinger_std=bollinger_std,
                initial_capital=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
                validation_pct=validation_pct,
                trade_mode=_trade_mode_for_strategy(strategy),
            )
        trade_log = _format_trade_log(backtest_result.trade_log)
        chart_data = analyzed_data.tail(max_rows)
        strategy_label = (
            f"Master System: {selected_master_config.name}"
            if selected_master_config is not None
            else _strategy_label(
                strategy=strategy,
                fast_period=fast_period,
                slow_period=slow_period,
                rsi_period=rsi_period,
                bollinger_period=bollinger_period,
                bollinger_std=bollinger_std,
            )
        )
        summary = {
            "path": str(path),
            "rows": len(filtered_data),
            "from": filtered_data.iloc[0]["datetime"] if not filtered_data.empty else "-",
            "to": filtered_data.iloc[-1]["datetime"] if not filtered_data.empty else "-",
            "shown_rows": len(chart_data),
            "exists": True,
            "strategy": strategy_label,
            "buy_signals": int((analyzed_data["signal"] == 1).sum())
            if "signal" in analyzed_data.columns
            else 0,
            "sell_signals": int((analyzed_data["signal"] == -1).sum())
            if "signal" in analyzed_data.columns
            else 0,
            "total_return_pct": f"{backtest_result.total_return_pct:.2f}%",
            "buy_and_hold_return_pct": f"{backtest_result.buy_and_hold_return_pct:.2f}%",
            "max_drawdown_pct": f"{backtest_result.max_drawdown_pct:.2f}%",
            "trades": backtest_result.trades,
            "win_rate_pct": f"{backtest_result.win_rate_pct:.2f}%",
            "profit_factor": _format_ratio(backtest_result.profit_factor),
            "best_trade_pct": f"{backtest_result.best_trade_pct:.2f}%",
            "worst_trade_pct": f"{backtest_result.worst_trade_pct:.2f}%",
            "average_trade_pct": f"{backtest_result.average_trade_pct:.2f}%",
            "max_winning_streak": backtest_result.max_winning_streak,
            "max_losing_streak": backtest_result.max_losing_streak,
            "fee_bps": f"{fee_bps:.2f}",
            "slippage_bps": f"{slippage_bps:.2f}",
            "initial_capital": _format_usd(initial_capital),
            "final_equity": _format_usd(backtest_result.final_equity),
            "profit_loss": _format_usd(backtest_result.final_equity - initial_capital),
            "position_mode": _position_mode_label(position_mode),
            "alpha_vs_buy_hold_pct": _format_pct(
                backtest_result.total_return_pct
                - backtest_result.buy_and_hold_return_pct
            ),
            "return_drawdown_ratio": _format_ratio(
                _calculate_return_drawdown_ratio(
                    backtest_result.total_return_pct,
                    backtest_result.max_drawdown_pct,
                )
            ),
            **split_metrics,
        }
        chart_html = _build_candlestick_chart(
            data=chart_data,
            symbol=symbol,
            timeframe=timeframe,
            show_sma_20=indicators["show_sma_20"],
            show_sma_50=indicators["show_sma_50"],
            show_ema_200=indicators["show_ema_200"],
            strategy=strategy,
            fast_period=fast_period,
            slow_period=slow_period,
            bollinger_period=bollinger_period,
        )
        equity_chart_html = _build_equity_curve_chart(
            equity_curve=backtest_result.equity_curve,
            strategy_name=strategy_label,
        )

    if _is_enabled(compare_timeframes):
        compare_results = _build_timeframe_comparison(
            exchange=exchange,
            symbol=symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            compound=compound,
            validation_pct=validation_pct,
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "request": request,
            "saved_masters": master_storage.list(),
            "master_config": master_config,
            "selected_master_config": selected_master_config,
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "max_rows": max_rows,
            "compare_timeframes": _is_enabled(compare_timeframes),
            "fetch_since": fetch_since,
            "fetch_until": fetch_until,
            "fetch_batch_limit": fetch_batch_limit,
            "fetch_max_batches": fetch_max_batches,
            "validation_pct": validation_pct,
            "show_sma_20": indicators["show_sma_20"],
            "show_sma_50": indicators["show_sma_50"],
            "show_ema_200": indicators["show_ema_200"],
            "strategy": strategy,
            "start_date": start_date,
            "end_date": end_date,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "bollinger_period": bollinger_period,
            "bollinger_std": bollinger_std,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "position_mode": position_mode,
            "summary": summary,
            "chart_html": chart_html,
            "equity_chart_html": equity_chart_html,
            "trade_log": trade_log,
            "compare_results": compare_results,
            "snip_url": snip_url,
            "snip_trigger_up_bid": snip_trigger_up_bid,
            "snip_trigger_down_bid": snip_trigger_down_bid,
            "snip_limit_up": snip_limit_up,
            "snip_limit_down": snip_limit_down,
            "snip_preview": snip_preview,
            "snip_error": snip_error,
            "data_message": data_message,
            "data_error": data_error,
            "data_messages": data_messages,
            "data_errors": data_errors,
        },
    )


def _is_enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_master_list(value: str) -> list[str]:
    masters = []
    for raw_symbol in value.split(","):
        symbol = raw_symbol.strip().upper()
        if not symbol:
            continue
        if "/" not in symbol:
            symbol = f"{symbol}/USDT"
        if symbol not in masters:
            masters.append(symbol)

    return masters or ["BTC/USDT"]


def _active_flag_names(flags: dict[str, bool]) -> list[str]:
    return [name for name, enabled in flags.items() if enabled]


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_ratio(value: float) -> str:
    if value == float("inf"):
        return "inf"

    return f"{value:.2f}"


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _format_probability(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value * 100:.2f}c"


def _format_snip_preview(preview) -> dict[str, str]:
    return {
        "slug": preview.slug,
        "title": preview.title,
        "interval": preview.interval,
        "up_label": preview.up_label,
        "down_label": preview.down_label,
        "up_bid": _format_probability(preview.up_bid),
        "down_bid": _format_probability(preview.down_bid),
        "up_ask": _format_probability(preview.up_ask),
        "down_ask": _format_probability(preview.down_ask),
        "condition": "YES" if preview.condition_met else "NO",
        "guaranteed_margin": f"{preview.guaranteed_margin * 100:+.2f}c/share",
        "message": preview.message,
    }


def _format_price(value: float) -> str:
    return f"{value:,.2f}"


def _format_datetime(value) -> str:
    if value is None:
        return "-"

    return str(value)


def _format_trade_log(trade_log) -> list[dict[str, str]]:
    if trade_log.empty:
        return []

    recent_trades = trade_log.tail(50).iloc[::-1]
    formatted = []
    for row in recent_trades.itertuples(index=False):
        formatted.append(
            {
                "entry_datetime": _format_datetime(row.entry_datetime),
                "exit_datetime": _format_datetime(row.exit_datetime),
                "entry_price": _format_price(row.entry_price),
                "exit_price": _format_price(row.exit_price),
                "return_pct": f"{row.return_pct:.2f}%",
                "pl_usd": _format_usd(row.pl_usd),
                "equity_after": _format_usd(row.equity_after),
            }
        )

    return formatted


def _build_split_metrics(
    data,
    strategy: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
    bollinger_period: int,
    bollinger_std: float,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    compound: bool,
    validation_pct: float,
    trade_mode: str,
) -> dict[str, str | int]:
    if data.empty:
        return {
            "train_rows": 0,
            "validation_rows": 0,
            "validation_from": "-",
            "train_return_pct": "0.00%",
            "train_drawdown_pct": "0.00%",
            "train_profit_factor": "0.00",
            "validation_return_pct": "0.00%",
            "validation_buy_and_hold_pct": "0.00%",
            "validation_alpha_pct": "0.00%",
            "validation_drawdown_pct": "0.00%",
            "validation_profit_factor": "0.00",
            "validation_return_drawdown_ratio": "0.00",
            "validation_verdict": "No data",
        }

    validation_rows = max(1, int(len(data) * validation_pct / 100))
    split_index = max(1, len(data) - validation_rows)
    train_data = data.iloc[:split_index].reset_index(drop=True)
    validation_data = data.iloc[split_index:].reset_index(drop=True)
    train_data = _apply_strategy(
        data=add_default_moving_averages(train_data),
        strategy=strategy,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        bollinger_period=bollinger_period,
        bollinger_std=bollinger_std,
    )
    validation_data = _apply_strategy(
        data=add_default_moving_averages(validation_data),
        strategy=strategy,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        bollinger_period=bollinger_period,
        bollinger_std=bollinger_std,
    )

    train_result = run_long_only_signal_backtest(
        data=train_data,
        initial_equity=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        compound=compound,
        trade_mode=trade_mode,
    )
    validation_result = run_long_only_signal_backtest(
        data=validation_data,
        initial_equity=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        compound=compound,
        trade_mode=trade_mode,
    )

    validation_from = (
        validation_data.iloc[0]["datetime"] if not validation_data.empty else "-"
    )
    validation_alpha = (
        validation_result.total_return_pct
        - validation_result.buy_and_hold_return_pct
    )
    return {
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "validation_from": validation_from,
        "train_return_pct": f"{train_result.total_return_pct:.2f}%",
        "train_drawdown_pct": f"{train_result.max_drawdown_pct:.2f}%",
        "train_profit_factor": _format_ratio(train_result.profit_factor),
        "validation_return_pct": f"{validation_result.total_return_pct:.2f}%",
        "validation_buy_and_hold_pct": (
            f"{validation_result.buy_and_hold_return_pct:.2f}%"
        ),
        "validation_alpha_pct": _format_pct(validation_alpha),
        "validation_drawdown_pct": f"{validation_result.max_drawdown_pct:.2f}%",
        "validation_profit_factor": _format_ratio(validation_result.profit_factor),
        "validation_return_drawdown_ratio": _format_ratio(
            _calculate_return_drawdown_ratio(
                validation_result.total_return_pct,
                validation_result.max_drawdown_pct,
            )
        ),
        "validation_verdict": _build_validation_verdict(
            validation_return_pct=validation_result.total_return_pct,
            validation_alpha_pct=validation_alpha,
            validation_profit_factor=validation_result.profit_factor,
        ),
    }


def _build_timeframe_comparison(
    exchange: str,
    symbol: str,
    strategy: str,
    start_date: str,
    end_date: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
    bollinger_period: int,
    bollinger_std: float,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    compound: bool,
    validation_pct: float,
) -> list[dict[str, str | int]]:
    timeframes = ["15m", "30m", "1h", "4h", "12h", "1d", "3d", "1w"]
    rows: list[dict[str, str | int]] = []

    for candidate_timeframe in timeframes:
        dataset = storage.load(exchange, symbol, candidate_timeframe)
        if dataset.empty:
            rows.append(
                {
                    "timeframe": candidate_timeframe,
                    "status": "Missing dataset",
                    "rows": 0,
                    "from": "-",
                    "to": "-",
                    "strategy_gain": "-",
                    "buy_hold": "-",
                    "alpha": "-",
                    "drawdown": "-",
                    "trades": "-",
                    "profit_factor": "-",
                    "validation_gain": "-",
                    "validation_alpha": "-",
                    "validation_profit_factor": "-",
                    "verdict": "-",
                }
            )
            continue

        filtered = _filter_by_date_range(dataset, start_date, end_date)
        if filtered.empty:
            rows.append(
                {
                    "timeframe": candidate_timeframe,
                    "status": "No rows in date range",
                    "rows": 0,
                    "from": "-",
                    "to": "-",
                    "strategy_gain": "-",
                    "buy_hold": "-",
                    "alpha": "-",
                    "drawdown": "-",
                    "trades": "-",
                    "profit_factor": "-",
                    "validation_gain": "-",
                    "validation_alpha": "-",
                    "validation_profit_factor": "-",
                    "verdict": "-",
                }
            )
            continue

        analyzed = _apply_strategy(
            data=add_default_moving_averages(filtered),
            strategy=strategy,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
        )
        trade_mode = _trade_mode_for_strategy(strategy)
        result = run_long_only_signal_backtest(
            data=analyzed,
            initial_equity=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            compound=compound,
            trade_mode=trade_mode,
        )
        split = _build_split_metrics(
            data=filtered,
            strategy=strategy,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            bollinger_period=bollinger_period,
            bollinger_std=bollinger_std,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            compound=compound,
            validation_pct=validation_pct,
            trade_mode=trade_mode,
        )
        alpha = result.total_return_pct - result.buy_and_hold_return_pct
        rows.append(
            {
                "timeframe": candidate_timeframe,
                "status": "OK",
                "rows": len(filtered),
                "from": str(filtered.iloc[0]["datetime"]),
                "to": str(filtered.iloc[-1]["datetime"]),
                "strategy_gain": _format_pct(result.total_return_pct),
                "buy_hold": _format_pct(result.buy_and_hold_return_pct),
                "alpha": _format_pct(alpha),
                "drawdown": _format_pct(result.max_drawdown_pct),
                "trades": result.trades,
                "profit_factor": _format_ratio(result.profit_factor),
                "validation_gain": split["validation_return_pct"],
                "validation_alpha": split["validation_alpha_pct"],
                "validation_profit_factor": split["validation_profit_factor"],
                "verdict": split["validation_verdict"],
            }
        )

    return rows


def _build_master_split_metrics(
    data,
    master,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    compound: bool,
    validation_pct: float,
) -> dict[str, str | int]:
    if data.empty:
        return {
            "train_rows": 0,
            "validation_rows": 0,
            "validation_from": "-",
            "train_return_pct": "0.00%",
            "train_drawdown_pct": "0.00%",
            "train_profit_factor": "0.00",
            "validation_return_pct": "0.00%",
            "validation_buy_and_hold_pct": "0.00%",
            "validation_alpha_pct": "0.00%",
            "validation_drawdown_pct": "0.00%",
            "validation_profit_factor": "0.00",
            "validation_return_drawdown_ratio": "0.00",
            "validation_verdict": "No data",
        }

    validation_rows = max(1, int(len(data) * validation_pct / 100))
    split_index = max(1, len(data) - validation_rows)
    train_data = data.iloc[:split_index].reset_index(drop=True)
    validation_data = data.iloc[split_index:].reset_index(drop=True)
    train_result = run_master_system_backtest(
        data=train_data,
        master=master,
        initial_equity=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        compound=compound,
    ).backtest
    validation_result = run_master_system_backtest(
        data=validation_data,
        master=master,
        initial_equity=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        compound=compound,
    ).backtest
    validation_alpha = (
        validation_result.total_return_pct
        - validation_result.buy_and_hold_return_pct
    )

    return {
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "validation_from": validation_data.iloc[0]["datetime"]
        if not validation_data.empty
        else "-",
        "train_return_pct": _format_pct(train_result.total_return_pct),
        "train_drawdown_pct": _format_pct(train_result.max_drawdown_pct),
        "train_profit_factor": _format_ratio(train_result.profit_factor),
        "validation_return_pct": _format_pct(validation_result.total_return_pct),
        "validation_buy_and_hold_pct": _format_pct(
            validation_result.buy_and_hold_return_pct
        ),
        "validation_alpha_pct": _format_pct(validation_alpha),
        "validation_drawdown_pct": _format_pct(validation_result.max_drawdown_pct),
        "validation_profit_factor": _format_ratio(validation_result.profit_factor),
        "validation_return_drawdown_ratio": _format_ratio(
            _calculate_return_drawdown_ratio(
                validation_result.total_return_pct,
                validation_result.max_drawdown_pct,
            )
        ),
        "validation_verdict": _build_validation_verdict(
            validation_return_pct=validation_result.total_return_pct,
            validation_alpha_pct=validation_alpha,
            validation_profit_factor=validation_result.profit_factor,
        ),
    }


def _download_all_listed_datasets(
    exchange: str,
    since: str,
    until: str | None,
    limit_per_request: int,
    max_batches: int | None,
) -> tuple[list[str], list[str]]:
    messages: list[str] = []
    errors: list[str] = []

    for candidate_symbol in SUPPORTED_SYMBOLS:
        for candidate_timeframe in SUPPORTED_TIMEFRAMES:
            try:
                fetched_data = provider.fetch_ohlcv_history(
                    exchange=exchange,
                    symbol=candidate_symbol,
                    timeframe=candidate_timeframe,
                    since=since,
                    until=until,
                    limit_per_request=limit_per_request,
                    max_batches=max_batches,
                )
                path = storage.save(
                    data=fetched_data,
                    exchange=exchange,
                    symbol=candidate_symbol,
                    timeframe=candidate_timeframe,
                )
                saved_data = storage.load(
                    exchange,
                    candidate_symbol,
                    candidate_timeframe,
                )
                messages.append(
                    f"Saved {len(saved_data)} candles for {exchange} "
                    f"{candidate_symbol} {candidate_timeframe} into {path}"
                )
            except (MarketDataError, ValueError) as exc:
                errors.append(
                    f"{exchange} {candidate_symbol} {candidate_timeframe}: {exc}"
                )

    return messages, errors


def _calculate_return_drawdown_ratio(
    return_pct: float,
    drawdown_pct: float,
) -> float:
    if drawdown_pct == 0:
        return 0.0

    return return_pct / abs(drawdown_pct)


def _build_validation_verdict(
    validation_return_pct: float,
    validation_alpha_pct: float,
    validation_profit_factor: float,
) -> str:
    if (
        validation_return_pct > 0
        and validation_alpha_pct > 0
        and validation_profit_factor > 1
    ):
        return "Promising"
    if validation_alpha_pct > 0:
        return "Defensive"
    if validation_return_pct > 0:
        return "Profitable but weak"

    return "Weak validation"


def _trade_mode_for_strategy(strategy: str) -> str:
    if strategy == "bear_trend_rider":
        return "short_only"
    if strategy == "ultimate_hybrid":
        return "bidirectional"

    return "long_only"


def _position_mode_label(position_mode: str) -> str:
    if position_mode == "fixed":
        return "Fixed stake"

    return "Compound equity"


def _filter_by_date_range(data, start_date: str, end_date: str):
    result = data.copy()
    if start_date.strip():
        start = _parse_date_filter(start_date, "start_date")
        result = result[result["datetime"] >= start]
    if end_date.strip():
        end = _parse_date_filter(end_date, "end_date")
        result = result[result["datetime"] <= end]

    return result.reset_index(drop=True)


def _parse_date_filter(value: str, field_name: str):
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} must be a valid date")

    return parsed


def _apply_strategy(
    data,
    strategy: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
    bollinger_period: int,
    bollinger_std: float,
):
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


def _strategy_label(
    strategy: str,
    fast_period: int = 20,
    slow_period: int = 50,
    rsi_period: int = 14,
    bollinger_period: int = 40,
    bollinger_std: float = 2.2,
) -> str:
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


def _build_candlestick_chart(
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
    if len(data) > 3000:
        figure = go.Figure(
            data=[
                go.Scattergl(
                    x=data["datetime"],
                    y=data["close"],
                    mode="lines",
                    name=f"{symbol} close",
                    line={"color": "#0f766e", "width": 1.5},
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
        _add_indicator_line(figure, data, "sma_20", "SMA 20", "#2563eb")
    if show_sma_50:
        _add_indicator_line(figure, data, "sma_50", "SMA 50", "#f59e0b")
    if show_ema_200:
        _add_indicator_line(figure, data, "ema_200", "EMA 200", "#7c3aed")
    if strategy == "ema_crossover":
        _add_indicator_line(
            figure,
            data,
            f"ema_{fast_period}",
            f"EMA {fast_period}",
            "#0891b2",
        )
        _add_indicator_line(
            figure,
            data,
            f"ema_{slow_period}",
            f"EMA {slow_period}",
            "#be123c",
        )
    if strategy == "ema_trend_capture":
        _add_indicator_line(figure, data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, data, "ema_300", "EMA 300", "#be123c")
    if strategy == "ema_donchian_trend":
        _add_indicator_line(figure, data, "ema_50", "EMA 50", "#0891b2")
        _add_indicator_line(figure, data, "ema_200", "EMA 200", "#be123c")
    if strategy == "adaptive_trend_guard":
        _add_indicator_line(figure, data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, data, "ema_300", "EMA 300", "#be123c")
    if strategy == "bull_trend_rider":
        _add_indicator_line(figure, data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, data, "ema_300", "EMA 300", "#be123c")
    if strategy == "bear_trend_rider":
        _add_indicator_line(figure, data, "ema_80", "EMA 80", "#0891b2")
        _add_indicator_line(figure, data, "ema_300", "EMA 300", "#be123c")
    if strategy == "panic_dip_buyer":
        _add_indicator_line(figure, data, "ema_50", "EMA 50", "#0891b2")
        _add_indicator_line(figure, data, "ema_300", "EMA 300", "#be123c")
        _add_indicator_line(figure, data, "bb_lower_40", "BB Lower", "#64748b")
        _add_indicator_line(figure, data, "bb_middle_40", "BB Middle", "#94a3b8")
    if strategy == "ultimate_hybrid":
        _add_indicator_line(figure, data, f"ema_{fast_period}", f"EMA {fast_period}", "#0891b2")
        _add_indicator_line(figure, data, f"ema_{slow_period}", f"EMA {slow_period}", "#be123c")
        _add_indicator_line(figure, data, f"bb_upper_{bollinger_period}", "BB Upper", "#64748b")
        _add_indicator_line(figure, data, f"bb_lower_{bollinger_period}", "BB Lower", "#64748b")
    if strategy in {"bollinger_mean_reversion", "bollinger_trend_reversion"}:
        _add_indicator_line(
            figure,
            data,
            f"bb_upper_{bollinger_period}",
            "BB Upper",
            "#64748b",
        )
        _add_indicator_line(
            figure,
            data,
            f"bb_middle_{bollinger_period}",
            "BB Middle",
            "#94a3b8",
        )
        _add_indicator_line(
            figure,
            data,
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
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )


def _build_equity_curve_chart(equity_curve, strategy_name: str) -> str:
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
    indicator = data.dropna(subset=[column])
    if indicator.empty:
        return

    figure.add_trace(
        go.Scattergl(
            x=indicator["datetime"],
            y=indicator[column],
            mode="lines",
            name=name,
            line={"color": color, "width": 1.6},
        )
    )
