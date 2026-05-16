import json
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.backtesting.simple_backtester import BacktestResult, run_long_only_signal_backtest
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
from app.strategies.custom_loader import (
    apply_custom_strategy,
    delete_custom_strategy,
    list_custom_strategies,
    save_custom_strategy,
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
IRT_RUNS_PATH = Path("data/irt_runs.json")
IRT_CLOSED_RUNS_PATH = Path("data/irt_closed_runs.json")
BACKTEST_RESULTS_PATH = Path("data/backtest_results.json")
PAPER_POLL_SECONDS = 60
_paper_worker_started = False

SUPPORTED_EXCHANGES = ["binance", "coinbase", "kraken", "okx", "bybit"]
SUPPORTED_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
]
SUPPORTED_TIMEFRAMES = ["1m", "15m", "30m", "1h", "4h", "12h", "1d", "3d", "1w"]


@router.on_event("startup")
def start_paper_trading_worker() -> None:
    global _paper_worker_started
    if _paper_worker_started:
        return

    _paper_worker_started = True
    worker = threading.Thread(target=_paper_trading_loop, daemon=True)
    worker.start()


@router.get("/home", response_class=HTMLResponse)
def home(
    request: Request,
    upload_message: str = Query(default=""),
    upload_error: str = Query(default=""),
    refresh_message: str = Query(default=""),
    refresh_error: str = Query(default=""),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "home.html",
        context={
            "request": request,
            "strategy_cards": _build_home_strategy_cards(),
            "custom_strategies": list_custom_strategies(),
            "upload_message": upload_message,
            "upload_error": upload_error,
            "refresh_message": refresh_message.replace("+", " "),
            "refresh_error": refresh_error.replace("+", " "),
        },
    )


@router.get("/irt", response_class=HTMLResponse)
def irt(
    request: Request,
    symbol: str = Query(default="BTC/USDT"),
    timeframe: str = Query(default="1h"),
    strategy: str = Query(default="adaptive_trend_guard"),
    initial_capital: float = Query(default=1000.0, gt=0.0),
    lookback: int = Query(default=500, ge=50, le=1500),
    fee_bps: float = Query(default=10.0, ge=0.0, le=1000.0),
    slippage_bps: float = Query(default=2.0, ge=0.0, le=1000.0),
    position_mode: str = Query(default="compound"),
    run_irt: str = Query(default="false"),
    refresh_message: str = Query(default=""),
    refresh_error: str = Query(default=""),
    show_chart: str = Query(default="false"),
) -> HTMLResponse:
    data_error = refresh_error.replace("+", " ")
    data_message = refresh_message.replace("+", " ")
    latest_price = "-"
    latest_datetime = "-"
    summary = _build_empty_irt_summary(initial_capital)
    trade_log = []
    chart_html = ""
    equity_chart_html = ""
    is_active_run = _get_irt_run(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
        }
    ) is not None
    active_run = _get_irt_run(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
        }
    )
    run_enabled = _is_run_enabled(active_run) if active_run else True

    if _is_enabled(run_irt):
        try:
            live_data = provider.fetch_ohlcv(
                exchange="binance",
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback,
            )
            storage.save(live_data, "binance", symbol, timeframe)
            analyzed_data = add_default_moving_averages(live_data)
            analyzed_data = _apply_strategy(
                data=analyzed_data,
                strategy=strategy,
                fast_period=20,
                slow_period=50,
                rsi_period=14,
                rsi_oversold=30,
                rsi_overbought=70,
                bollinger_period=40,
                bollinger_std=2.2,
            )
            latest = analyzed_data.iloc[-1]
            latest_price = _format_usd(float(latest["close"]))
            latest_datetime = latest["datetime"]
            run_payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "initial_capital": initial_capital,
                "lookback": lookback,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "position_mode": position_mode,
            }
            existing_run = _get_irt_run(run_payload)
            started_timestamp = int(
                existing_run.get("started_timestamp", latest["timestamp"])
            ) if existing_run else int(latest["timestamp"])
            started_datetime = (
                existing_run.get("started_datetime")
                if existing_run and existing_run.get("started_datetime") is not None
                else str(latest_datetime)
            )
            chart_data = _hide_pre_irt_signals(analyzed_data, started_timestamp)
            chart_html = _build_candlestick_chart(
                data=chart_data,
                symbol=symbol,
                timeframe=timeframe,
                show_sma_20=True,
                show_sma_50=True,
                show_ema_200=True,
                strategy=strategy,
                fast_period=20,
                slow_period=50,
                bollinger_period=40,
            )
            saved_run = _save_irt_run(
                {
                    **run_payload,
                    "started_timestamp": started_timestamp,
                    "started_datetime": started_datetime,
                }
            )
            refreshed_run = _refresh_single_paper_run(saved_run)
            summary = _build_paper_summary(refreshed_run, latest)
            trade_log = _format_paper_trade_log(refreshed_run)
            equity_chart_html = _build_equity_curve_chart(
                _paper_equity_curve(refreshed_run),
                _strategy_label(strategy),
            )
            data_message = "Paper trading is active. It will process new candles automatically."
            is_active_run = True
        except (MarketDataError, ValueError) as exc:
            data_error = str(exc)
    elif is_active_run:
        try:
            run = _get_irt_run(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": strategy,
                }
            )
            refreshed_run = _normalize_paper_run(run) if run else None
            if refreshed_run is not None:
                latest_price = _format_optional_usd(refreshed_run.get("latest_price"))
                latest_datetime = refreshed_run.get("latest_datetime", "-")
                summary = _build_paper_summary(refreshed_run, None)
                trade_log = _format_paper_trade_log(refreshed_run)
                if _is_enabled(show_chart):
                    live_data = storage.load("binance", symbol, timeframe).tail(lookback)
                    if not live_data.empty:
                        analyzed_data = add_default_moving_averages(live_data)
                        analyzed_data = _apply_strategy(
                            data=analyzed_data,
                            strategy=strategy,
                            fast_period=20,
                            slow_period=50,
                            rsi_period=14,
                            rsi_oversold=30,
                            rsi_overbought=70,
                            bollinger_period=40,
                            bollinger_std=2.2,
                        )
                        latest = analyzed_data.iloc[-1]
                        latest_price = _format_usd(float(latest["close"]))
                        latest_datetime = latest["datetime"]
                        started_timestamp = int(refreshed_run.get("started_timestamp", latest["timestamp"]))
                        chart_data = _hide_pre_irt_signals(analyzed_data, started_timestamp)
                        chart_html = _build_candlestick_chart(
                            data=chart_data,
                            symbol=symbol,
                            timeframe=timeframe,
                            show_sma_20=True,
                            show_sma_50=True,
                            show_ema_200=True,
                            strategy=strategy,
                            fast_period=20,
                            slow_period=50,
                            bollinger_period=40,
                        )
                        equity_chart_html = _build_equity_curve_chart(
                            _paper_equity_curve(refreshed_run),
                            _strategy_label(strategy),
                        )
        except (MarketDataError, ValueError) as exc:
            data_error = str(exc)

    return templates.TemplateResponse(
        request,
        "irt.html",
        context={
            "request": request,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "initial_capital": initial_capital,
            "lookback": lookback,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "position_mode": position_mode,
            "latest_price": latest_price,
            "latest_datetime": latest_datetime,
            "strategy_options": _strategy_options(),
            "timeframe_options": SUPPORTED_TIMEFRAMES,
            "symbol_options": SUPPORTED_SYMBOLS,
            "summary": summary,
            "trade_log": trade_log,
            "chart_html": chart_html,
            "equity_chart_html": equity_chart_html,
            "data_message": data_message,
            "data_error": data_error,
            "is_active_run": is_active_run,
            "run_enabled": run_enabled,
            "next_enabled": "false" if run_enabled else "true",
            "toggle_label": "Apagar" if run_enabled else "Prender",
            "toggle_icon": "power" if run_enabled else "power-off",
            "state_label": "Prendida" if run_enabled else "Apagada",
            "state_class": "on" if run_enabled else "off",
            "finalize_href": _build_finalize_href(symbol, timeframe, strategy),
            "show_chart": _is_enabled(show_chart),
            "chart_href": _build_irt_href(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                initial_capital=initial_capital,
                lookback=lookback,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                position_mode=position_mode,
            ) + "&show_chart=true",
        },
    )


@router.get("/irt/finalize", response_class=HTMLResponse)
def confirm_finalize_irt(
    request: Request,
    symbol: str = Query(...),
    timeframe: str = Query(...),
    strategy: str = Query(...),
) -> HTMLResponse:
    run = _get_irt_run(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
        }
    )
    if run is None:
        return templates.TemplateResponse(
            request,
            "irt_finalize.html",
            context={
                "request": request,
                "found": False,
                "completed": False,
                "message": "La ejecucion IRT ya no esta activa.",
                "run": {},
                "summary": {},
                "close_preview": {},
                "trade_log": [],
            },
        )

    refreshed_run = _refresh_single_paper_run(run) or _normalize_paper_run(run)
    close_preview = _build_finalize_preview(refreshed_run)
    summary = _build_paper_summary(refreshed_run, None)
    return templates.TemplateResponse(
        request,
        "irt_finalize.html",
        context={
            "request": request,
            "found": True,
            "completed": False,
            "message": "",
            "run": refreshed_run,
            "summary": summary,
            "close_preview": close_preview,
            "trade_log": _format_paper_trade_log(refreshed_run),
            "irt_href": _build_irt_href(
                symbol=str(refreshed_run.get("symbol", symbol)),
                timeframe=str(refreshed_run.get("timeframe", timeframe)),
                strategy=str(refreshed_run.get("strategy", strategy)),
                initial_capital=float(refreshed_run.get("initial_capital", 1000.0)),
                lookback=int(refreshed_run.get("lookback", 500)),
                fee_bps=float(refreshed_run.get("fee_bps", 10.0)),
                slippage_bps=float(refreshed_run.get("slippage_bps", 2.0)),
                position_mode=str(refreshed_run.get("position_mode", "compound")),
            ),
        },
    )


@router.post("/irt/finalize", response_class=HTMLResponse)
def finalize_irt(
    request: Request,
    symbol: str = Form(...),
    timeframe: str = Form(...),
    strategy: str = Form(...),
    confirm: str = Form(default="no"),
) -> HTMLResponse:
    if confirm != "yes":
        return RedirectResponse(url="/home", status_code=303)

    run = _get_irt_run(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
        }
    )
    if run is None:
        return templates.TemplateResponse(
            request,
            "irt_finalize.html",
            context={
                "request": request,
                "found": False,
                "completed": False,
                "message": "La ejecucion IRT ya habia sido finalizada.",
                "run": {},
                "summary": {},
                "close_preview": {},
                "trade_log": [],
            },
        )

    refreshed_run = _refresh_single_paper_run(run) or _normalize_paper_run(run)
    finalized_run = _finalize_paper_run(refreshed_run)
    _save_closed_irt_run(finalized_run)
    _delete_irt_run(finalized_run)
    summary = _build_paper_summary(finalized_run, None)
    close_preview = _build_finalize_preview(finalized_run)
    return templates.TemplateResponse(
        request,
        "irt_finalize.html",
        context={
            "request": request,
            "found": True,
            "completed": True,
            "message": "Ejecucion IRT finalizada.",
            "run": finalized_run,
            "summary": summary,
            "close_preview": close_preview,
            "trade_log": _format_paper_trade_log(finalized_run),
        },
    )


@router.post("/irt/refresh")
def refresh_irt(
    symbol: str = Form(...),
    timeframe: str = Form(...),
    strategy: str = Form(...),
    redirect_to: str = Form(default="irt"),
) -> RedirectResponse:
    try:
        run = _get_irt_run(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
            }
        )
        if run is None:
            raise ValueError("La ejecucion IRT ya no esta activa.")
        if not _is_run_enabled(run):
            raise ValueError("La ejecucion esta apagada. Prendela antes de actualizar.")

        refreshed = _refresh_single_paper_run(run)
        if refreshed is None:
            raise ValueError("No se pudo actualizar la ejecucion.")
        message = "IRT+actualizado"
        error = ""
    except (MarketDataError, ValueError) as exc:
        message = ""
        error = str(exc).replace(" ", "+")

    if redirect_to == "home":
        suffix = f"refresh_message={message}" if message else f"refresh_error={error}"
        return RedirectResponse(url=f"/home?{suffix}", status_code=303)

    run_for_url = _get_irt_run(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
        }
    ) or {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "initial_capital": 1000.0,
        "lookback": 500,
        "fee_bps": 10.0,
        "slippage_bps": 2.0,
        "position_mode": "compound",
    }
    href = _build_irt_href(
        symbol=str(run_for_url.get("symbol", symbol)),
        timeframe=str(run_for_url.get("timeframe", timeframe)),
        strategy=str(run_for_url.get("strategy", strategy)),
        initial_capital=float(run_for_url.get("initial_capital", 1000.0)),
        lookback=int(run_for_url.get("lookback", 500)),
        fee_bps=float(run_for_url.get("fee_bps", 10.0)),
        slippage_bps=float(run_for_url.get("slippage_bps", 2.0)),
        position_mode=str(run_for_url.get("position_mode", "compound")),
    )
    suffix = f"refresh_message={message}" if message else f"refresh_error={error}"
    separator = "&" if "?" in href else "?"
    return RedirectResponse(url=f"{href}{separator}{suffix}", status_code=303)


@router.post("/irt/toggle")
def toggle_irt(
    symbol: str = Form(...),
    timeframe: str = Form(...),
    strategy: str = Form(...),
    enabled: str = Form(...),
    redirect_to: str = Form(default="home"),
) -> RedirectResponse:
    run = _get_irt_run(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
        }
    )
    if run is None:
        target = "/home?refresh_error=La+ejecucion+IRT+ya+no+esta+activa"
        return RedirectResponse(url=target, status_code=303)

    run = _normalize_paper_run(run)
    run["enabled"] = _is_enabled(enabled)
    _replace_irt_run(run)
    state = "prendida" if run["enabled"] else "apagada"
    message = f"Ejecucion+{state}"

    if redirect_to == "irt":
        href = _build_irt_href(
            symbol=str(run.get("symbol", symbol)),
            timeframe=str(run.get("timeframe", timeframe)),
            strategy=str(run.get("strategy", strategy)),
            initial_capital=float(run.get("initial_capital", 1000.0)),
            lookback=int(run.get("lookback", 500)),
            fee_bps=float(run.get("fee_bps", 10.0)),
            slippage_bps=float(run.get("slippage_bps", 2.0)),
            position_mode=str(run.get("position_mode", "compound")),
        )
        return RedirectResponse(url=f"{href}&refresh_message={message}", status_code=303)

    return RedirectResponse(url=f"/home?refresh_message={message}", status_code=303)


@router.post("/irt/refresh-all")
def refresh_all_irt() -> RedirectResponse:
    try:
        _refresh_paper_runs()
        return RedirectResponse(url="/home?refresh_message=IRT+actualizado", status_code=303)
    except (MarketDataError, ValueError) as exc:
        message = str(exc).replace(" ", "+")
        return RedirectResponse(url=f"/home?refresh_error={message}", status_code=303)


@router.get("/irt/history", response_class=HTMLResponse)
def irt_history(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "irt_history.html",
        context={
            "request": request,
            "history_rows": _build_closed_irt_rows(),
        },
    )


@router.post("/strategies/upload")
def upload_strategy(strategy_file: UploadFile = File(...)) -> RedirectResponse:
    try:
        content = strategy_file.file.read()
        strategy = save_custom_strategy(strategy_file.filename or "strategy.py", content)
        return RedirectResponse(
            url=f"/?strategy={strategy.key}&upload_message=Strategy+uploaded",
            status_code=303,
        )
    except ValueError as exc:
        return RedirectResponse(
            url=f"/?upload_error={str(exc).replace(' ', '+')}",
            status_code=303,
        )


@router.post("/strategies/delete")
def delete_strategy(strategy_key: str = Form(...)) -> RedirectResponse:
    try:
        deleted = delete_custom_strategy(strategy_key)
        return RedirectResponse(
            url=f"/?upload_message=Deleted+{deleted.label.replace(' ', '+')}",
            status_code=303,
        )
    except ValueError as exc:
        return RedirectResponse(
            url=f"/?upload_error={str(exc).replace(' ', '+')}",
            status_code=303,
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
    compare_strategies: str = Query(default="false"),
    strategy_compare_scope: str = Query(default="current"),
    walk_forward: str = Query(default="false"),
    save_result: str = Query(default="false"),
    run_analysis: str = Query(default="false"),
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
    upload_message: str = Query(default=""),
    upload_error: str = Query(default=""),
) -> HTMLResponse:
    selected_master_config = None
    if master_config.strip():
        selected_master_config = master_storage.get(master_config)
        if selected_master_config is not None:
            symbol = selected_master_config.symbol
            timeframe = selected_master_config.main_timeframe
            strategy = f"master:{selected_master_config.name}"
    elif strategy.startswith("master:"):
        selected_master_config = master_storage.get(strategy.removeprefix("master:"))
        if selected_master_config is not None:
            symbol = selected_master_config.symbol
            timeframe = selected_master_config.main_timeframe

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
    strategy_compare_results = []
    strategy_compare_summary = []
    walk_forward_results = []
    dataset_status = _build_dataset_status(exchange)
    saved_result_message = ""
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

    should_run_analysis = _is_enabled(run_analysis) or _is_enabled(save_result)

    if should_run_analysis and not data.empty:
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

    should_compare_strategies = (
        _is_enabled(compare_strategies) or strategy_compare_scope == "all"
    )
    if should_compare_strategies:
        strategy_compare_timeframes = (
            SUPPORTED_TIMEFRAMES
            if strategy_compare_scope == "all"
            else [timeframe]
        )
        strategy_compare_results = _build_strategy_comparison(
            exchange=exchange,
            symbol=symbol,
            timeframes=strategy_compare_timeframes,
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
        strategy_compare_summary = _build_strategy_comparison_summary(
            strategy_compare_results
        )

    if _is_enabled(walk_forward):
        walk_forward_results = _build_walk_forward_results(
            data=data,
            selected_master_config=selected_master_config,
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
        )

    if _is_enabled(save_result) and summary["rows"]:
        _save_backtest_snapshot(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy_label if "strategy_label" in locals() else str(summary["strategy"]),
            strategy_key=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_mode=position_mode,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            summary=summary,
        )
        saved_result_message = "Backtest result saved."

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
            "compare_strategies": should_compare_strategies,
            "strategy_compare_scope": strategy_compare_scope,
            "walk_forward": _is_enabled(walk_forward),
            "run_analysis": should_run_analysis,
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
            "strategy_compare_results": strategy_compare_results,
            "strategy_compare_summary": strategy_compare_summary,
            "walk_forward_results": walk_forward_results,
            "dataset_status": dataset_status,
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
            "saved_result_message": saved_result_message,
            "saved_results": _load_backtest_snapshots(limit=12),
            "strategy_options": _strategy_options(),
            "custom_strategies": list_custom_strategies(),
            "upload_message": upload_message,
            "upload_error": upload_error,
        },
    )


def _is_enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _load_backtest_snapshots(limit: int | None = None) -> list[dict[str, object]]:
    if not BACKTEST_RESULTS_PATH.exists():
        return []

    try:
        items = json.loads(BACKTEST_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    items = list(reversed(items))
    return items[:limit] if limit is not None else items


def _save_backtest_snapshot(
    exchange: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    strategy_key: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    position_mode: str,
    fee_bps: float,
    slippage_bps: float,
    summary: dict[str, object],
) -> None:
    BACKTEST_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = _load_backtest_snapshots(limit=None)
    items = list(reversed(items))
    items.append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "strategy_key": strategy_key,
            "start_date": start_date,
            "end_date": end_date,
            "date_range": f"{start_date or 'all'} to {end_date or 'latest'}",
            "initial_capital": initial_capital,
            "position_mode": position_mode,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "strategy_gain": summary["total_return_pct"],
            "buy_hold": summary["buy_and_hold_return_pct"],
            "alpha": summary["alpha_vs_buy_hold_pct"],
            "drawdown": summary["max_drawdown_pct"],
            "trades": summary["trades"],
            "profit_factor": summary["profit_factor"],
            "validation_gain": summary["validation_return_pct"],
            "validation_verdict": summary["validation_verdict"],
        }
    )
    BACKTEST_RESULTS_PATH.write_text(
        json.dumps(items[-300:], indent=2),
        encoding="utf-8",
    )


def _strategy_options() -> list[dict[str, str]]:
    options = _module_strategy_options()
    options.extend(
        {"value": f"master:{master.name}", "label": f"Master: {master.name}"}
        for master in master_storage.list()
    )
    return options


def _module_strategy_options() -> list[dict[str, str]]:
    options = _base_strategy_options()
    options.extend(
        {"value": item.key, "label": f"Custom: {item.label}"}
        for item in list_custom_strategies()
    )
    return options


def _base_strategy_options() -> list[dict[str, str]]:
    return [
        {"value": "none", "label": "None"},
        {"value": "sma_crossover", "label": "SMA 20/50 Crossover"},
        {"value": "ema_crossover", "label": "EMA 20/50 Crossover"},
        {"value": "rsi_mean_reversion", "label": "RSI 14 Mean Reversion"},
        {"value": "bollinger_mean_reversion", "label": "Bollinger Mean Reversion"},
        {"value": "bollinger_trend_reversion", "label": "Bollinger Trend Filtered"},
        {"value": "ema_trend_capture", "label": "EMA Trend Capture 80/300"},
        {"value": "ema_donchian_trend", "label": "EMA Donchian Trend 50/200"},
        {"value": "adaptive_trend_guard", "label": "Adaptive Trend Guard 80/300"},
        {"value": "bull_trend_rider", "label": "Bull Trend Rider 80/300"},
        {"value": "bear_trend_rider", "label": "Bear Trend Rider 80/300"},
        {"value": "panic_dip_buyer", "label": "Panic Dip Buyer"},
        {"value": "ultimate_hybrid", "label": "Ultimate Hybrid System"},
    ]


def _build_empty_irt_summary(initial_capital: float) -> dict[str, str | int]:
    return {
        "mode": "Waiting",
        "rows": 0,
        "strategy": "None",
        "initial_capital": _format_usd(initial_capital),
        "final_equity": _format_usd(initial_capital),
        "profit_loss": _format_usd(0),
        "total_return_pct": "0.00%",
        "buy_and_hold_return_pct": "0.00%",
        "max_drawdown_pct": "0.00%",
        "trades": 0,
        "win_rate_pct": "0.00%",
        "profit_factor": "0.00",
        "last_signal": "WAIT",
        "position": "Sin posicion",
        "entry_price": "-",
        "floating_pl": "$0.00",
        "latest_price": "-",
        "last_update": "-",
        "position_mode": "Compound equity",
        "fee_bps": "0.00",
        "slippage_bps": "0.00",
    }


def _build_irt_summary(
    data,
    strategy: str,
    initial_capital: float,
    position_mode: str,
    fee_bps: float,
    slippage_bps: float,
    result,
) -> dict[str, str | int]:
    latest_signal = int(data.iloc[-1]["signal"]) if not data.empty and "signal" in data.columns else 0
    signal_label = "BUY" if latest_signal == 1 else "SELL" if latest_signal == -1 else "WAIT"
    profit_loss = result.final_equity - initial_capital
    return {
        "mode": "Fictional IRT",
        "rows": len(data),
        "strategy": _strategy_label(strategy),
        "initial_capital": _format_usd(initial_capital),
        "final_equity": _format_usd(result.final_equity),
        "profit_loss": _format_usd(profit_loss),
        "total_return_pct": _format_pct(result.total_return_pct),
        "buy_and_hold_return_pct": _format_pct(result.buy_and_hold_return_pct),
        "max_drawdown_pct": _format_pct(result.max_drawdown_pct),
        "trades": result.trades,
        "win_rate_pct": _format_pct(result.win_rate_pct),
        "profit_factor": _format_ratio(result.profit_factor),
        "last_signal": signal_label,
        "position_mode": _position_mode_label(position_mode),
        "fee_bps": f"{fee_bps:.2f}",
        "slippage_bps": f"{slippage_bps:.2f}",
    }


def _build_home_strategy_cards() -> list[dict[str, str]]:
    cards = []
    runs = _load_irt_runs()

    for run in runs:
        run = _normalize_paper_run(run)
        symbol = str(run.get("symbol", "BTC/USDT"))
        timeframe = str(run.get("timeframe", "4h"))
        strategy = str(run.get("strategy", "none"))
        initial_capital = float(run.get("initial_capital", 1000.0))
        lookback = int(run.get("lookback", 500))
        fee_bps = float(run.get("fee_bps", 10.0))
        slippage_bps = float(run.get("slippage_bps", 2.0))
        position_mode = str(run.get("position_mode", "compound"))
        enabled = _is_run_enabled(run)
        created_at = str(run.get("created_at", ""))
        started_timestamp = run.get("started_timestamp")
        started_datetime = str(run.get("started_datetime", ""))
        start_date = started_datetime[:10] if started_datetime else "-"
        card = {
            "icon": _symbol_icon(symbol),
            "symbol": symbol,
            "strategy_key": strategy,
            "strategy_label": _strategy_label(strategy),
            "timeframe": timeframe,
            "start_date": start_date,
            "active_time": _format_active_time(created_at, pd.Timestamp.utcnow()),
            "initial_capital": _format_usd(initial_capital),
            "profit_loss": _format_usd(0),
            "profit_loss_class": "flat",
            "gain_pct": "0.00%",
            "drawdown_pct": "0.00%",
            "trades": "0",
            "last_signal": "WAIT",
            "position": _position_label(str(run.get("position", "flat"))),
            "entry_price": "-",
            "floating_pl": _format_usd(0),
            "floating_pl_class": "flat",
            "latest_price": _format_optional_usd(run.get("latest_price")),
            "last_update": _short_datetime(run.get("last_run_at")),
            "enabled": enabled,
            "next_enabled": "false" if enabled else "true",
            "toggle_label": "Apagar" if enabled else "Prender",
            "toggle_icon": "power" if enabled else "power-off",
            "state_label": "Prendida" if enabled else "Apagada",
            "state_class": "on" if enabled else "off",
            "status": "Apagada" if not enabled else "Waiting",
            "href": _build_irt_href(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                initial_capital=initial_capital,
                lookback=lookback,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                position_mode=position_mode,
            ),
            "finalize_href": _build_finalize_href(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
            ),
        }

        try:
            latest_close = _latest_close_for_run(run)
            marked_equity = _paper_marked_equity(run, latest_close)
            profit_loss = marked_equity - initial_capital
            entry_equity = float(run.get("entry_equity") or marked_equity)
            floating_pl = marked_equity - entry_equity if run.get("position") == "long" else 0.0
            latest_signal = int(run.get("last_signal", 0))
            card.update(
                {
                    "profit_loss": _format_usd(profit_loss),
                    "profit_loss_class": "gain" if profit_loss > 0 else "loss"
                    if profit_loss < 0
                    else "flat",
                    "gain_pct": _format_pct((marked_equity / initial_capital - 1) * 100),
                    "drawdown_pct": _format_pct(float(run.get("max_drawdown_pct", 0.0))),
                    "trades": str(run.get("trades", 0)),
                    "last_signal": _signal_label(latest_signal),
                    "position": _position_label(str(run.get("position", "flat"))),
                    "entry_price": _format_optional_usd(run.get("entry_price")),
                    "floating_pl": _format_usd(floating_pl),
                    "floating_pl_class": "gain" if floating_pl > 0 else "loss"
                    if floating_pl < 0
                    else "flat",
                    "latest_price": _format_optional_usd(latest_close),
                    "last_update": _short_datetime(run.get("last_run_at")),
                    "status": "Apagada" if not enabled else "In position"
                    if run.get("position") == "long"
                    else "Watching",
                }
            )
        except Exception as exc:
            card["status"] = f"Error: {exc}"

        cards.append(card)

    return cards


def _build_closed_irt_rows() -> list[dict[str, str]]:
    rows = []
    for run in _load_closed_irt_runs():
        run = _normalize_paper_run(run)
        initial_capital = float(run.get("initial_capital", 1000.0))
        final_equity = float(run.get("final_equity", run.get("equity", initial_capital)))
        profit_loss = final_equity - initial_capital
        trades = int(run.get("trades", 0))
        win_rate = int(run.get("winning_trades", 0)) / trades * 100 if trades else 0.0
        rows.append(
            {
                "symbol": str(run.get("symbol", "-")),
                "timeframe": str(run.get("timeframe", "-")),
                "strategy": _strategy_label(str(run.get("strategy", "none"))),
                "started": _short_datetime(run.get("started_datetime")),
                "closed": _short_datetime(run.get("closed_at")),
                "initial_capital": _format_usd(initial_capital),
                "final_equity": _format_usd(final_equity),
                "profit_loss": _format_usd(profit_loss),
                "profit_loss_class": "gain" if profit_loss > 0 else "loss"
                if profit_loss < 0
                else "flat",
                "return_pct": _format_pct((final_equity / initial_capital - 1) * 100),
                "trades": str(trades),
                "win_rate": _format_pct(win_rate),
                "profit_factor": _paper_profit_factor(run),
                "drawdown": _format_pct(float(run.get("max_drawdown_pct", 0.0))),
                "last_price": _format_optional_usd(run.get("latest_price")),
            }
        )

    return rows


def _default_timeframe_for_strategy(strategy_key: str) -> str:
    lowered = strategy_key.lower()
    if "1d" in lowered:
        return "1d"
    if "3d" in lowered:
        return "3d"

    return "4h"


def _load_irt_runs() -> list[dict]:
    if not IRT_RUNS_PATH.exists():
        return []

    try:
        raw = json.loads(IRT_RUNS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    return raw if isinstance(raw, list) else []


def _get_irt_run(run: dict) -> dict | None:
    key = _irt_run_key(run)
    for item in _load_irt_runs():
        if _irt_run_key(item) == key:
            return item

    return None


def _save_irt_run(run: dict) -> dict:
    IRT_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs = _load_irt_runs()
    key = _irt_run_key(run)
    now = pd.Timestamp.utcnow().isoformat()
    next_runs = []
    existing_created_at = None
    existing_started_timestamp = None
    existing_started_datetime = None

    for item in runs:
        if _irt_run_key(item) == key:
            existing_created_at = item.get("created_at")
            existing_started_timestamp = item.get("started_timestamp")
            existing_started_datetime = item.get("started_datetime")
            continue
        next_runs.append(item)

    saved_run = _normalize_paper_run(
        {
        **run,
        "created_at": existing_created_at or now,
        "started_timestamp": existing_started_timestamp
        if existing_started_timestamp is not None
        else run.get("started_timestamp"),
        "started_datetime": existing_started_datetime
        if existing_started_datetime is not None
        else run.get("started_datetime"),
        "last_run_at": now,
        }
    )
    next_runs.append(saved_run)
    next_runs = sorted(
        next_runs,
        key=lambda item: str(item.get("last_run_at", "")),
        reverse=True,
    )
    IRT_RUNS_PATH.write_text(
        json.dumps(next_runs, indent=2),
        encoding="utf-8",
    )
    return saved_run


def _replace_irt_run(updated_run: dict) -> dict:
    runs = _load_irt_runs()
    key = _irt_run_key(updated_run)
    next_runs = [item for item in runs if _irt_run_key(item) != key]
    normalized = _normalize_paper_run(updated_run)
    normalized["last_run_at"] = pd.Timestamp.utcnow().isoformat()
    next_runs.append(normalized)
    next_runs = sorted(
        next_runs,
        key=lambda item: str(item.get("last_run_at", "")),
        reverse=True,
    )
    IRT_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    IRT_RUNS_PATH.write_text(json.dumps(next_runs, indent=2), encoding="utf-8")
    return normalized


def _delete_irt_run(run: dict) -> None:
    key = _irt_run_key(run)
    next_runs = [item for item in _load_irt_runs() if _irt_run_key(item) != key]
    IRT_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    IRT_RUNS_PATH.write_text(json.dumps(next_runs, indent=2), encoding="utf-8")


def _load_closed_irt_runs() -> list[dict]:
    if not IRT_CLOSED_RUNS_PATH.exists():
        return []

    try:
        raw = json.loads(IRT_CLOSED_RUNS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    return raw if isinstance(raw, list) else []


def _save_closed_irt_run(run: dict) -> None:
    IRT_CLOSED_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs = _load_closed_irt_runs()
    runs.insert(0, run)
    IRT_CLOSED_RUNS_PATH.write_text(
        json.dumps(runs[:200], indent=2),
        encoding="utf-8",
    )


def _normalize_paper_run(run: dict) -> dict:
    initial_capital = float(run.get("initial_capital", 1000.0))
    result = dict(run)
    result.setdefault("enabled", True)
    result.setdefault("equity", initial_capital)
    result.setdefault("cash_equity", result.get("equity", initial_capital))
    result.setdefault("position", "flat")
    result.setdefault("entry_price", None)
    result.setdefault("entry_datetime", None)
    result.setdefault("entry_equity", None)
    result.setdefault("last_processed_timestamp", result.get("started_timestamp"))
    result.setdefault("last_signal", 0)
    result.setdefault("trades", 0)
    result.setdefault("winning_trades", 0)
    result.setdefault("losing_trades", 0)
    result.setdefault("max_drawdown_pct", 0.0)
    result.setdefault("peak_equity", initial_capital)
    result.setdefault("trade_log", [])
    result.setdefault("equity_curve", [])
    return result


def _paper_trading_loop() -> None:
    while True:
        try:
            _refresh_paper_runs()
        except Exception:
            pass
        time.sleep(PAPER_POLL_SECONDS)


def _refresh_paper_runs() -> None:
    for run in _load_irt_runs():
        if not _is_run_enabled(run):
            continue
        try:
            _refresh_single_paper_run(run)
        except Exception:
            continue


def _refresh_single_paper_run(run: dict | None) -> dict | None:
    if run is None:
        return None

    run = _normalize_paper_run(run)
    if not _is_run_enabled(run):
        return run
    symbol = str(run.get("symbol", "BTC/USDT"))
    timeframe = str(run.get("timeframe", "4h"))
    strategy = str(run.get("strategy", "none"))
    lookback = int(run.get("lookback", 500))
    started_timestamp = run.get("started_timestamp")
    if started_timestamp is None:
        return _replace_irt_run(run)

    live_data = provider.fetch_ohlcv(
        exchange="binance",
        symbol=symbol,
        timeframe=timeframe,
        limit=lookback,
    )
    storage.save(live_data, "binance", symbol, timeframe)
    analyzed = add_default_moving_averages(live_data)
    analyzed = _apply_strategy(
        data=analyzed,
        strategy=strategy,
        fast_period=20,
        slow_period=50,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        bollinger_period=40,
        bollinger_std=2.2,
    )
    last_processed = int(run.get("last_processed_timestamp") or started_timestamp)
    new_rows = analyzed[analyzed["timestamp"] > last_processed]

    for row in new_rows.itertuples(index=False):
        run = _process_paper_row(run, row)

    if not analyzed.empty:
        latest = analyzed.iloc[-1]
        run["latest_price"] = float(latest["close"])
        run["latest_datetime"] = str(latest["datetime"])

    return _replace_irt_run(run)


def _is_run_enabled(run: dict) -> bool:
    return bool(run.get("enabled", True))


def _process_paper_row(run: dict, row) -> dict:
    signal = int(getattr(row, "signal", 0))
    close = float(getattr(row, "close"))
    timestamp = int(getattr(row, "timestamp"))
    datetime = str(getattr(row, "datetime"))
    fee_rate = float(run.get("fee_bps", 0.0)) / 10_000
    slippage_rate = float(run.get("slippage_bps", 0.0)) / 10_000
    equity = float(run.get("equity", run.get("initial_capital", 1000.0)))
    position = run.get("position", "flat")
    run["last_signal"] = signal
    run["last_processed_timestamp"] = timestamp

    if position == "flat" and signal == 1:
        entry_price = close * (1 + slippage_rate)
        entry_equity = equity * (1 - fee_rate)
        run.update(
            {
                "position": "long",
                "entry_price": entry_price,
                "entry_datetime": datetime,
                "entry_equity": entry_equity,
                "equity": entry_equity,
            }
        )
    elif position == "long" and signal == -1:
        entry_price = float(run.get("entry_price") or close)
        entry_equity = float(run.get("entry_equity") or equity)
        exit_price = close * (1 - slippage_rate)
        gross_return = exit_price / entry_price if entry_price > 0 else 1.0
        final_equity = entry_equity * gross_return * (1 - fee_rate)
        trade_pl = final_equity - entry_equity
        trade_return_pct = (final_equity / entry_equity - 1) * 100 if entry_equity > 0 else 0.0
        trade_log = list(run.get("trade_log", []))
        trade_log.append(
            {
                "entry_datetime": run.get("entry_datetime"),
                "exit_datetime": datetime,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": trade_return_pct,
                "pl_usd": trade_pl,
                "equity_after": final_equity,
            }
        )
        run.update(
            {
                "position": "flat",
                "entry_price": None,
                "entry_datetime": None,
                "entry_equity": None,
                "equity": final_equity,
                "trade_log": trade_log[-100:],
                "trades": int(run.get("trades", 0)) + 1,
                "winning_trades": int(run.get("winning_trades", 0)) + (1 if trade_pl > 0 else 0),
                "losing_trades": int(run.get("losing_trades", 0)) + (1 if trade_pl < 0 else 0),
            }
        )

    marked_equity = _paper_marked_equity(run, close)
    peak_equity = max(float(run.get("peak_equity", marked_equity)), marked_equity)
    drawdown_pct = (marked_equity / peak_equity - 1) * 100 if peak_equity > 0 else 0.0
    run["peak_equity"] = peak_equity
    run["max_drawdown_pct"] = min(float(run.get("max_drawdown_pct", 0.0)), drawdown_pct)
    curve = list(run.get("equity_curve", []))
    curve.append({"datetime": datetime, "equity": marked_equity})
    run["equity_curve"] = curve[-1000:]
    return run


def _finalize_paper_run(run: dict) -> dict:
    run = _normalize_paper_run(run)
    latest = _latest_market_point_for_run(run)
    close = latest["close"]
    timestamp = latest["timestamp"]
    datetime = latest["datetime"]
    run["latest_price"] = close
    run["latest_datetime"] = datetime

    if run.get("position") == "long":
        run = _close_paper_long_position(
            run=run,
            close=close,
            timestamp=timestamp,
            datetime=datetime,
            forced=True,
        )
    else:
        marked_equity = _paper_marked_equity(run, close)
        run["equity"] = marked_equity
        curve = list(run.get("equity_curve", []))
        curve.append({"datetime": datetime, "equity": marked_equity})
        run["equity_curve"] = curve[-1000:]

    final_equity = _paper_marked_equity(run, close)
    run["status"] = "closed"
    run["closed_at"] = pd.Timestamp.utcnow().isoformat()
    run["closed_timestamp"] = timestamp
    run["closed_datetime"] = datetime
    run["final_equity"] = final_equity
    run["final_profit_loss"] = final_equity - float(run.get("initial_capital", 1000.0))
    run["position"] = "flat"
    run["entry_price"] = None
    run["entry_datetime"] = None
    run["entry_equity"] = None
    return run


def _close_paper_long_position(
    run: dict,
    close: float,
    timestamp: int,
    datetime: str,
    forced: bool = False,
) -> dict:
    fee_rate = float(run.get("fee_bps", 0.0)) / 10_000
    slippage_rate = float(run.get("slippage_bps", 0.0)) / 10_000
    equity = float(run.get("equity", run.get("initial_capital", 1000.0)))
    entry_price = float(run.get("entry_price") or close)
    entry_equity = float(run.get("entry_equity") or equity)
    exit_price = close * (1 - slippage_rate)
    gross_return = exit_price / entry_price if entry_price > 0 else 1.0
    final_equity = entry_equity * gross_return * (1 - fee_rate)
    trade_pl = final_equity - entry_equity
    trade_return_pct = (final_equity / entry_equity - 1) * 100 if entry_equity > 0 else 0.0
    trade_log = list(run.get("trade_log", []))
    trade_log.append(
        {
            "entry_datetime": run.get("entry_datetime"),
            "exit_datetime": datetime,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": trade_return_pct,
            "pl_usd": trade_pl,
            "equity_after": final_equity,
            "forced_exit": forced,
        }
    )
    run.update(
        {
            "position": "flat",
            "entry_price": None,
            "entry_datetime": None,
            "entry_equity": None,
            "equity": final_equity,
            "last_signal": -1 if forced else int(run.get("last_signal", -1)),
            "last_processed_timestamp": timestamp,
            "trade_log": trade_log[-100:],
            "trades": int(run.get("trades", 0)) + 1,
            "winning_trades": int(run.get("winning_trades", 0)) + (1 if trade_pl > 0 else 0),
            "losing_trades": int(run.get("losing_trades", 0)) + (1 if trade_pl < 0 else 0),
        }
    )
    marked_equity = _paper_marked_equity(run, close)
    peak_equity = max(float(run.get("peak_equity", marked_equity)), marked_equity)
    drawdown_pct = (marked_equity / peak_equity - 1) * 100 if peak_equity > 0 else 0.0
    run["peak_equity"] = peak_equity
    run["max_drawdown_pct"] = min(float(run.get("max_drawdown_pct", 0.0)), drawdown_pct)
    curve = list(run.get("equity_curve", []))
    curve.append({"datetime": datetime, "equity": marked_equity})
    run["equity_curve"] = curve[-1000:]
    return run


def _latest_market_point_for_run(run: dict) -> dict[str, float | int | str]:
    symbol = str(run.get("symbol", "BTC/USDT"))
    timeframe = str(run.get("timeframe", "4h"))
    try:
        live_data = provider.fetch_ohlcv(
            exchange="binance",
            symbol=symbol,
            timeframe=timeframe,
            limit=2,
        )
        storage.save(live_data, "binance", symbol, timeframe)
        if not live_data.empty:
            latest = live_data.iloc[-1]
            return {
                "close": float(latest["close"]),
                "timestamp": int(latest["timestamp"]),
                "datetime": str(latest["datetime"]),
            }
    except (MarketDataError, ValueError):
        pass

    data = storage.load("binance", symbol, timeframe)
    if not data.empty:
        latest = data.iloc[-1]
        return {
            "close": float(latest["close"]),
            "timestamp": int(latest["timestamp"]),
            "datetime": str(latest["datetime"]),
        }

    close = float(run.get("latest_price") or 0.0)
    return {
        "close": close,
        "timestamp": int(run.get("last_processed_timestamp") or run.get("started_timestamp") or 0),
        "datetime": str(run.get("latest_datetime") or run.get("started_datetime") or "-"),
    }


def _build_finalize_preview(run: dict) -> dict[str, str | bool]:
    run = _normalize_paper_run(run)
    latest = _latest_market_point_for_run(run)
    close = float(latest["close"])
    initial_capital = float(run.get("initial_capital", 1000.0))
    current_equity = _paper_marked_equity(run, close)
    forced_equity = current_equity
    forced_exit_price = close

    if run.get("position") == "long":
        fee_rate = float(run.get("fee_bps", 0.0)) / 10_000
        slippage_rate = float(run.get("slippage_bps", 0.0)) / 10_000
        entry_price = float(run.get("entry_price") or close)
        entry_equity = float(run.get("entry_equity") or current_equity)
        forced_exit_price = close * (1 - slippage_rate)
        gross_return = forced_exit_price / entry_price if entry_price > 0 else 1.0
        forced_equity = entry_equity * gross_return * (1 - fee_rate)

    forced_pl = forced_equity - initial_capital
    return {
        "has_open_position": run.get("position") == "long",
        "latest_price": _format_usd(close),
        "latest_datetime": str(latest["datetime"]),
        "current_equity": _format_usd(current_equity),
        "forced_exit_price": _format_usd(forced_exit_price),
        "final_equity": _format_usd(forced_equity),
        "final_profit_loss": _format_usd(forced_pl),
        "final_return_pct": _format_pct((forced_equity / initial_capital - 1) * 100),
        "profit_loss_class": "gain" if forced_pl > 0 else "loss" if forced_pl < 0 else "flat",
    }


def _paper_marked_equity(run: dict, latest_close: float | None) -> float:
    equity = float(run.get("equity", run.get("initial_capital", 1000.0)))
    if latest_close is None or run.get("position") != "long":
        return equity

    entry_price = float(run.get("entry_price") or latest_close)
    entry_equity = float(run.get("entry_equity") or equity)
    if entry_price <= 0:
        return equity

    return entry_equity * (latest_close / entry_price)


def _latest_close_for_run(run: dict) -> float | None:
    if run.get("latest_price") is not None:
        return float(run["latest_price"])

    data = storage.load(
        "binance",
        str(run.get("symbol", "BTC/USDT")),
        str(run.get("timeframe", "4h")),
    )
    if data.empty:
        return None

    return float(data.iloc[-1]["close"])


def _build_paper_summary(run: dict, latest_row) -> dict[str, str | int]:
    run = _normalize_paper_run(run)
    initial_capital = float(run.get("initial_capital", 1000.0))
    latest_close = float(latest_row["close"]) if latest_row is not None else _latest_close_for_run(run)
    marked_equity = _paper_marked_equity(run, latest_close)
    profit_loss = marked_equity - initial_capital
    entry_equity = float(run.get("entry_equity") or marked_equity)
    floating_pl = marked_equity - entry_equity if run.get("position") == "long" else 0.0
    trades = int(run.get("trades", 0))
    win_rate = int(run.get("winning_trades", 0)) / trades * 100 if trades else 0.0
    return {
        "mode": "Paper trading",
        "rows": len(run.get("equity_curve", [])),
        "strategy": _strategy_label(str(run.get("strategy", "none"))),
        "initial_capital": _format_usd(initial_capital),
        "final_equity": _format_usd(marked_equity),
        "profit_loss": _format_usd(profit_loss),
        "total_return_pct": _format_pct((marked_equity / initial_capital - 1) * 100),
        "buy_and_hold_return_pct": "0.00%",
        "max_drawdown_pct": _format_pct(float(run.get("max_drawdown_pct", 0.0))),
        "trades": trades,
        "win_rate_pct": _format_pct(win_rate),
        "profit_factor": _paper_profit_factor(run),
        "last_signal": _signal_label(int(run.get("last_signal", 0))),
        "position": _position_label(str(run.get("position", "flat"))),
        "entry_price": _format_optional_usd(run.get("entry_price")),
        "floating_pl": _format_usd(floating_pl),
        "latest_price": _format_optional_usd(latest_close),
        "last_update": _short_datetime(run.get("last_run_at")),
        "position_mode": _position_mode_label(str(run.get("position_mode", "compound"))),
        "fee_bps": f"{float(run.get('fee_bps', 0.0)):.2f}",
        "slippage_bps": f"{float(run.get('slippage_bps', 0.0)):.2f}",
    }


def _paper_profit_factor(run: dict) -> str:
    trades = run.get("trade_log", [])
    gross_profit = sum(float(item.get("pl_usd", 0.0)) for item in trades if float(item.get("pl_usd", 0.0)) > 0)
    gross_loss = abs(sum(float(item.get("pl_usd", 0.0)) for item in trades if float(item.get("pl_usd", 0.0)) < 0))
    if gross_loss == 0:
        return "inf" if gross_profit > 0 else "0.00"

    return _format_ratio(gross_profit / gross_loss)


def _format_paper_trade_log(run: dict) -> list[dict[str, str]]:
    return [
        {
            "entry_datetime": item.get("entry_datetime", "-"),
            "exit_datetime": item.get("exit_datetime", "-"),
            "entry_price": _format_price(float(item.get("entry_price", 0.0))),
            "exit_price": _format_price(float(item.get("exit_price", 0.0))),
            "return_pct": _format_pct(float(item.get("return_pct", 0.0))),
            "pl_usd": _format_usd(float(item.get("pl_usd", 0.0))),
            "equity_after": _format_usd(float(item.get("equity_after", 0.0))),
        }
        for item in run.get("trade_log", [])[-50:]
    ]


def _paper_equity_curve(run: dict) -> pd.DataFrame:
    curve = run.get("equity_curve", [])
    if not curve:
        return pd.DataFrame(columns=["datetime", "equity"])

    return pd.DataFrame(curve)


def _slice_irt_data(data: pd.DataFrame, started_timestamp: int) -> pd.DataFrame:
    if data.empty or "timestamp" not in data.columns:
        return data.copy()

    result = data[data["timestamp"] >= started_timestamp].copy()
    return result.reset_index(drop=True)


def _hide_pre_irt_signals(data: pd.DataFrame, started_timestamp: int) -> pd.DataFrame:
    result = data.copy()
    if "timestamp" in result.columns and "signal" in result.columns:
        result.loc[result["timestamp"] < started_timestamp, "signal"] = 0

    return result


def _irt_run_key(run: dict) -> str:
    return "|".join(
        [
            str(run.get("symbol", "")),
            str(run.get("timeframe", "")),
            str(run.get("strategy", "")),
        ]
    )


def _build_irt_href(
    symbol: str,
    timeframe: str,
    strategy: str,
    initial_capital: float,
    lookback: int,
    fee_bps: float,
    slippage_bps: float,
    position_mode: str,
) -> str:
    return (
        "/irt?"
        f"symbol={symbol.replace('/', '%2F')}&"
        f"timeframe={timeframe}&"
        f"strategy={strategy.replace(':', '%3A')}&"
        f"initial_capital={initial_capital:g}&"
        f"lookback={lookback}&"
        f"fee_bps={fee_bps:g}&"
        f"slippage_bps={slippage_bps:g}&"
        f"position_mode={position_mode}"
    )


def _build_finalize_href(symbol: str, timeframe: str, strategy: str) -> str:
    return (
        "/irt/finalize?"
        f"symbol={symbol.replace('/', '%2F')}&"
        f"timeframe={timeframe}&"
        f"strategy={strategy.replace(':', '%3A')}"
    )


def _symbol_icon(symbol: str) -> str:
    return symbol.split("/")[0][:3].upper()


def _signal_label(signal: int) -> str:
    if signal == 1:
        return "BUY"
    if signal == -1:
        return "SELL"

    return "WAIT"


def _format_active_time(start, end) -> str:
    if not start:
        return "-"

    delta = pd.to_datetime(end) - pd.to_datetime(start)
    days = max(0, int(delta.days))
    years = days // 365
    months = (days % 365) // 30
    if years > 0:
        return f"{years}y {months}m"
    if months > 0:
        return f"{months}m"

    return f"{days}d"


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


def _format_optional_usd(value) -> str:
    if value is None:
        return "-"

    try:
        return _format_usd(float(value))
    except (TypeError, ValueError):
        return "-"


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


def _short_datetime(value) -> str:
    if value in {None, ""}:
        return "-"

    return str(value).replace("T", " ")[:19]


def _position_label(position: str) -> str:
    if position == "long":
        return "Compra activa"

    return "Sin posicion"


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
                "module": str(getattr(row, "module", "-")),
                "allocation_pct": f"{float(getattr(row, 'allocation_pct', 100.0)):.1f}%",
                "reason": str(getattr(row, "reason", "-")),
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
    timeframes = SUPPORTED_TIMEFRAMES
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

        if strategy.startswith("master:"):
            master = master_storage.get(strategy.removeprefix("master:"))
            if master is None:
                continue
            candidate_master = MasterConfig(
                name=master.name,
                symbol=symbol,
                main_strategy=master.main_strategy,
                main_timeframe=candidate_timeframe,
                main_allocation_pct=master.main_allocation_pct,
                rebound_strategy=master.rebound_strategy,
                rebound_allocation_pct=master.rebound_allocation_pct,
                rebound_flags=master.rebound_flags,
                sideways_strategy=master.sideways_strategy,
                sideways_allocation_pct=master.sideways_allocation_pct,
                sideways_flags=master.sideways_flags,
                defensive_strategy=master.defensive_strategy,
                defensive_cash_pct=master.defensive_cash_pct,
                defensive_flags=master.defensive_flags,
            )
            result = run_master_system_backtest(
                data=filtered,
                master=candidate_master,
                initial_equity=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
            ).backtest
            split = _build_master_split_metrics(
                data=filtered,
                master=candidate_master,
                initial_capital=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
                validation_pct=validation_pct,
            )
        else:
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


def _build_walk_forward_results(
    data: pd.DataFrame,
    selected_master_config: MasterConfig | None,
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
    window_years: int = 2,
) -> list[dict[str, str | int]]:
    if data.empty:
        return []

    filtered = _filter_by_date_range(data, start_date, end_date)
    if filtered.empty:
        return []

    start = pd.to_datetime(filtered.iloc[0]["datetime"])
    end = pd.to_datetime(filtered.iloc[-1]["datetime"])
    if pd.isna(start) or pd.isna(end) or start >= end:
        return []

    rows: list[dict[str, str | int]] = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + pd.DateOffset(years=window_years), end)
        window_data = filtered[
            (pd.to_datetime(filtered["datetime"]) >= window_start)
            & (pd.to_datetime(filtered["datetime"]) < window_end)
        ].reset_index(drop=True)

        label = f"{window_start.date()} to {window_end.date()}"
        if len(window_data) < 20:
            rows.append(
                {
                    "window": label,
                    "status": "Too few rows",
                    "rows": len(window_data),
                    "strategy_gain": "-",
                    "buy_hold": "-",
                    "alpha": "-",
                    "drawdown": "-",
                    "trades": "-",
                    "profit_factor": "-",
                    "exposure": "-",
                    "avg_trade_days": "-",
                    "verdict": "-",
                }
            )
            window_start = window_end
            continue

        if selected_master_config is not None:
            result = run_master_system_backtest(
                data=window_data,
                master=selected_master_config,
                initial_equity=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
            ).backtest
        else:
            analyzed = _apply_strategy(
                data=add_default_moving_averages(window_data),
                strategy=strategy,
                fast_period=fast_period,
                slow_period=slow_period,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                bollinger_period=bollinger_period,
                bollinger_std=bollinger_std,
            )
            result = run_long_only_signal_backtest(
                data=analyzed,
                initial_equity=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                compound=compound,
                trade_mode=_trade_mode_for_strategy(strategy),
            )

        alpha = result.total_return_pct - result.buy_and_hold_return_pct
        exposure = _build_exposure_metrics(window_data, result)
        rows.append(
            {
                "window": label,
                "status": "OK",
                "rows": len(window_data),
                "strategy_gain": _format_pct(result.total_return_pct),
                "buy_hold": _format_pct(result.buy_and_hold_return_pct),
                "alpha": _format_pct(alpha),
                "drawdown": _format_pct(result.max_drawdown_pct),
                "trades": result.trades,
                "profit_factor": _format_ratio(result.profit_factor),
                "exposure": exposure["exposure_pct"],
                "avg_trade_days": exposure["avg_trade_days"],
                "verdict": _build_validation_verdict(
                    validation_return_pct=result.total_return_pct,
                    validation_alpha_pct=alpha,
                    validation_profit_factor=result.profit_factor,
                ),
            }
        )
        window_start = window_end

    return rows


def _build_strategy_comparison(
    exchange: str,
    symbol: str,
    timeframes: list[str],
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
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []

    for candidate_timeframe in timeframes:
        dataset = storage.load(exchange, symbol, candidate_timeframe)
        rows: list[dict[str, str | int]] = []
        group: dict[str, object] = {
            "timeframe": candidate_timeframe,
            "status": "Missing",
            "rows": 0,
            "from": "-",
            "to": "-",
            "strategies": rows,
        }

        if dataset.empty:
            groups.append(group)
            continue

        filtered = _filter_by_date_range(dataset, start_date, end_date)
        if filtered.empty:
            group["status"] = "No rows in date range"
            groups.append(group)
            continue

        group.update(
            {
                "status": "OK",
                "rows": len(filtered),
                "from": str(filtered.iloc[0]["datetime"]),
                "to": str(filtered.iloc[-1]["datetime"]),
            }
        )
        base_data = add_default_moving_averages(filtered)

        for option in _module_strategy_options():
            candidate_strategy = option["value"]
            if candidate_strategy == "none":
                continue

            try:
                analyzed = _apply_strategy(
                    data=base_data,
                    strategy=candidate_strategy,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    rsi_period=rsi_period,
                    rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought,
                    bollinger_period=bollinger_period,
                    bollinger_std=bollinger_std,
                )
                trade_mode = _trade_mode_for_strategy(candidate_strategy)
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
                    strategy=candidate_strategy,
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
                exposure = _build_exposure_metrics(filtered, result)
            except Exception as exc:
                rows.append(
                    {
                        "strategy": option["label"],
                        "status": f"Error: {exc}",
                        "trades": "-",
                        "strategy_gain": "-",
                        "buy_hold": "-",
                        "alpha": "-",
                        "drawdown": "-",
                        "profit_factor": "-",
                        "exposure": "-",
                        "avg_trade_days": "-",
                        "validation_gain": "-",
                        "validation_alpha": "-",
                        "validation_profit_factor": "-",
                        "verdict": "-",
                    }
                )
                continue

            rows.append(
                {
                    "strategy": option["label"],
                    "status": "OK",
                    "trades": result.trades,
                    "strategy_gain": _format_pct(result.total_return_pct),
                    "buy_hold": _format_pct(result.buy_and_hold_return_pct),
                    "alpha": _format_pct(
                        result.total_return_pct - result.buy_and_hold_return_pct
                    ),
                    "drawdown": _format_pct(result.max_drawdown_pct),
                    "profit_factor": _format_ratio(result.profit_factor),
                    "exposure": exposure["exposure_pct"],
                    "avg_trade_days": exposure["avg_trade_days"],
                    "validation_gain": split["validation_return_pct"],
                    "validation_alpha": split["validation_alpha_pct"],
                    "validation_profit_factor": split["validation_profit_factor"],
                    "verdict": split["validation_verdict"],
                }
            )

        def sort_key(row: dict[str, str | int]) -> float:
            value = str(row["validation_alpha"]).replace("%", "")
            try:
                return float(value)
            except ValueError:
                return -999999.0

        group["strategies"] = sorted(rows, key=sort_key, reverse=True)
        groups.append(group)

    return groups


def _build_exposure_metrics(data: pd.DataFrame, result: BacktestResult) -> dict[str, str]:
    if data.empty or result.trade_log.empty:
        return {"exposure_pct": "0.00%", "avg_trade_days": "0.00"}

    start = pd.to_datetime(data.iloc[0]["datetime"])
    end = pd.to_datetime(data.iloc[-1]["datetime"])
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return {"exposure_pct": "0.00%", "avg_trade_days": "0.00"}

    entries = pd.to_datetime(result.trade_log["entry_datetime"])
    exits = pd.to_datetime(result.trade_log["exit_datetime"])
    durations = (exits - entries).dt.total_seconds().clip(lower=0)
    exposure_pct = float(durations.sum() / total_seconds * 100)
    avg_trade_days = float(durations.mean() / 86_400) if not durations.empty else 0.0

    return {
        "exposure_pct": _format_pct(exposure_pct),
        "avg_trade_days": f"{avg_trade_days:.2f}",
    }


def _build_strategy_comparison_summary(
    groups: list[dict[str, object]],
) -> list[dict[str, str | int]]:
    by_strategy: dict[str, dict[str, object]] = {}

    for group in groups:
        timeframe = str(group["timeframe"])
        for row in group.get("strategies", []):
            if not isinstance(row, dict) or row.get("status") != "OK":
                continue

            strategy = str(row["strategy"])
            stats = by_strategy.setdefault(
                strategy,
                {
                    "strategy": strategy,
                    "tested": 0,
                    "promising": 0,
                    "validation_alphas": [],
                    "drawdowns": [],
                    "exposures": [],
                    "best_timeframe": "-",
                    "best_validation_alpha": -999999.0,
                    "best_gain": "-",
                    "best_profit_factor": "-",
                },
            )
            validation_alpha = _parse_pct(row.get("validation_alpha"))
            drawdown = _parse_pct(row.get("drawdown"))
            exposure = _parse_pct(row.get("exposure"))

            stats["tested"] = int(stats["tested"]) + 1
            if row.get("verdict") == "Promising":
                stats["promising"] = int(stats["promising"]) + 1
            if validation_alpha is not None:
                stats["validation_alphas"].append(validation_alpha)
                if validation_alpha > float(stats["best_validation_alpha"]):
                    stats["best_validation_alpha"] = validation_alpha
                    stats["best_timeframe"] = timeframe
                    stats["best_gain"] = str(row.get("validation_gain", "-"))
                    stats["best_profit_factor"] = str(
                        row.get("validation_profit_factor", "-")
                    )
            if drawdown is not None:
                stats["drawdowns"].append(drawdown)
            if exposure is not None:
                stats["exposures"].append(exposure)

    summary: list[dict[str, str | int]] = []
    for stats in by_strategy.values():
        validation_alphas = stats["validation_alphas"]
        drawdowns = stats["drawdowns"]
        exposures = stats["exposures"]
        avg_validation_alpha = (
            sum(validation_alphas) / len(validation_alphas)
            if validation_alphas
            else 0.0
        )
        worst_drawdown = min(drawdowns) if drawdowns else 0.0
        avg_exposure = sum(exposures) / len(exposures) if exposures else 0.0
        summary.append(
            {
                "strategy": str(stats["strategy"]),
                "best_timeframe": str(stats["best_timeframe"]),
                "tested": int(stats["tested"]),
                "promising": int(stats["promising"]),
                "avg_validation_alpha": _format_pct(avg_validation_alpha),
                "worst_drawdown": _format_pct(worst_drawdown),
                "avg_exposure": _format_pct(avg_exposure),
                "best_validation_gain": str(stats["best_gain"]),
                "best_validation_pf": str(stats["best_profit_factor"]),
            }
        )

    def sort_key(row: dict[str, str | int]) -> tuple[int, float]:
        return (
            int(row["promising"]),
            _parse_pct(row["avg_validation_alpha"]) or -999999.0,
        )

    return sorted(summary, key=sort_key, reverse=True)


def _parse_pct(value: object) -> float | None:
    try:
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _build_dataset_status(exchange: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_symbol in SUPPORTED_SYMBOLS:
        timeframe_rows = []
        total_rows = 0
        available = 0
        for candidate_timeframe in SUPPORTED_TIMEFRAMES:
            dataset = storage.load(exchange, candidate_symbol, candidate_timeframe)
            row_count = len(dataset)
            total_rows += row_count
            if row_count:
                available += 1
            timeframe_rows.append(
                {
                    "timeframe": candidate_timeframe,
                    "rows": row_count,
                    "status": "OK" if row_count else "Missing",
                }
            )

        rows.append(
            {
                "symbol": candidate_symbol,
                "available": available,
                "total": len(SUPPORTED_TIMEFRAMES),
                "total_rows": total_rows,
                "timeframes": timeframe_rows,
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
    if strategy.startswith("master:"):
        master = master_storage.get(strategy.removeprefix("master:"))
        if master is not None:
            return run_master_system_backtest(
                data=data,
                master=master,
                initial_equity=1.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                compound=True,
            ).analyzed_data
    if strategy.startswith("custom:"):
        return apply_custom_strategy(data=data, key=strategy)
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
    if strategy.startswith("master:"):
        return f"Master: {strategy.removeprefix('master:')}"
    if strategy.startswith("custom:"):
        custom_strategy = next(
            (item for item in list_custom_strategies() if item.key == strategy),
            None,
        )
        if custom_strategy is not None:
            return custom_strategy.label
        return strategy.removeprefix("custom:").replace("_", " ").title()
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
        _add_indicator_line(figure, plot_data, f"ema_{fast_period}", f"EMA {fast_period}", "#0891b2")
        _add_indicator_line(figure, plot_data, f"ema_{slow_period}", f"EMA {slow_period}", "#be123c")
        _add_indicator_line(figure, plot_data, f"bb_upper_{bollinger_period}", "BB Upper", "#64748b")
        _add_indicator_line(figure, plot_data, f"bb_lower_{bollinger_period}", "BB Lower", "#64748b")
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
