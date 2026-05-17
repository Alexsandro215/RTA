from pathlib import Path
import threading
from typing import cast

import pandas as pd
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from app.backtesting.simple_backtester import run_long_only_signal_backtest
from app.backtesting.system_backtester import run_master_system_backtest
from app.data.ccxt_provider import CCXTMarketDataProvider
from app.data.csv_storage import CsvMarketDataStorage
from app.data.market_data_provider import MarketDataError
from app.indicators.moving_averages import add_default_moving_averages
from app.strategies.custom_loader import (
    delete_custom_strategy,
    get_custom_strategy_metadata,
    list_custom_strategies,
    save_custom_strategy,
)
from app.strategies.snip_hedge import (
    SnipHedgeConfig,
    preview_snip_hedge,
)
from app.services.backtest_snapshot_service import BacktestSnapshotService
from app.services.backtest_analysis_service import (
    BacktestAnalysisService,
    BacktestAnalysisSettings,
)
from app.services.background_job_service import BackgroundJobService
from app.services.chart_service import build_candlestick_chart, build_equity_curve_chart
from app.services.dataset_service import DatasetService
from app.services.paper_trading_service import PaperTradingService
from app.services.strategy_registry import (
    apply_strategy,
    base_strategy_options,
    module_strategy_options,
    strategy_label,
    strategy_options,
    trade_mode_for_strategy,
)
from app.systems.master_config import MasterConfigStorage


router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")
storage = CsvMarketDataStorage()
provider = CCXTMarketDataProvider()
master_storage = MasterConfigStorage()
snapshot_service = BacktestSnapshotService()
job_service = BackgroundJobService()
IRT_RUNS_PATH = Path("data/irt_runs.json")
IRT_CLOSED_RUNS_PATH = Path("data/irt_closed_runs.json")
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
dataset_service = DatasetService(
    storage=storage,
    provider=provider,
    symbols=SUPPORTED_SYMBOLS,
    timeframes=SUPPORTED_TIMEFRAMES,
)
analysis_service = BacktestAnalysisService(
    storage=storage,
    master_storage=master_storage,
    supported_timeframes=SUPPORTED_TIMEFRAMES,
)
paper_service = PaperTradingService(
    storage=storage,
    provider=provider,
    master_storage=master_storage,
    runs_path=IRT_RUNS_PATH,
    closed_runs_path=IRT_CLOSED_RUNS_PATH,
    poll_seconds=PAPER_POLL_SECONDS,
)


def start_paper_trading_worker() -> None:
    global _paper_worker_started
    if _paper_worker_started:
        return

    _paper_worker_started = True
    worker = threading.Thread(target=paper_service.trading_loop, daemon=True)
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
            if refreshed_run is None:
                raise ValueError("Paper run could not be refreshed.")
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
) -> Response:
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
) -> Response:
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


@router.get("/datasets/status")
def get_dataset_status(exchange: str = Query(default="binance")) -> JSONResponse:
    return JSONResponse(
        {
            "exchange": exchange,
            "rows": dataset_service.build_status(exchange),
        }
    )


@router.get("/saved-results")
def get_saved_results(limit: int = Query(default=12, ge=1, le=100)) -> JSONResponse:
    return JSONResponse({"rows": snapshot_service.load(limit=limit)})


@router.get("/custom-strategies")
def get_custom_strategies() -> JSONResponse:
    rows = [
        {"key": item.key, "label": item.label, "path": str(item.path)}
        for item in list_custom_strategies()
    ]
    return JSONResponse({"rows": rows})


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    return JSONResponse(job_service.serialize(job_service.get(job_id)))


@router.post("/jobs/strategy-comparison")
def start_strategy_comparison_job(
    exchange: str = Form(default="binance"),
    symbol: str = Form(default="BTC/USDT"),
    start_date: str = Form(default=""),
    end_date: str = Form(default=""),
    fast_period: int = Form(default=20),
    slow_period: int = Form(default=50),
    rsi_period: int = Form(default=14),
    rsi_oversold: float = Form(default=30.0),
    rsi_overbought: float = Form(default=70.0),
    bollinger_period: int = Form(default=40),
    bollinger_std: float = Form(default=2.2),
    fee_bps: float = Form(default=10.0),
    slippage_bps: float = Form(default=2.0),
    initial_capital: float = Form(default=1000.0),
    position_mode: str = Form(default="compound"),
    validation_pct: float = Form(default=30.0),
) -> JSONResponse:
    settings = BacktestAnalysisSettings(
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
        compound=position_mode != "fixed",
        validation_pct=validation_pct,
    )

    def task(update):
        update(5, "Comparing all strategies across all timeframes")
        rows = analysis_service.build_strategy_comparison(
            exchange=exchange,
            symbol=symbol,
            timeframes=SUPPORTED_TIMEFRAMES,
            settings=settings,
        )
        update(90, "Building summary")
        summary = analysis_service.build_strategy_comparison_summary(rows)
        return {"rows": rows, "summary": summary}

    job = job_service.start("Compare all TFs", task)
    return JSONResponse({"job_id": job.id})


@router.post("/jobs/download-all")
def start_download_all_job(
    exchange: str = Form(default="binance"),
    fetch_since: str = Form(default="2024-01-01"),
    fetch_until: str = Form(default=""),
    fetch_batch_limit: int = Form(default=1000),
    fetch_max_batches: int = Form(default=0),
) -> JSONResponse:
    def task(update):
        update(5, "Downloading listed datasets")
        messages, errors = dataset_service.download_all_listed(
            exchange=exchange,
            since=fetch_since,
            until=fetch_until or None,
            limit_per_request=fetch_batch_limit,
            max_batches=fetch_max_batches or None,
        )
        return {"messages": messages, "errors": errors}

    job = job_service.start("Download all listed", task)
    return JSONResponse({"job_id": job.id})


@router.post("/datasets/redownload")
def redownload_dataset(
    exchange: str = Form(default="binance"),
    symbol: str = Form(...),
    timeframe: str = Form(...),
    fetch_since: str = Form(default="2024-01-01"),
    fetch_until: str = Form(default=""),
    fetch_batch_limit: int = Form(default=1000),
    fetch_max_batches: int = Form(default=0),
) -> JSONResponse:
    message, error = dataset_service.redownload_dataset(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        since=fetch_since,
        until=fetch_until or None,
        limit_per_request=fetch_batch_limit,
        max_batches=fetch_max_batches or None,
    )
    return JSONResponse({"ok": error is None, "message": message, "error": error})


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
    strategy_job_id: str = Query(default=""),
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

    strategy_context_message = ""
    if strategy.startswith("custom:"):
        metadata = get_custom_strategy_metadata(strategy)
        supported_symbols = metadata.get("symbols")
        supported_timeframes = metadata.get("timeframes")
        if (
            isinstance(supported_symbols, list)
            and supported_symbols
            and symbol not in supported_symbols
        ):
            next_symbol = str(supported_symbols[0])
            strategy_context_message = (
                f"Strategy is specialized for {next_symbol}; switched symbol."
            )
            symbol = next_symbol
        if (
            isinstance(supported_timeframes, list)
            and supported_timeframes
            and timeframe not in supported_timeframes
        ):
            next_timeframe = str(supported_timeframes[0])
            strategy_context_message = (
                f"Strategy is specialized for {next_timeframe}; switched timeframe."
            )
            timeframe = next_timeframe

    path = storage.get_path(exchange, symbol, timeframe)
    data_message = ""
    data_error = ""
    data_messages = []
    data_errors = []
    if _is_enabled(fetch_all_history):
        data_messages, data_errors = dataset_service.download_all_listed(
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

    if strategy_context_message:
        data_message = (
            f"{strategy_context_message} "
            + (data_message if data_message else "")
        ).strip()

    compound = position_mode != "fixed"
    should_run_analysis = _is_enabled(run_analysis) or _is_enabled(save_result)
    should_load_selected_data = should_run_analysis or _is_enabled(walk_forward)
    data = (
        storage.load(exchange, symbol, timeframe)
        if should_load_selected_data
        else pd.DataFrame()
    )
    analysis_settings = BacktestAnalysisSettings(
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

    if should_run_analysis and not data.empty:
        filtered_data = cast(pd.DataFrame, _filter_by_date_range(data, start_date, end_date))
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
        strategy_name = (
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
            "strategy": strategy_name,
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
            strategy_name=strategy_name,
        )

    if _is_enabled(compare_timeframes):
        compare_results = analysis_service.build_timeframe_comparison(
            exchange=exchange,
            symbol=symbol,
            strategy=strategy,
            settings=analysis_settings,
        )

    should_compare_strategies = (
        _is_enabled(compare_strategies) or strategy_compare_scope == "all"
    )
    if should_compare_strategies:
        if strategy_job_id:
            job = job_service.get(strategy_job_id)
            if job is not None and job.status == "completed" and isinstance(job.result, dict):
                strategy_compare_results = job.result.get("rows", [])
                strategy_compare_summary = job.result.get("summary", [])
        else:
            strategy_compare_timeframes = (
                SUPPORTED_TIMEFRAMES
                if strategy_compare_scope == "all"
                else [timeframe]
            )
            strategy_compare_results = analysis_service.build_strategy_comparison(
                exchange=exchange,
                symbol=symbol,
                timeframes=strategy_compare_timeframes,
                settings=analysis_settings,
            )
            strategy_compare_summary = analysis_service.build_strategy_comparison_summary(
                strategy_compare_results
            )

    if _is_enabled(walk_forward):
        walk_forward_results = analysis_service.build_walk_forward_results(
            data=data,
            selected_master_config=selected_master_config,
            strategy=strategy,
            settings=analysis_settings,
        )

    if _is_enabled(save_result) and summary["rows"]:
        snapshot_service.save(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            strategy=str(summary["strategy"]),
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
            "saved_results": [],
            "strategy_options": _strategy_options(),
            "custom_strategies": [],
            "supported_symbols": SUPPORTED_SYMBOLS,
            "supported_timeframes": SUPPORTED_TIMEFRAMES,
            "upload_message": upload_message,
            "upload_error": upload_error,
            "strategy_job_id": strategy_job_id,
        },
    )


def _is_enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _strategy_options() -> list[dict[str, str]]:
    return strategy_options(master_storage)


def _module_strategy_options() -> list[dict[str, str]]:
    return module_strategy_options()


def _base_strategy_options() -> list[dict[str, str]]:
    return base_strategy_options()


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
    return paper_service.load_runs()


def _get_irt_run(run: dict) -> dict | None:
    return paper_service.get_run(run)


def _save_irt_run(run: dict) -> dict:
    return paper_service.save_run(run)


def _replace_irt_run(updated_run: dict) -> dict:
    return paper_service.replace_run(updated_run)


def _delete_irt_run(run: dict) -> None:
    paper_service.delete_run(run)


def _load_closed_irt_runs() -> list[dict]:
    return paper_service.load_closed_runs()


def _save_closed_irt_run(run: dict) -> None:
    paper_service.save_closed_run(run)


def _normalize_paper_run(run: dict) -> dict:
    return paper_service.normalize_run(run)


def _paper_trading_loop() -> None:
    paper_service.trading_loop()


def _refresh_paper_runs() -> None:
    paper_service.refresh_runs()


def _refresh_single_paper_run(run: dict | None) -> dict | None:
    return paper_service.refresh_single_run(run)


def _is_run_enabled(run: dict) -> bool:
    return paper_service.is_run_enabled(run)


def _process_paper_row(run: dict, row) -> dict:
    return paper_service.process_paper_row(run, row)


def _finalize_paper_run(run: dict) -> dict:
    return paper_service.finalize_run(run)


def _close_paper_long_position(
    run: dict,
    close: float,
    timestamp: int,
    exit_datetime: str,
    forced: bool = False,
) -> dict:
    return paper_service.close_paper_long_position(
        run=run,
        close=close,
        timestamp=timestamp,
        exit_datetime=exit_datetime,
        forced=forced,
    )


def _latest_market_point_for_run(run: dict) -> dict[str, float | int | str]:
    return paper_service.latest_market_point(run)


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
    return paper_service.marked_equity(run, latest_close)


def _latest_close_for_run(run: dict) -> float | None:
    return paper_service.latest_close(run)


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
    return paper_service.equity_curve(run)


def _slice_irt_data(data: pd.DataFrame, started_timestamp: int) -> pd.DataFrame:
    return paper_service.slice_irt_data(data, started_timestamp)


def _hide_pre_irt_signals(data: pd.DataFrame, started_timestamp: int) -> pd.DataFrame:
    return paper_service.hide_pre_irt_signals(data, started_timestamp)


def _irt_run_key(run: dict) -> str:
    return paper_service.run_key(run)


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
    return trade_mode_for_strategy(strategy)


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
    return apply_strategy(
        data=data,
        strategy=strategy,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        bollinger_period=bollinger_period,
        bollinger_std=bollinger_std,
        master_storage=master_storage,
    )


def _strategy_label(
    strategy: str,
    fast_period: int = 20,
    slow_period: int = 50,
    rsi_period: int = 14,
    bollinger_period: int = 40,
    bollinger_std: float = 2.2,
) -> str:
    return strategy_label(
        strategy=strategy,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        bollinger_period=bollinger_period,
        bollinger_std=bollinger_std,
    )


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
    return build_candlestick_chart(
        data=data,
        symbol=symbol,
        timeframe=timeframe,
        show_sma_20=show_sma_20,
        show_sma_50=show_sma_50,
        show_ema_200=show_ema_200,
        strategy=strategy,
        fast_period=fast_period,
        slow_period=slow_period,
        bollinger_period=bollinger_period,
    )


def _build_equity_curve_chart(equity_curve, strategy_name: str) -> str:
    return build_equity_curve_chart(equity_curve, strategy_name)
