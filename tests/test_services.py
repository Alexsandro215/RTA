import pandas as pd
from typing import cast

from app.data.csv_storage import CsvMarketDataStorage
from app.data.ccxt_provider import CCXTMarketDataProvider
from app.services.backtest_analysis_service import (
    BacktestAnalysisService,
    BacktestAnalysisSettings,
)
from app.services.backtest_snapshot_service import BacktestSnapshotService
from app.services.background_job_service import BackgroundJob, BackgroundJobService
from app.services.paper_trading_service import PaperTradingService
from app.systems.master_config import MasterConfigStorage


def _sample_ohlcv() -> pd.DataFrame:
    timestamps = [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1.0, 1.0, 1.0],
            "datetime": pd.to_datetime(timestamps, unit="ms", utc=True),
        }
    )


def test_background_job_completes():
    jobs = BackgroundJobService()
    job = jobs.start("sample", lambda update: {"ok": True})

    while cast(BackgroundJob, jobs.get(job.id)).status not in {"completed", "failed"}:
        pass

    finished = cast(BackgroundJob, jobs.get(job.id))
    assert finished.status == "completed"
    assert finished.result == {"ok": True}


def test_paper_service_processes_entry_and_exit(tmp_path):
    service = PaperTradingService(
        storage=CsvMarketDataStorage(base_dir=tmp_path / "raw"),
        provider=cast(CCXTMarketDataProvider, None),
        master_storage=MasterConfigStorage(path=tmp_path / "masters.json"),
        runs_path=tmp_path / "runs.json",
        closed_runs_path=tmp_path / "closed.json",
    )
    run = service.normalize_run({"initial_capital": 1000.0})

    buy = type("Row", (), {"signal": 1, "close": 100.0, "timestamp": 1, "datetime": "t1"})
    sell = type("Row", (), {"signal": -1, "close": 110.0, "timestamp": 2, "datetime": "t2"})
    run = service.process_paper_row(run, buy)
    run = service.process_paper_row(run, sell)

    assert run["position"] == "flat"
    assert run["trades"] == 1
    assert run["equity"] > 1000.0


def test_snapshot_service_roundtrip(tmp_path):
    service = BacktestSnapshotService(path=tmp_path / "snapshots.json")
    service.save(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="4h",
        strategy="Demo",
        strategy_key="demo",
        start_date="2024-01-01",
        end_date="",
        initial_capital=1000.0,
        position_mode="compound",
        fee_bps=10.0,
        slippage_bps=2.0,
        summary={
            "total_return_pct": "1.00%",
            "buy_and_hold_return_pct": "0.50%",
            "alpha_vs_buy_hold_pct": "0.50%",
            "max_drawdown_pct": "-1.00%",
            "trades": 1,
            "profit_factor": "2.00",
            "validation_return_pct": "1.00%",
            "validation_verdict": "Promising",
        },
    )

    rows = service.load()
    assert rows[0]["symbol"] == "BTC/USDT"
    assert rows[0]["strategy"] == "Demo"


def test_analysis_service_writes_persistent_cache(tmp_path):
    storage = CsvMarketDataStorage(base_dir=tmp_path / "raw")
    storage.save(_sample_ohlcv(), "binance", "BTC/USDT", "1h")
    service = BacktestAnalysisService(
        storage=storage,
        master_storage=MasterConfigStorage(path=tmp_path / "masters.json"),
        supported_timeframes=["1h"],
        cache_dir=tmp_path / "cache",
    )
    settings = BacktestAnalysisSettings(
        start_date="",
        end_date="",
        fast_period=20,
        slow_period=50,
        rsi_period=14,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        bollinger_period=40,
        bollinger_std=2.2,
        initial_capital=1000.0,
        fee_bps=10.0,
        slippage_bps=2.0,
        compound=True,
        validation_pct=30.0,
    )

    rows = service.build_strategy_comparison(
        exchange="binance",
        symbol="BTC/USDT",
        timeframes=["1h"],
        settings=settings,
    )

    assert rows[0]["status"] == "OK"
    assert list((tmp_path / "cache").glob("*.json"))
