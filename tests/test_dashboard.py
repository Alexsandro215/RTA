import pandas as pd
from fastapi.testclient import TestClient
from typing import cast

import app.web.routes as routes
from app.data.csv_storage import CsvMarketDataStorage
from app.services.dataset_service import DatasetService
from main_api import app


def _sample_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000],
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1.0, 1.0, 1.0],
            "datetime": pd.to_datetime(
                [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000],
                unit="ms",
                utc=True,
            ),
        }
    )


def test_home_does_not_run_backtest_automatically(monkeypatch):
    routes._paper_worker_started = True

    def fail_load(*args, **kwargs):
        raise AssertionError("Home should not load the selected CSV until analysis is requested.")

    def fail_backtest(*args, **kwargs):
        raise AssertionError("Backtest should not run on initial dashboard load.")

    monkeypatch.setattr(routes.storage, "load", fail_load)
    monkeypatch.setattr(routes, "run_long_only_signal_backtest", fail_backtest)
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Save result" not in response.text


def test_save_result_button_only_appears_after_analysis(monkeypatch):
    routes._paper_worker_started = True
    monkeypatch.setattr(routes.storage, "load", lambda *args, **kwargs: _sample_ohlcv())

    client = TestClient(app)
    assert "Save result" not in client.get("/").text
    assert "Save result" in client.get("/?run_analysis=true&max_rows=50").text


def test_dataset_status_marks_corrupt_csv_invalid(tmp_path):
    storage = CsvMarketDataStorage(base_dir=tmp_path)
    path = storage.get_path("binance", "BTC/USDT", "1h")
    path.parent.mkdir(parents=True)
    path.write_text("bad,header\n1,2\n", encoding="utf-8")

    service = DatasetService(
        storage=storage,
        provider=routes.provider,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
    )
    status = service.build_status("binance")
    timeframe_rows = cast(list[dict[str, object]], status[0]["timeframes"])

    assert status[0]["available"] == 0
    assert timeframe_rows[0]["status"] == "Invalid header"


def test_dataset_status_endpoint_is_lazy(monkeypatch):
    routes._paper_worker_started = True
    called = {"status": False}

    def fake_status(exchange):
        called["status"] = True
        return []

    monkeypatch.setattr(routes.dataset_service, "build_status", fake_status)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert called["status"] is False

    response = client.get("/datasets/status?exchange=binance")
    assert response.status_code == 200
    assert called["status"] is True
