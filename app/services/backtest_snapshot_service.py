import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BacktestSnapshotService:
    """Persist lightweight backtest snapshots for later reopening."""

    def __init__(self, path: str | Path = "data/backtest_results.json") -> None:
        self.path = Path(path)

    def load(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        try:
            items = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        if not isinstance(items, list):
            return []

        items = list(reversed(items))
        return items[:limit] if limit is not None else items

    def save(
        self,
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = self.load(limit=None)
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
        self.path.write_text(
            json.dumps(items[-300:], indent=2),
            encoding="utf-8",
        )

