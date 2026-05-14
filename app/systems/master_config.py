from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class MasterConfig:
    name: str
    symbol: str
    main_strategy: str
    main_timeframe: str
    main_allocation_pct: float
    rebound_strategy: str
    rebound_allocation_pct: float
    rebound_flags: list[str]
    sideways_strategy: str
    sideways_allocation_pct: float
    sideways_flags: list[str]
    defensive_strategy: str
    defensive_cash_pct: float
    defensive_flags: list[str]


class MasterConfigStorage:
    def __init__(self, path: str | Path = "data/masters.json") -> None:
        self.path = Path(path)

    def list(self) -> list[MasterConfig]:
        if not self.path.exists():
            return []

        raw_items = json.loads(self.path.read_text(encoding="utf-8"))
        return [MasterConfig(**item) for item in raw_items]

    def save(self, config: MasterConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        configs = self.list()
        next_configs = [item for item in configs if item.name != config.name]
        next_configs.append(config)
        next_configs = sorted(next_configs, key=lambda item: item.name.lower())
        self.path.write_text(
            json.dumps([asdict(item) for item in next_configs], indent=2),
            encoding="utf-8",
        )

    def get(self, name: str) -> MasterConfig | None:
        for config in self.list():
            if config.name == name:
                return config

        return None
