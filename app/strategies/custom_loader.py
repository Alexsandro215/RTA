from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
from types import ModuleType
from typing import Callable, cast

import pandas as pd


CUSTOM_STRATEGY_DIR = Path("data/user_strategies")


@dataclass(frozen=True)
class CustomStrategy:
    key: str
    label: str
    path: Path


def save_custom_strategy(filename: str, content: bytes) -> CustomStrategy:
    if not filename.lower().endswith(".py"):
        raise ValueError("Strategy file must be a .py file")

    stem = Path(filename).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    if not safe_stem:
        raise ValueError("Strategy filename must include letters or numbers")

    CUSTOM_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    path = CUSTOM_STRATEGY_DIR / f"{safe_stem}.py"
    path.write_bytes(content)

    strategy = CustomStrategy(
        key=f"custom:{safe_stem}",
        label=_label_from_stem(safe_stem),
        path=path,
    )
    _load_apply_function(strategy)
    return strategy


def list_custom_strategies() -> list[CustomStrategy]:
    if not CUSTOM_STRATEGY_DIR.exists():
        return []

    strategies = []
    for path in sorted(CUSTOM_STRATEGY_DIR.glob("*.py")):
        strategies.append(
            CustomStrategy(
                key=f"custom:{path.stem}",
                label=_label_from_stem(path.stem),
                path=path,
            )
        )
    return strategies


def get_custom_strategy(key: str) -> CustomStrategy | None:
    if not key.startswith("custom:"):
        return None

    stem = key.removeprefix("custom:")
    path = CUSTOM_STRATEGY_DIR / f"{stem}.py"
    if not path.exists():
        return None

    return CustomStrategy(key=key, label=_label_from_stem(stem), path=path)


def delete_custom_strategy(key: str) -> CustomStrategy:
    strategy = get_custom_strategy(key)
    if strategy is None:
        raise ValueError(f"Custom strategy '{key}' was not found")

    resolved_dir = CUSTOM_STRATEGY_DIR.resolve()
    resolved_path = strategy.path.resolve()
    if resolved_dir not in resolved_path.parents:
        raise ValueError("Strategy path is outside the custom strategy directory")

    resolved_path.unlink()
    return strategy


def apply_custom_strategy(data: pd.DataFrame, key: str) -> pd.DataFrame:
    strategy = get_custom_strategy(key)
    if strategy is None:
        raise ValueError(f"Custom strategy '{key}' was not found")

    apply_strategy = _load_apply_function(strategy)
    result = apply_strategy(data.copy())
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Custom apply_strategy must return a pandas DataFrame")
    if "signal" not in result.columns:
        raise ValueError("Custom strategy must return a DataFrame with a signal column")

    result = result.copy()
    result["signal"] = result["signal"].fillna(0).astype(int)
    if "strategy" not in result.columns:
        result["strategy"] = strategy.label
    return result


def get_custom_strategy_metadata(key: str) -> dict[str, object]:
    strategy = get_custom_strategy(key)
    if strategy is None:
        return {}

    module = _load_module(strategy)
    metadata = getattr(module, "STRATEGY_METADATA", {})
    return metadata if isinstance(metadata, dict) else {}


def _load_apply_function(strategy: CustomStrategy) -> Callable[[pd.DataFrame], pd.DataFrame]:
    module = _load_module(strategy)
    apply_strategy = getattr(module, "apply_strategy", None)
    if not callable(apply_strategy):
        raise ValueError("Strategy file must define apply_strategy(data)")

    return cast(Callable[[pd.DataFrame], pd.DataFrame], apply_strategy)


def _load_module(strategy: CustomStrategy) -> ModuleType:
    module_name = f"user_strategy_{strategy.path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, strategy.path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load strategy file {strategy.path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _label_from_stem(stem: str) -> str:
    return stem.replace("_", " ").title()
