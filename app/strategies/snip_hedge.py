from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"


@dataclass(frozen=True)
class SnipHedgeConfig:
    trigger_up_bid: float = 0.51
    trigger_down_bid: float = 0.50
    limit_up: float = 0.50
    limit_down: float = 0.48


@dataclass(frozen=True)
class SnipHedgePreview:
    slug: str
    title: str
    interval: str
    up_label: str
    down_label: str
    up_bid: float | None
    down_bid: float | None
    up_ask: float | None
    down_ask: float | None
    condition_met: bool
    guaranteed_margin: float
    message: str


def preview_snip_hedge(url_or_slug: str, config: SnipHedgeConfig) -> SnipHedgePreview:
    """Preview the Polymarket SnipHedge condition without placing orders."""
    slug = extract_slug(url_or_slug)
    event = _fetch_event(slug)
    labels, token_ids = _get_binary_tokens(event)

    up_index = _find_label_index(labels, "up", fallback=0)
    down_index = _find_label_index(labels, "down", fallback=1)

    up_bid = _get_price(token_ids[up_index], side="sell")
    down_bid = _get_price(token_ids[down_index], side="sell")
    up_ask = _get_price(token_ids[up_index], side="buy")
    down_ask = _get_price(token_ids[down_index], side="buy")

    condition_met = (
        up_bid is not None
        and down_bid is not None
        and up_bid >= config.trigger_up_bid
        and down_bid >= config.trigger_down_bid
    )
    guaranteed_margin = 1.0 - (config.limit_up + config.limit_down)
    message = "Condition met" if condition_met else "Condition not met"

    return SnipHedgePreview(
        slug=slug,
        title=str(event.get("title", slug)),
        interval=_interval_from_slug(slug),
        up_label=labels[up_index],
        down_label=labels[down_index],
        up_bid=up_bid,
        down_bid=down_bid,
        up_ask=up_ask,
        down_ask=down_ask,
        condition_met=condition_met,
        guaranteed_margin=guaranteed_margin,
        message=message,
    )


def extract_slug(url_or_slug: str) -> str:
    raw = url_or_slug.strip()
    if not raw:
        raise ValueError("Polymarket URL or slug is required")

    if not raw.startswith("http"):
        return raw.split("/")[-1].split("?")[0]

    path_parts = [part for part in urlparse(raw).path.split("/") if part]
    for index, part in enumerate(path_parts):
        if part == "event" and index + 1 < len(path_parts):
            return path_parts[index + 1]

    if len(path_parts) == 1:
        return path_parts[0]

    raise ValueError(f"Could not extract event slug from: {url_or_slug}")


def _interval_from_slug(slug: str) -> str:
    match = re.search(r"btc-updown-(\d+)m-", slug, flags=re.IGNORECASE)
    if not match:
        return "unknown"

    return f"{int(match.group(1))}m"


def _fetch_event(slug: str) -> dict[str, Any]:
    response = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=20)
    response.raise_for_status()
    events = response.json()
    if not events:
        raise ValueError(f"No Polymarket event found for slug: {slug}")

    return events[0]


def _get_binary_tokens(event: dict[str, Any]) -> tuple[list[str], list[str]]:
    for market in event.get("markets") or []:
        if market.get("closed") or not market.get("active", True):
            continue

        token_labels, token_ids = _tokens_from_market_tokens(market)
        if len(token_ids) >= 2:
            return token_labels[:2], token_ids[:2]

        token_labels, token_ids = _tokens_from_clob_fields(market)
        if len(token_ids) >= 2:
            return token_labels[:2], token_ids[:2]

    raise ValueError("No active binary market with CLOB token ids was found")


def _tokens_from_market_tokens(market: dict[str, Any]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    token_ids: list[str] = []

    for token in market.get("tokens") or []:
        if not isinstance(token, dict):
            continue

        labels.append(str(token.get("outcome", "?")))
        token_ids.append(str(token.get("token_id") or token.get("tokenId") or ""))

    return labels, [token_id for token_id in token_ids if token_id]


def _tokens_from_clob_fields(market: dict[str, Any]) -> tuple[list[str], list[str]]:
    clob_token_ids = _parse_json_field(market.get("clobTokenIds", []))
    outcomes = _parse_json_field(market.get("outcomes", []))

    if not isinstance(clob_token_ids, list):
        return [], []

    labels = []
    for index in range(len(clob_token_ids)):
        if isinstance(outcomes, list) and index < len(outcomes):
            labels.append(str(outcomes[index]))
        else:
            labels.append(f"[{index}]")

    return labels, [str(token_id) for token_id in clob_token_ids if token_id]


def _parse_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _find_label_index(labels: list[str], needle: str, fallback: int) -> int:
    return next(
        (index for index, label in enumerate(labels) if needle in label.lower()),
        fallback,
    )


def _get_price(token_id: str, side: str) -> float | None:
    try:
        response = requests.get(
            f"{CLOB_HOST}/price",
            params={"token_id": token_id, "side": side},
            timeout=15,
        )
        if response.status_code != 200:
            return None

        value = response.json().get("price")
        return None if value is None else float(value)
    except (requests.RequestException, TypeError, ValueError):
        return None
