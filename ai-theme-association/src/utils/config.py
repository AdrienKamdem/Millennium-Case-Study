"""Config loading. Single source of truth for the universe, taxonomy and factor basket."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load config/<name>.yaml."""
    return yaml.safe_load((CONFIG / f"{name}.yaml").read_text())


def universe() -> list[dict]:
    return load("universe")["companies"]


def tickers() -> list[str]:
    return [c["ticker"] for c in universe()]


def window() -> tuple[str, str]:
    w = load("universe")["window"]
    return w["start"], w["end"]


def sector_map() -> dict[str, str]:
    return {c["ticker"]: c["sector_etf"] for c in universe()}


def sec_headers() -> dict[str, str]:
    """SEC blocks or throttles requests without a descriptive User-Agent."""
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is unset. Copy env.example to .env and fill it in.")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
