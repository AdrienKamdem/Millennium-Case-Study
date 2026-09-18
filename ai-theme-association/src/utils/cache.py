"""Disk cache for network responses, keyed by request hash.

Every external fetch goes through here. Two reasons:

1. `make all` becomes reproducible offline. A reviewer can regenerate every figure
   from the committed cache without a network connection or an API key.
2. A harvest that dies halfway costs nothing. Re-running resumes rather than
   restarting, which matters when the GDELT sweep is 405 sequential calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

EXT = {"json": ".json", "text": ".txt", "bytes": ".bin"}

# SEC throttles by answering 403, not 429, so both have to be retried. A cold EDGAR
# harvest is ~1,400 requests and will hit it at least once.
RETRY_STATUS = (403, 429, 500, 502, 503, 504)


def _key(url: str, params: dict | None) -> str:
    """Stable hash of a request. Params are sorted so dict ordering cannot change the key."""
    blob = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cached_get(
    source: str,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    *,
    parse: str = "json",
    min_interval: float = 0.0,
    force: bool = False,
    retries: int = 3,
) -> Any:
    """GET `url`, returning the cached response if one exists.

    Args:
        source: subdirectory under data/raw/ — one per upstream (gdelt, edgar, ...).
        parse: "json", "text", or "bytes".
        min_interval: seconds to sleep after a live fetch. Use to respect rate limits
            (SEC caps at 10 req/s; be far more conservative with GDELT, which
            publishes no limit and will simply stop answering).
        force: bypass the cache and refetch.
        retries: attempts on a throttle/5xx/timeout, with exponential backoff.

    Returns:
        Parsed response body. `bytes` hands back the raw body undecoded -- use it for
        HTML, where the charset is the parser's business and requests' guess is not
        good enough. It also sidesteps write_text()'s locale-dependent encoding, which
        can blow up partway through a filing on a non-UTF-8 machine.
    """
    if parse not in EXT:
        raise ValueError(f"parse must be one of {tuple(EXT)}, got {parse!r}")

    path = CACHE_ROOT / source / f"{_key(url, params)}{EXT[parse]}"

    if path.exists() and not force:
        if parse == "bytes":
            return path.read_bytes()
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if parse == "json" else text

    log.info("fetch %s %s", url, params or "")
    resp = _get_with_retries(url, params, headers, retries)

    path.parent.mkdir(parents=True, exist_ok=True)
    if parse == "bytes":
        path.write_bytes(resp.content)
    else:
        path.write_text(resp.text, encoding="utf-8")

    if min_interval:
        time.sleep(min_interval)

    if parse == "bytes":
        return resp.content
    return resp.json() if parse == "json" else resp.text


def _get_with_retries(url, params, headers, retries: int) -> requests.Response:
    """GET with exponential backoff on throttling, 5xx and timeouts. Other 4xx fail fast."""
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code in RETRY_STATUS and attempt < retries:
                log.warning("HTTP %s on %s -- retry %d/%d in %.0fs",
                            resp.status_code, url, attempt + 1, retries, delay)
            else:
                resp.raise_for_status()
                return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt >= retries:
                raise
            log.warning("%s on %s -- retry %d/%d in %.0fs",
                        type(e).__name__, url, attempt + 1, retries, delay)
        time.sleep(delay)
        delay *= 4
    raise RuntimeError(f"unreachable: retries exhausted for {url}")


def cached_call(source: str, key: str, fn: Callable[[], Any], *, force: bool = False) -> Any:
    """Cache an arbitrary expensive call (e.g. a BigQuery extract) under an explicit key."""
    path = CACHE_ROOT / source / f"{key}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result
