"""Alpha Vantage NEWS_SENTIMENT harvest.

Chosen after GDELT returned sustained HTTP 429s. Advantages worth stating:
  - TICKER-TAGGED by the provider, which removes the entity-disambiguation risk
    that made INTC ("intel" the noun), MU ("micron" the unit) and WM ("waste
    management" the industry phrase) the three most dangerous names here.
  - TITLE + SUMMARY, not headline alone, so aspect sentiment has context.

TWO MODES

  batched (default)  One request per QUARTER carrying all 9 tickers, then split
                     by each article's per-ticker relevance_score and keep the
                     top N per ticker. 15 requests total -- inside the free tier.

  per-ticker         One request per ticker-quarter. 135 requests, cleaner
                     per-cell coverage, but needs a paid tier (free is 25/day).

WHAT THIS MEASURES

  With a fixed N kept per cell, the salience denominator is constant by
  construction, so salience = "AI share of the top-N most relevant stories"
  rather than "AI share of all coverage". That is a legitimate and comparable
  measure -- relevance is scored against the TICKER, not against AI, so it does
  not systematically favour AI stories -- but it is a different construct from a
  true coverage share. Say so in the write-up.

  Caveat to keep honest: at N=10 the share is noisy (3/10 vs 4/10 is a 10-point
  swing). Empirical-Bayes shrinkage in index/salience.py absorbs some of that.
  Raising --keep to 25 costs no extra quota.

    python -m src.corpus.alphavantage --check      # 0 requests, shows the plan
    python -m src.corpus.alphavantage --sample     # 1 request
    python -m src.corpus.alphavantage              # 15 requests, batched
    python -m src.corpus.alphavantage --per-ticker # 135 requests, needs premium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils import config, paths
from src.utils import paths

ENDPOINT = "https://www.alphavantage.co/query"
FETCH_LIMIT = 1000        # costs the same as limit=10; only the response size differs
KEEP_PER_CELL = 10        # how many to retain per ticker-quarter
MIN_INTERVAL = 15.0       # free tier ~5 req/min; 15s = 4/min, safely under
CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "alphavantage"

QSTART = {1: "0101", 2: "0401", 3: "0701", 4: "1001"}
QEND = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}


class QuotaExhausted(RuntimeError):
    pass


def quarters(start_year=2023, end_year=2026, end_q=3):
    """2023Q1 .. 2026Q3 inclusive = 15 cells."""
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            if y == end_year and q > end_q:
                return
            yield y, q


def _cache_path(tag: str, year: int, q: int, sort: str) -> Path:
    # Deliberately excludes the API key, so rotating the key does not invalidate
    # the cache and force you to re-spend quota.
    h = hashlib.sha256(f"{tag}-{year}Q{q}-{FETCH_LIMIT}-{sort}".encode()).hexdigest()[:12]
    return CACHE / f"{tag}_{year}Q{q}_{h}.json"


def _request(tickers: list[str], year: int, q: int, api_key: str,
             sort: str, force: bool = False) -> dict:
    tag = "+".join(tickers) if len(tickers) <= 3 else f"ALL{len(tickers)}"
    path = _cache_path(tag, year, q, sort)
    if path.exists() and not force:
        return json.loads(path.read_text())

    r = requests.get(ENDPOINT, timeout=60, params={
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "time_from": f"{year}{QSTART[q]}T0000",
        "time_to": f"{year}{QEND[q]}T2359",
        "limit": FETCH_LIMIT,
        "sort": sort,
        "apikey": api_key,
    })
    r.raise_for_status()
    payload = r.json()

    # Quota / tier messages arrive as HTTP 200 with a different shape.
    for key in ("Note", "Information", "Error Message"):
        if key in payload:
            raise QuotaExhausted(f"{key}: {str(payload[key])[:300]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    time.sleep(MIN_INTERVAL)
    return payload


def _rows_for(payload: dict, ticker: str, year: int, q: int) -> list[dict]:
    """Extract this ticker's articles, carrying ITS OWN relevance score.

    An article can be tagged to several tickers with different relevance; always
    pull the entry matching the ticker we are attributing the document to.
    """
    out = []
    for a in payload.get("feed", []) or []:
        ts = next((t for t in a.get("ticker_sentiment", [])
                   if t.get("ticker") == ticker), None)
        if ts is None:
            continue
        url = a.get("url", "")
        out.append({
            # (article, ticker) is the unit -- a URL-only hash collides when one story
            # is tagged to several tickers, and the Batch API rejects duplicate ids.
            "doc_id": hashlib.sha256(f"{url}|{ticker}".encode()).hexdigest()[:16],
            "article_id": hashlib.sha256(url.encode()).hexdigest()[:16],
            "ticker": ticker,
            "period": f"{year}Q{q}",
            "date": pd.to_datetime(a.get("time_published", ""),
                                   format="%Y%m%dT%H%M%S", errors="coerce", utc=True),
            "title": (a.get("title") or "").strip(),
            "summary": (a.get("summary") or "").strip(),
            "url": url,
            "domain": a.get("source_domain", ""),
            "source": a.get("source", ""),
            "source_category": "news",
            # Provider's own scores: free second opinions to compare your LLM
            # valence against. Divergence between them belongs in the appendix.
            "av_relevance": pd.to_numeric(ts.get("relevance_score"), errors="coerce"),
            "av_ticker_sentiment": pd.to_numeric(ts.get("ticker_sentiment_score"), errors="coerce"),
            "av_overall_sentiment": pd.to_numeric(a.get("overall_sentiment_score"), errors="coerce"),
        })
    return out


def harvest(sample=False, per_ticker=False, keep=KEEP_PER_CELL,
            sort="RELEVANCE", force=False) -> pd.DataFrame:
    api_key = os.environ.get("AV_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("AV_KEY unset. Add to .env then: set -a && source .env && set +a")
        return pd.DataFrame()

    tickers = config.tickers()
    cells = list(quarters())
    if sample:
        cells = cells[:1]
        print(f"SAMPLE: {cells[0][0]}Q{cells[0][1]}, {len(tickers)} tickers, 1 request")

    rows, audit = [], []
    try:
        for (year, q) in cells:
            groups = [[t] for t in tickers] if per_ticker else [tickers]
            for grp in groups:
                payload = _request(grp, year, q, api_key, sort, force)
                n_feed = len(payload.get("feed", []) or [])
                for t in grp:
                    got = _rows_for(payload, t, year, q)
                    # Rank by THIS ticker's relevance, then keep the top N.
                    got.sort(key=lambda r: (r["av_relevance"] is None, -(r["av_relevance"] or 0)))
                    kept = got[:keep]
                    rows.extend(kept)
                    audit.append({"ticker": t, "period": f"{year}Q{q}",
                                  "n_available": len(got), "n_kept": len(kept),
                                  "short": len(got) < keep})
                flag = "  <-- FEED AT CAP" if n_feed >= FETCH_LIMIT else ""
                print(f"  {year}Q{q}  feed={n_feed:4}  kept={sum(a['n_kept'] for a in audit if a['period']==f'{year}Q{q}'):3}{flag}",
                      flush=True)
    except QuotaExhausted as e:
        print(f"\nSTOPPED: {e}")
        print("Cached cells are kept -- rerunning resumes from here.")

    df = pd.DataFrame(rows)
    aud = pd.DataFrame(audit)
    if df.empty:
        print("no rows")
        return df

    df = df.drop_duplicates(subset=["doc_id", "ticker"])
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.AV_NEWS, index=False)
    aud.to_parquet(paths.AV_CELL_AUDIT, index=False)
    print(f"\nwrote data/processed/av_news.parquet       ({len(df):,} articles)")
    print(f"wrote data/processed/av_cell_audit.parquet ({len(aud)} cells)")

    short = aud[aud["short"]]
    if len(short):
        print(f"\n{len(short)} cells returned FEWER than {keep} articles -- their salience "
              f"is noisier still. Worst:")
        print(short.nsmallest(6, "n_available")[["ticker","period","n_available"]].to_string(index=False))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="0 requests: plan + cache state")
    ap.add_argument("--sample", action="store_true", help="1 request")
    ap.add_argument("--per-ticker", action="store_true", help="135 requests instead of 15")
    ap.add_argument("--keep", type=int, default=KEEP_PER_CELL, help="articles kept per ticker-quarter")
    ap.add_argument("--sort", default="RELEVANCE", choices=["RELEVANCE", "EARLIEST", "LATEST"])
    ap.add_argument("--force", action="store_true", help="ignore cache")
    a = ap.parse_args()

    cells, tickers = list(quarters()), config.tickers()
    if a.check:
        n = len(cells) * (len(tickers) if a.per_ticker else 1)
        cached = sum(_cache_path("ALL9" if not a.per_ticker else t, y, q, a.sort).exists()
                     for y, q in cells for t in ([None] if not a.per_ticker else tickers))
        print(f"mode      : {'per-ticker' if a.per_ticker else 'batched (all 9 per request)'}")
        print(f"periods   : {cells[0][0]}Q{cells[0][1]} .. {cells[-1][0]}Q{cells[-1][1]}  ({len(cells)} quarters)")
        print(f"requests  : {n}   cached: {cached}   to fetch: {n - cached}")
        print(f"time      : ~{(n - cached) * MIN_INTERVAL / 60:.0f} min at {MIN_INTERVAL:.0f}s spacing")
        print(f"free tier : 25/day -> {'FITS' if n <= 25 else str(-(-n // 25)) + ' days, needs premium'}")
        print(f"keeps     : {a.keep}/ticker/quarter -> {a.keep * len(cells) * len(tickers):,} articles max")
        return 0

    df = harvest(sample=a.sample, per_ticker=a.per_ticker, keep=a.keep,
                 sort=a.sort, force=a.force)
    if df.empty:
        return 1

    print("\nArticles per ticker per quarter:")
    print(df.pivot_table(index="period", columns="ticker", values="doc_id",
                         aggfunc="count", fill_value=0).to_string())
    print("\nDate range:", df["date"].min(), "->", df["date"].max())
    print("\nSample:")
    r = df.iloc[0]
    print(f"  [{r['ticker']} {r['period']}] relevance={r['av_relevance']} src={r['source']}")
    print(f"  {r['title']}")
    print(f"  {r['summary'][:200]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
