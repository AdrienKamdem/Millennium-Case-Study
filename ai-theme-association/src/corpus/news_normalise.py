"""Alpha Vantage NEWS_SENTIMENT dump -> av_news.parquet, with an honest coverage audit.

READ THE AUDIT BEFORE USING THE OUTPUT. The existing dump
(data/NEWS_API/alpha_vantage_news_*.json) was fetched as one call per ticker with
limit=1000 and time_from=20230101. Alpha Vantage answers with the most RECENT 1000,
so eight of nine tickers returned 2026 only:

    NVDA  1000 articles spanning 20 days   (2026-08-25 -> 09-14)
    MU    1000 articles spanning 6 weeks
    WM    1000 articles spanning 6 months
    TEAM   109 articles spanning 2023-2026  (never hit the cap)

That is not a Jan-2023 corpus, and salience-as-a-share across it would compare NVDA
over twenty days against WM over six months. This module computes and prints that
span per ticker so the defect is visible in the run log rather than discovered in
the index. `python -m src.corpus.alphavantage` is the fix: same API, sliced by
quarter, so the window is even.

Emits the schema src/classify/prefilter.py expects, so news and SEC chunks land in
one frame: doc_id, ticker, period, date, title, summary, text, url, domain,
source_category.

    python -m src.corpus.news_normalise
    python -m src.corpus.news_normalise --min-relevance 0.5
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from src.utils import paths

RAW_GLOB = paths.NEWS_RAW_GLOB
OUT = paths.AV_NEWS
AUDIT = paths.AV_NEWS_COVERAGE

# Alpha Vantage tags an article against every ticker it mentions. A NVDA query returns
# pieces whose subject is AMD. Its own relevance_score is the filter; 0.5 keeps ~96-99%
# of rows, so this is a floor against obvious mistags, not a content judgement.
MIN_RELEVANCE = 0.5

TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "guccounter"}


def canonical_url(url: str) -> str:
    """Strip tracking params and the fragment so the same article hashes once."""
    p = urlsplit(url)
    q = "&".join(kv for kv in p.query.split("&")
                 if kv and kv.split("=")[0] not in TRACKING)
    return urlunsplit((p.scheme, p.netloc.lower(), p.path.rstrip("/"), q, ""))


def _doc_id(url: str, ticker: str) -> str:
    """Identity is (article, ticker), not article.

    The same piece can be tagged to several tickers and we keep one row per
    (article, ticker) on purpose -- an article mentioning Micron and Nvidia is real
    coverage of both. Hashing the URL alone therefore emits duplicate ids, which the
    Batch API rejects outright ("custom_id`s must be unique within a batch") and which
    would otherwise make collect()'s merge attach one classification to both tickers.
    """
    return hashlib.sha256(f"{canonical_url(url)}|{ticker}".encode()).hexdigest()[:16]


def _article_id(url: str) -> str:
    """Ticker-independent id, for spotting the same story across companies."""
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:16]


def load(path: str | None = None) -> pd.DataFrame:
    """Flatten the per-ticker feeds into one row per (article, ticker)."""
    path = path or max(glob.glob(RAW_GLOB), default="")
    if not path:
        raise FileNotFoundError(f"no dump matching {RAW_GLOB}")
    print(f"reading {path}")
    payload = json.loads(Path(path).read_text())

    rows = []
    for ticker, block in payload["data"].items():
        for a in block.get("feed", []):
            rel = sent = None
            for t in a.get("ticker_sentiment", []):
                if t.get("ticker") == ticker:
                    rel = float(t.get("relevance_score", 0) or 0)
                    sent = float(t.get("ticker_sentiment_score", 0) or 0)
                    break
            ts = pd.to_datetime(a["time_published"], format="%Y%m%dT%H%M%S", utc=True)
            title, summary = a.get("title") or "", a.get("summary") or ""
            rows.append({
                "doc_id": _doc_id(a["url"], ticker),
                "article_id": _article_id(a["url"]),
                "ticker": ticker,
                "date": ts,
                "period": str(pd.Period(ts, freq="Q")),
                "title": title,
                "summary": summary,
                # Headline plus summary IS the document here -- no body was fetched.
                # The brief treats headlines and metadata at scale as a sufficient corpus.
                "text": f"{title}. {summary}".strip(),
                "url": a["url"],
                "domain": (a.get("source_domain") or "").lower(),
                "source": a.get("source") or "",
                "source_category": "news",
                "av_relevance": rel,
                "av_ticker_sentiment": sent,
                "av_overall_sentiment": float(a.get("overall_sentiment_score") or 0),
            })
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker span. The point of this table is to make the truncation undeniable."""
    g = df.groupby("ticker")["date"]
    cov = pd.DataFrame({"n": g.size(), "earliest": g.min(), "latest": g.max()})
    cov["days_spanned"] = (cov["latest"] - cov["earliest"]).dt.days
    cov["reaches_2023"] = cov["earliest"] < pd.Timestamp("2023-07-01", tz="UTC")
    return cov.sort_values("days_spanned")


def build(min_relevance: float = MIN_RELEVANCE, path: str | None = None) -> pd.DataFrame:
    df = load(path)
    before = len(df)
    df = df[df["av_relevance"].fillna(0) >= min_relevance]
    print(f"{before:,} rows -> {len(df):,} after relevance >= {min_relevance}")

    # Same article, two tickers, is two observations -- both are real coverage of that
    # company. Only exact (doc_id, ticker) repeats are duplicates.
    dedup = df.drop_duplicates(subset=["doc_id", "ticker"])
    print(f"{len(df) - len(dedup):,} exact (article, ticker) duplicates removed")

    cov = coverage(dedup)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(AUDIT)
    print(f"\ncoverage by ticker (wrote {AUDIT}):")
    print(cov.to_string())

    bad = cov[~cov["reaches_2023"]]
    if len(bad):
        print(f"\n  *** {len(bad)} of {len(cov)} tickers do not reach H1 2023. ***")
        print("  This dump cannot support a Jan-2023 time series. Either restrict news to")
        print("  a current-period cross-check, or re-pull with src.corpus.alphavantage,")
        print("  which slices by quarter instead of taking the most recent 1000.")

    dedup.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nwrote {OUT}  ({len(dedup):,} articles)")
    return dedup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-relevance", type=float, default=MIN_RELEVANCE)
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    build(a.min_relevance, a.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
