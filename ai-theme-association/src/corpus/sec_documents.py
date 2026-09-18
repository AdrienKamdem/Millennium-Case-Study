"""Full-text SEC corpus: one row per document, text included.

This is the artifact the rest of Part 1 reads. It replaces the URL-only index with
bodies, and it replaces src/corpus/edgar.py's paragraph-level output with whole
documents plus the offsets needed to chunk them.

Why whole documents rather than AI-keyword paragraphs. src/classify/prefilter.py
opens with "never decides, never drops -- every document stays in the corpus because
salience is a SHARE of total coverage; filtering non-AI documents out here would
silently inflate every score by shrinking the denominator." True, and then the SEC
rows arriving at prefilter had already been filtered to AI paragraphs upstream --
prefilter.py:86 concedes it in a print statement. So for the primary-disclosure half
of the corpus the denominator was hits-over-hits. Full text restores it.

Every 8-K is kept, not just item 2.02. The brief's taxonomy leads with "AI
partnership or adoption announcements whose materiality to the core business is
unclear", and those file under items 1.01, 7.01 and 8.01 -- never 2.02. Filtering on
2.02 discards the first category in the rubric.

Resumable by construction: raw bytes are cached per URL and interim frames are
written per ticker, so a run that dies on company seven keeps companies one to six.

    python -m src.corpus.sec_documents               # all 9, cold run ~20-30 min
    python -m src.corpus.sec_documents --sample      # NVDA only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.corpus import sec_extract, sec_index
from src.utils import config, paths
from src.utils.cache import cached_get
from src.utils import paths

log = logging.getLogger(__name__)

OUT = paths.SEC_DOCUMENTS
INTERIM = paths.INTERIM


def _doc_id(accession: str, seq: int) -> str:
    return hashlib.sha256(f"{accession}:{seq}".encode()).hexdigest()[:16]


def fetch_document(url: str) -> bytes:
    """Raw document bytes, cached on disk. Decoding is sec_extract's job, not requests'."""
    return cached_get("edgar", url, headers=config.sec_headers(), parse="bytes",
                      min_interval=sec_index.MIN_INTERVAL)


def build_company(company: dict, index: pd.DataFrame) -> pd.DataFrame:
    """Every document for one company: fetch, extract, assemble."""
    ticker, cik = company["ticker"], int(company["cik"])
    filings = index[index["ticker"] == ticker]

    rows = []
    for _, f in filings.iterrows():
        docs = sec_index.list_documents(cik, f["accessionNumber"], f["primaryDocument"])
        for d in docs:
            doc_type = d["doc_type"] or f["form"]
            try:
                raw = fetch_document(d["url"])
            except Exception as e:
                log.warning("%s %s %s: %s", ticker, f["accessionNumber"], d["filename"], e)
                continue

            ex = sec_extract.extract(raw, doc_type=doc_type)
            text = ex.text
            rows.append({
                "doc_id": _doc_id(f["accessionNumber"], d["seq"]),
                "accession": f["accessionNumber"],
                "ticker": ticker,
                "company": f["company"],
                "cik": cik,
                "form": f["form"],
                "doc_type": doc_type,
                "doc_role": d["doc_role"],
                "seq": d["seq"],
                "filename": d["filename"],
                "url": d["url"],
                "date": pd.to_datetime(f["filingDate"], utc=True),
                "filing_date": f["filingDate"],
                "report_date": f["reportDate"],
                "items": f.get("items", ""),
                "title": f"{ticker} {f['form']} {f['filingDate']} {doc_type}".strip(),
                "text": text,
                "source_category": "sec_filing",
                "domain": "sec.gov",
                "n_chars": len(text),
                "n_words": len(text.split()),
                "n_tokens_est": len(text) // 4,
                "n_tables": len(ex.tables),
                "n_tables_numeric": ex.n_tables_numeric,
                "table_spans": [asdict(t) for t in ex.tables],
                "section_spans": [asdict(s) for s in ex.sections],
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
                "extractor_version": sec_extract.EXTRACTOR_VERSION,
                "warnings": ex.warnings,
                "fetched_at": datetime.now(timezone.utc),
            })
        log.info("%s %s %s: %d documents", ticker, f["form"], f["filingDate"], len(docs))

    return pd.DataFrame(rows)


def harvest(sample: bool = False, as_of: str = sec_index.AS_OF) -> pd.DataFrame:
    companies = config.universe()
    if sample:
        companies = [c for c in companies if c["ticker"] == "NVDA"]
        print("SAMPLE MODE: NVDA only")

    # Built once: list_filings walks all nine submission files, and calling it per
    # company would re-read the whole index nine times.
    index = sec_index.list_filings(as_of=as_of)

    INTERIM.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in companies:
        if not c.get("cik"):
            print(f"  {c['ticker']:5} SKIPPED -- no cik in universe.yaml")
            continue
        df = build_company(c, index)
        if df.empty:
            print(f"  {c['ticker']:5} NO DOCUMENTS")
            continue
        # Written per company so a mid-harvest failure costs one company, not nine.
        df.to_parquet(INTERIM / f"sec_documents_{c['ticker']}.parquet", index=False)
        frames.append(df)
        print(f"  {c['ticker']:5} {len(df):4,} docs  {df['n_chars'].sum():12,} chars", flush=True)

    if not frames:
        print("NO ROWS -- check SEC_USER_AGENT, then run `make sec-index`.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nwrote {OUT}  ({len(out):,} documents, {out['n_chars'].sum():,} chars)")
    return out




def coverage(docs: pd.DataFrame) -> None:
    print("\ndocuments by ticker and role:")
    print(pd.crosstab(docs["ticker"], docs["doc_role"]).to_string())
    print("\ncharacters by form:")
    print(docs.groupby("form")["n_chars"].agg(["count", "median", "sum"]).to_string())
    print("\ndocuments by year:")
    print(pd.crosstab(pd.to_datetime(docs["filing_date"]).dt.year, docs["ticker"]).to_string())
    real = docs["warnings"].map(lambda ws: [w for w in ws if not w.startswith("info:")])
    print(f"\ndocuments carrying warnings: {int((real.map(len) > 0).sum()):,} / {len(docs):,}")
    counts = pd.Series([w.split(":")[0] for ws in real for w in ws]).value_counts()
    if len(counts):
        print(counts.to_string())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="NVDA only")
    ap.add_argument("--as-of", default=sec_index.AS_OF)
    args = ap.parse_args()

    docs = harvest(sample=args.sample, as_of=args.as_of)
    if docs.empty:
        return 1
    coverage(docs)


    return 0


if __name__ == "__main__":
    sys.exit(main())
