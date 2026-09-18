"""Did the extraction actually work? Measured, not asserted.

"As accurate as possible" is only meaningful if it is checkable, so this is a module
rather than a print statement at the end of the harvest. Two layers:

PER-DOCUMENT CHECKS catch the failures that have a signature -- XBRL that survived
the pre-strip, unescaped entities, a document that came back empty, running headers
that were not removed.

A CROSS-EXTRACTOR CHECK catches the failure that does not: cleaning that fails open
and deletes real prose. A character count will not show it, and neither will any
single-extractor statistic, because the output looks perfectly well-formed. So a
deliberately naive second extractor is run over a sample and the two are compared.
The invariant that matters is alpha_chars(primary) >= 0.98 * alpha_chars(baseline):
the structural extractor should only ever ADD newlines and pipes, never remove
letters. Jaccard is the softer signal; this one is the tripwire.

    python -m src.corpus.quality
    python -m src.corpus.quality --dump 5      # write cleaned text out to read
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils import paths
from bs4 import BeautifulSoup

from src.corpus import sec_extract
from src.corpus.sec_documents import OUT, fetch_document

log = logging.getLogger(__name__)

QC_OUT = paths.EXTRACTION_QC
DUMP_DIR = paths.EXTRACTION_SAMPLES

XBRL_MARKERS = ("xbrli:", "us-gaap:", "iso4217:", "utr:", "dei:", "xbrldi:")
ENTITY_RESIDUE = re.compile(r"&[a-zA-Z]{2,8};|&#\d{2,5};")
TOC_RESIDUE = re.compile(r"(?mi)^table\s+of\s+contents\.?$")

# Bounds are deliberately wide: they are outlier detection, not validation. A 10-K
# outside 80k-900k chars is either truncated or full of something it should not be.
CHAR_BOUNDS = {
    "10-K": (80_000, 900_000),
    "10-Q": (30_000, 400_000),
    "8-K": (1_000, 60_000),
    "_exhibit": (500, 200_000),
}
# A primary document under 50 words is broken. An exhibit under 50 words may simply be
# short: PG files a 41-word EX-99.1 every year summarising its D&O insurance limits, and
# INTC a 388-char one. Judge the two by different bars rather than failing the run on a
# document that extracted perfectly.
MIN_WORDS = 50
MIN_WORDS_EXHIBIT = 15
MIN_ALPHA_RATIO = 0.50
WORD_LEN_BOUNDS = (3.5, 7.0)
MAX_FRAGMENT_RATE = 0.05

SAMPLE_PER_FORM = 10
JACCARD_FLOOR = {"10-K": 0.85, "10-Q": 0.85, "8-K": 0.95}
JACCARD_EYEBALL = 0.70
ALPHA_RETENTION_FLOOR = 0.98

# Digits belong INSIDE the token. Without them, "Q2" yields the one-letter word "Q",
# and an Atlassian shareholder letter says "Q2" on every other line -- which is how the
# first full run flagged 65 documents as fragmented when the prose was perfectly clean.
# Same for H100, GB200, FY24. A genuine mid-word space ("t he") still scores a length-1
# token, which is what the check is for.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def _alpha_chars(s: str) -> int:
    return sum(c.isalpha() for c in s)


def _fragment_rate(text: str) -> float:
    """Share of one-letter words. Spaces injected mid-word are the tell."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    singles = sum(1 for w in words if len(w) == 1 and w not in ("a", "A", "I"))
    return singles / len(words)


def check_document(row) -> dict:
    """Every per-document check, as one flat record."""
    text = row["text"] or ""
    words = text.split()
    n_words = len(words)
    alpha = _alpha_chars(text)

    lo, hi = CHAR_BOUNDS.get(row["form"] if row["doc_role"] == "primary" else "_exhibit",
                             CHAR_BOUNDS["_exhibit"])
    mean_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    frag = _fragment_rate(text)
    alpha_ratio = alpha / len(text) if text else 0.0

    return {
        "doc_id": row["doc_id"],
        "ticker": row["ticker"],
        "form": row["form"],
        "doc_type": row["doc_type"],
        "doc_role": row["doc_role"],
        "filing_date": row["filing_date"],
        "n_chars": len(text),
        "n_words": n_words,
        "alpha_ratio": round(alpha_ratio, 3),
        "mean_word_len": round(mean_word_len, 2),
        "fragment_rate": round(frag, 4),
        "n_tables": row["n_tables"],
        "n_sections": len(row["section_spans"]),
        "xbrl_leak": any(m in text for m in XBRL_MARKERS),
        "entity_residue": bool(ENTITY_RESIDUE.search(text)),
        "toc_residue": bool(TOC_RESIDUE.search(text)),
        "near_empty": n_words < (MIN_WORDS_EXHIBIT if row["doc_role"] == "exhibit"
                                 else MIN_WORDS),
        "short_exhibit": row["doc_role"] == "exhibit" and n_words < MIN_WORDS,
        "size_outlier": not (lo <= len(text) <= hi),
        "low_alpha": alpha_ratio < MIN_ALPHA_RATIO,
        "odd_word_len": not (WORD_LEN_BOUNDS[0] <= mean_word_len <= WORD_LEN_BOUNDS[1]),
        "fragmented": frag > MAX_FRAGMENT_RATE,
        "warnings": ";".join(row["warnings"]),
    }


FAIL_FLAGS = ["xbrl_leak", "entity_residue", "toc_residue", "near_empty"]
WARN_FLAGS = ["size_outlier", "low_alpha", "odd_word_len", "fragmented", "short_exhibit"]


def baseline_text(raw: bytes) -> str:
    """Deliberately naive extraction, for comparison only.

    Same pre-strip so the comparison is about the tree walk and nothing else, then
    bs4's get_text(" ") -- the obvious implementation, and the one whose block-gluing
    behaviour the real extractor exists to avoid.
    """
    s = sec_extract._decode(raw, [])
    s = sec_extract._strip_sgml_wrapper(s)
    s = sec_extract.IX_BLOCK.sub(" ", s)
    soup = BeautifulSoup(s, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def cross_check(docs: pd.DataFrame, per_form: int = SAMPLE_PER_FORM,
                seed: int = 42) -> pd.DataFrame:
    """Compare the structural extractor against the naive one on a stratified sample."""
    # Sampling by index rather than groupby.apply: pandas 3 drops the grouping column
    # from the frame handed to apply, which silently loses `form` here.
    picks = []
    for _, g in docs.groupby("form"):
        picks.extend(g.sample(min(per_form, len(g)), random_state=seed).index)
    sample = docs.loc[picks]

    rows = []
    for _, d in sample.iterrows():
        try:
            raw = fetch_document(d["url"])
        except Exception as e:
            log.warning("cross-check fetch failed %s: %s", d["doc_id"], e)
            continue
        base = baseline_text(raw)
        a = set(_WORD.findall(d["text"].lower()))
        b = set(_WORD.findall(base.lower()))
        union = a | b
        alpha_primary, alpha_base = _alpha_chars(d["text"]), _alpha_chars(base)
        retention = alpha_primary / alpha_base if alpha_base else 1.0
        rows.append({
            "doc_id": d["doc_id"], "ticker": d["ticker"], "form": d["form"],
            "doc_role": d["doc_role"],
            "jaccard": round(len(a & b) / len(union), 4) if union else 1.0,
            "alpha_primary": alpha_primary,
            "alpha_baseline": alpha_base,
            "alpha_retention": round(retention, 4),
            "content_loss": retention < ALPHA_RETENTION_FLOOR,
            "jaccard_low": (len(a & b) / len(union) if union else 1.0)
                           < JACCARD_FLOOR.get(d["form"], 0.85),
        })
    return pd.DataFrame(rows)


def report(docs: pd.DataFrame, per_form: int = SAMPLE_PER_FORM) -> pd.DataFrame:
    qc = pd.DataFrame([check_document(r) for _, r in docs.iterrows()])
    QC_OUT.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(QC_OUT, index=False)
    print(f"wrote {QC_OUT}  ({len(qc):,} documents)\n")

    print("hard failures (expected 0 for every row):")
    for f in FAIL_FLAGS:
        n = int(qc[f].sum())
        print(f"  {'FAIL' if n else 'ok  '}  {f:16s} {n:4d}")

    print("\nsoft flags (review, not necessarily wrong):")
    for f in WARN_FLAGS:
        print(f"        {f:16s} {int(qc[f].sum()):4d}")

    print("\nchars by form:")
    print(qc.groupby("form")["n_chars"].describe()[["count", "min", "50%", "max"]].to_string())

    dups = docs[docs.duplicated("content_sha256", keep=False)]
    print(f"\nduplicate documents by content hash: {len(dups):,}")
    if len(dups):
        print(dups.groupby("content_sha256")["title"].apply(list).head(5).to_string())

    print(f"\ncross-extractor check ({per_form} per form):")
    cc = cross_check(docs, per_form=per_form)
    if cc.empty:
        print("  skipped -- no documents fetchable")
        return qc
    print(cc[["ticker", "form", "doc_role", "jaccard", "alpha_retention"]].to_string(index=False))
    loss = cc[cc["content_loss"]]
    print(f"\n  content_loss (alpha_retention < {ALPHA_RETENTION_FLOOR}): "
          f"{'FAIL ' + str(len(loss)) if len(loss) else 'ok 0'}")
    print(f"  jaccard below form floor: {int(cc['jaccard_low'].sum())} "
          f"(eyeball anything under {JACCARD_EYEBALL})")
    return qc


def dump(docs: pd.DataFrame, n: int, seed: int = 42) -> None:
    """Write cleaned text to disk. Automated checks do not catch prose that reads wrong."""
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    sample = docs.sample(min(n, len(docs)), random_state=seed)
    for _, d in sample.iterrows():
        stem = f"{d['ticker']}_{d['filing_date']}_{d['doc_type'].replace(' ', '_')}_{d['doc_id'][:8]}"
        p = DUMP_DIR / f"{stem}.txt"
        p.write_text(d["text"], encoding="utf-8")
        print(f"  {p}  ({d['n_chars']:,} chars)")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0, help="write N cleaned documents to disk")
    ap.add_argument("--per-form", type=int, default=SAMPLE_PER_FORM,
                    help="cross-extractor sample size per form")
    args = ap.parse_args()

    if not OUT.exists():
        print(f"{OUT} not found -- run `make sec-corpus` first")
        return 1

    docs = pd.read_parquet(OUT)
    report(docs, per_form=args.per_form)
    if args.dump:
        print(f"\ndumping {args.dump} documents:")
        dump(docs, args.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
