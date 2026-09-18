"""Split documents into classifier-sized chunks. Offline; no network, no LLM.

Every constant here is set by something measured earlier in this corpus, not by taste.

WHY CHUNK AT ALL. A 10-K runs 402k chars at the median and 643k at the max. One
document was never going to be one classification call. src/classify/llm_classify.py
clips its user block at text[:4000], so feeding it whole filings would have silently
discarded 99% of every 10-K while still reporting a confident answer.

TARGET_CHARS = 3000. Sized to sit under that 4000-char clip with room for the
"Company / Date / Source type" preamble, so nothing is ever truncated in transit. At
~4 chars/token that is ~750 tokens of content per call.

TABLES. 12.3% of 10-K text volume, 26% of cells numeric. Dropping every table loses
narrative, because filers lay prose out in tables -- that is why the extractor keeps
them and records numeric_frac per table. Here we drop only the numeric-dense ones
(>0.40, the same threshold the extractor flags on), so financial statements stop
consuming classifier tokens while a prose table inside Item 1A survives.

SECTIONS. 119 of 134 periodic filings carry Item boundaries; Intel's 15 do not,
because it files by cross-reference. Chunks never straddle an Item where one is
known, so a chunk from Item 1A is labelled as risk-factor language -- the strongest
negative-valence signal in the corpus. Where sections are absent the chunker falls
back to paragraphs and says so in the `section` column rather than guessing.

NOTHING IS DROPPED FOR BEING OFF-THEME. Salience is a share, so the denominator has
to be every chunk. Gating happens later, in prefilter, and is recorded rather than
applied here.

    python -m src.corpus.chunk
    python -m src.corpus.chunk --sample NVDA
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils import paths

from src.corpus.sec_extract import NUMERIC_TABLE_THRESHOLD

log = logging.getLogger(__name__)

IN = paths.SEC_DOCUMENTS
OUT = paths.CHUNKS

TARGET_CHARS = 3_000     # ~750 tokens; sits under llm_classify's 4000-char clip
MAX_CHARS = 3_800        # hard ceiling: never emit something the clip would cut
MIN_CHARS = 400          # below this, merge into the neighbour rather than emit a sliver
OVERLAP_CHARS = 200      # ~one sentence of carry-over so boundaries do not orphan context

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
PARA_SPLIT = re.compile(r"\n{2,}")

CHUNKER_VERSION = "chunk/1.1"

# The same Part/Item label means different things in a 10-K and a 10-Q, which makes the
# raw label useless for analysis: 10-K Part I Item 2 is Properties, 10-Q Part I Item 2 is
# MD&A, and pooling them put 4.8M chars under "I.2". Risk factors live at I.1A in a 10-K
# and II.1A in a 10-Q. section_name resolves both to one canonical name so "give me the
# risk-factor chunks" is a single filter.
SECTION_NAMES = {
    ("10-K", "I", "1"): "business",
    ("10-K", "I", "1A"): "risk_factors",
    ("10-K", "I", "1B"): "unresolved_staff_comments",
    ("10-K", "I", "1C"): "cybersecurity",
    ("10-K", "I", "2"): "properties",
    ("10-K", "I", "3"): "legal_proceedings",
    ("10-K", "I", "4"): "mine_safety",
    ("10-K", "II", "5"): "market_for_equity",
    ("10-K", "II", "6"): "reserved",
    ("10-K", "II", "7"): "mdna",
    ("10-K", "II", "7A"): "market_risk",
    ("10-K", "II", "8"): "financial_statements",
    ("10-K", "II", "9"): "accountant_changes",
    ("10-K", "II", "9A"): "controls",
    ("10-K", "II", "9B"): "other_information",
    ("10-K", "III", "10"): "directors",
    ("10-K", "III", "11"): "executive_compensation",
    ("10-K", "III", "12"): "security_ownership",
    ("10-K", "III", "13"): "related_transactions",
    ("10-K", "III", "14"): "accountant_fees",
    ("10-K", "IV", "15"): "exhibits",
    ("10-K", "IV", "16"): "summary",
    ("10-Q", "I", "1"): "financial_statements",
    ("10-Q", "I", "2"): "mdna",
    ("10-Q", "I", "3"): "market_risk",
    ("10-Q", "I", "4"): "controls",
    ("10-Q", "II", "1"): "legal_proceedings",
    ("10-Q", "II", "1A"): "risk_factors",
    ("10-Q", "II", "2"): "unregistered_sales",
    ("10-Q", "II", "3"): "defaults",
    ("10-Q", "II", "4"): "mine_safety",
    ("10-Q", "II", "5"): "other_information",
    ("10-Q", "II", "6"): "exhibits",
}


def section_name(form: str, label: str) -> str:
    """Canonical name for a Part/Item label, which is meaningless without the form.

    A 10-K with no Part marker detected yields a bare "1A"; default it to Part I, where
    the 10-K risk factors actually live.
    """
    if not label:
        return ""
    part, _, item = label.rpartition(".")
    if not part:
        part = "I"
    return SECTION_NAMES.get((form, part, item.upper()), f"{form}:{part}.{item}")


def _chunk_id(doc_id: str, ix: int) -> str:
    return hashlib.sha256(f"{doc_id}:{ix}".encode()).hexdigest()[:16]


def _drop_numeric_tables(text: str, table_spans) -> tuple[str, int]:
    """Remove numeric-dense tables, keeping prose tables. Returns text and chars dropped.

    Works back-to-front so that earlier offsets stay valid as later spans are cut.
    """
    spans = sorted((s for s in table_spans if s["numeric_frac"] > NUMERIC_TABLE_THRESHOLD),
                   key=lambda s: s["start"], reverse=True)
    dropped = 0
    for s in spans:
        dropped += s["end"] - s["start"]
        text = text[:s["start"]] + "\n\n" + text[s["end"]:]
    return text, dropped


def _units(text: str) -> list[str]:
    """Paragraphs, then sentences for any paragraph too long to place whole."""
    out = []
    for para in PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHARS:
            out.append(para)
            continue
        sentences, buf = SENTENCE_END.split(para), ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= MAX_CHARS:
                buf = f"{buf} {s}".strip()
            else:
                if buf:
                    out.append(buf)
                # A single sentence over the ceiling (usually a run-on table caption)
                # gets hard-split; losing a boundary beats losing the text.
                while len(s) > MAX_CHARS:
                    out.append(s[:MAX_CHARS])
                    s = s[MAX_CHARS:]
                buf = s
        if buf:
            out.append(buf)
    return out


def _tail(text: str, n: int) -> str:
    """Last ~n chars, snapped back to a sentence boundary so overlap reads cleanly."""
    if len(text) <= n:
        return text
    piece = text[-n:]
    m = SENTENCE_END.search(piece)
    return piece[m.end():] if m else piece


def _pack(units: list[str]) -> list[str]:
    """Greedily fill to TARGET_CHARS, carrying OVERLAP_CHARS across boundaries."""
    chunks, buf = [], ""
    for u in units:
        if buf and len(buf) + len(u) + 2 > TARGET_CHARS:
            chunks.append(buf)
            # Overlap is a nicety; the ceiling is not. A full-size unit plus 200 chars
            # of carry-over plus the join comes to 4002, which is past llm_classify's
            # 4000-char clip -- the exact truncation this module exists to avoid. Drop
            # the overlap rather than the text.
            cand = (_tail(buf, OVERLAP_CHARS) + "\n\n" + u).strip() if OVERLAP_CHARS else u
            buf = cand if len(cand) <= MAX_CHARS else u
        else:
            buf = f"{buf}\n\n{u}".strip() if buf else u
    if buf:
        # A short final chunk is folded back rather than emitted as a sliver, unless
        # doing so would breach the ceiling.
        if chunks and len(buf) < MIN_CHARS and len(chunks[-1]) + len(buf) + 2 <= MAX_CHARS:
            chunks[-1] = f"{chunks[-1]}\n\n{buf}"
        else:
            chunks.append(buf)
    return chunks


def chunk_document(row) -> list[dict]:
    """One document -> its chunks, section-aware where sections are trustworthy."""
    text, dropped = _drop_numeric_tables(row["text"], list(row["table_spans"]))
    sections = list(row["section_spans"])

    if sections:
        # Offsets index the ORIGINAL text, so re-derive them against the trimmed copy
        # by slicing the original and dropping tables per section instead.
        pieces = []
        for s in sections:
            seg, seg_drop = _drop_numeric_tables(
                row["text"][s["start"]:s["end"]],
                [{**t, "start": t["start"] - s["start"], "end": t["end"] - s["start"]}
                 for t in row["table_spans"]
                 if t["start"] >= s["start"] and t["end"] <= s["end"]])
            pieces.append((s["item"], seg, seg_drop))
    else:
        pieces = [("", text, dropped)]

    out, ix = [], 0
    for item, seg, seg_drop in pieces:
        for j, body in enumerate(_pack(_units(seg))):
            out.append({
                "chunk_id": _chunk_id(row["doc_id"], ix),
                "doc_id": row["doc_id"],
                "chunk_ix": ix,
                "section": item,
                "section_name": section_name(row["form"], item),
                "text": body,
                "n_chars": len(body),
                "n_words": len(body.split()),
                # Booked once per SECTION, on its first chunk, so summing the column
                # over a document gives the document's true dropped total. Keyed on the
                # per-section index, not the running one.
                "table_chars_dropped": seg_drop if j == 0 else 0,
            })
            ix += 1
    return out


def build(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, d in docs.iterrows():
        for c in chunk_document(d):
            rows.append({
                **c,
                "ticker": d["ticker"],
                "company": d["company"],
                "date": d["date"],
                "period": str(pd.Period(pd.Timestamp(d["date"]), freq="Q")),
                "form": d["form"],
                "doc_type": d["doc_type"],
                "doc_role": d["doc_role"],
                "url": d["url"],
                "source_category": d["source_category"],
                "domain": d["domain"],
                "title": f"{d['ticker']} {d['doc_type']} {d['filing_date']}"
                         + (f" Item {c['section']}" if c["section"] else "")
                         + f" #{c['chunk_ix']}",
                "chunker_version": CHUNKER_VERSION,
            })
    return pd.DataFrame(rows)


def coverage(chunks: pd.DataFrame, docs: pd.DataFrame) -> None:
    print(f"\n{len(chunks):,} chunks from {len(docs):,} documents "
          f"({len(chunks)/len(docs):.1f} per document)")
    print(f"chars: {chunks['n_chars'].sum():,} kept, "
          f"{docs['n_chars'].sum() - chunks['n_chars'].sum():,} dropped "
          f"(numeric tables + overlap accounting)")
    print(f"est. tokens: {chunks['n_chars'].sum() // 4:,}")
    print("\nchunk size:")
    print(chunks["n_chars"].describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_string())
    print(f"  over the {MAX_CHARS}-char ceiling: {int((chunks.n_chars > MAX_CHARS).sum())}")
    print("\nchunks by form:")
    print(chunks.groupby("form").agg(chunks=("chunk_id", "size"),
                                     chars=("n_chars", "sum")).to_string())
    print("\nchunks by ticker:")
    print(chunks["ticker"].value_counts().to_string())
    sectioned = chunks[chunks["section"] != ""]
    print(f"\nsection-labelled chunks: {len(sectioned):,} / {len(chunks):,} "
          f"({100*len(sectioned)/max(len(chunks),1):.0f}%)")
    if len(sectioned):
        print("top sections by volume (canonical, 10-K and 10-Q resolved):")
        print(sectioned.groupby("section_name")
              .agg(chunks=("chunk_id", "size"), chars=("n_chars", "sum"))
              .sort_values("chars", ascending=False).head(10).to_string())
    tiny = chunks[chunks["n_chars"] < MIN_CHARS]
    print(f"\nchunks under MIN_CHARS ({MIN_CHARS}): {len(tiny):,} "
          f"({tiny['n_chars'].sum():,} chars). These are real but trivial sections -- "
          f"'Item 4. Mine Safety Disclosures. Not applicable.' -- kept so the salience "
          f"denominator stays honest; the lexicon gate excludes them from the batch.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="one ticker only")
    args = ap.parse_args()

    if not IN.exists():
        print(f"{IN} not found -- run `make sec-corpus` first")
        return 1

    docs = pd.read_parquet(IN)
    if args.sample:
        docs = docs[docs["ticker"] == args.sample]
        print(f"SAMPLE MODE: {args.sample} only")

    chunks = build(docs)
    if chunks.empty:
        print("NO CHUNKS produced")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(OUT, index=False, compression="zstd")
    print(f"wrote {OUT}  ({len(chunks):,} chunks)")
    coverage(chunks, docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
