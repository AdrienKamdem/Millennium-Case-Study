"""Tier 0: free lexicon prefilter.

TAGS candidates; never decides, never drops. Every document stays in the corpus
because salience is a SHARE of total coverage -- filtering non-AI documents out
here would silently inflate every score by shrinking the denominator.

Two jobs:
  1. A free baseline to compare the LLM against. If the LLM and the lexicon
     disagree wildly, one of them is broken and you want to know which.
  2. It defines the "buzzword trap" set for validation: documents the lexicon
     flags but the LLM calls `incidental` are exactly the boilerplate cases.

The \\bAI\\b trap: a bare substring match on "AI" hits Air, AIG, Dubai, Shanghai,
Mumbai. Word boundaries are not optional.

    python -m src.classify.prefilter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from src.utils import paths

AI_TERMS = [
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "large language model", "foundation model", "generative ai", "gen ai",
    "transformer model", "copilot", "chatbot", "agentic", "inference",
    "accelerator", "datacenter gpu", "data center gpu", "training cluster",
    "llm", "gpu", "npu", "tpu", "hbm",
]
AI_PATTERN = re.compile(
    "|".join(rf"\b{re.escape(t)}\b" for t in AI_TERMS) + r"|\bAI\b|\bA\.I\.",
    re.IGNORECASE,
)
# Case-SENSITIVE for the bare acronym: "AI" is a signal, "ai" inside a word is not.
BARE_AI = re.compile(r"\bAI\b|\bA\.I\.")


def flag(docs: pd.DataFrame, text_cols: tuple[str, ...] = ("title", "summary", "text")) -> pd.DataFrame:
    """Add `lexicon_hit` and `lexicon_terms`. Does not filter."""
    present = [c for c in text_cols if c in docs.columns]
    blob = docs[present].fillna("").agg(" ".join, axis=1)
    docs = docs.copy()
    docs["lexicon_terms"] = blob.map(lambda s: sorted({m.group(0).lower()
                                                       for m in AI_PATTERN.finditer(s)}))
    docs["lexicon_hit"] = docs["lexicon_terms"].map(bool)
    return docs


# section_name, not section: a raw Part/Item label means different things in a 10-K and
# a 10-Q (Part I Item 2 is Properties in one, MD&A in the other). Carrying the canonical
# name through to classified.parquet is what lets valence be conditioned on risk_factors
# vs mdna without joining back to chunks.parquet.
COLUMNS = ["doc_id", "parent_doc_id", "ticker", "period", "date", "title", "text",
           "source_category", "section", "section_name", "url"]


def load_corpus() -> pd.DataFrame:
    """Combine news and SEC chunks into one frame with a common schema.

    `doc_id` is the unit of classification, so for SEC it is the CHUNK id; the filing
    it came from is kept as parent_doc_id for aggregating back up. llm_classify keys
    its batch on doc_id, so this is what custom_id ends up being.

    Reads chunks.parquet: whole filings, so salience has a real denominator. An earlier
    version read AI-keyword paragraphs only, which made salience a count of hits over a
    count of hits; that path is gone.
    """
    frames = []

    p = paths.AV_NEWS
    if p.exists():
        n = pd.read_parquet(p)
        if "text" not in n:
            n["text"] = (n["title"].fillna("") + ". " + n["summary"].fillna("")).str.strip()
        n["parent_doc_id"] = n["doc_id"]
        n["section"] = ""
        n["section_name"] = ""
        frames.append(n[COLUMNS])
        print(f"  news   : {len(n):,} articles")

    chunks = paths.CHUNKS
    if chunks.exists():
        s = pd.read_parquet(chunks)
        s = s.rename(columns={"chunk_id": "doc_id", "doc_id": "parent_doc_id"})
        frames.append(s[COLUMNS])
        print(f"  sec    : {len(s):,} chunks from "
              f"{s['parent_doc_id'].nunique():,} filings")

    if not frames:
        raise FileNotFoundError("no corpus found -- run the harvesters first")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    docs = flag(load_corpus())
    out = paths.CORPUS
    docs.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(docs):,} documents)")

    print("\nlexicon hit rate by source:")
    print(docs.groupby("source_category")["lexicon_hit"].agg(["sum", "count", "mean"]).round(3).to_string())
    print("\nlexicon hit rate by ticker -- this IS raw salience, before any LLM:")
    print(docs.groupby(["source_category", "ticker"])["lexicon_hit"].mean()
              .round(3).unstack(0).to_string())

    # What the gate will cost. llm_classify --gate lexicon submits only these.
    hits = int(docs["lexicon_hit"].sum())
    tok = int(docs.loc[docs["lexicon_hit"], "text"].str.len().sum() // 4)
    print(f"\ngate: {hits:,} of {len(docs):,} units carry an AI term "
          f"({hits/max(len(docs),1):.0%}), ~{tok:,} input tokens to classify.")
    print("The other rows stay in corpus.parquet: salience is a share, and dropping "
          "them here would shrink the denominator.")

    sec = docs[docs["source_category"] == "sec_filing"]
    if len(sec) and "section_name" in sec and (sec["section_name"] != "").any():
        print("\nlexicon hit rate by section -- where the AI language actually lives:")
        top = sec[sec["section_name"] != ""].groupby("section_name")["lexicon_hit"].agg(["mean", "count"])
        print(top[top["count"] >= 20].sort_values("mean", ascending=False).round(3).head(10).to_string())
    print("\nmost common matched terms:")
    print(pd.Series([t for ts in docs["lexicon_terms"] for t in ts]).value_counts().head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
