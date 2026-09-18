"""LLM classification + aspect sentiment, in ONE structured call per document.

COST DECISIONS, all three worth defending in the write-up:

  One call, not two. Classification and aspect sentiment come back together,
  roughly halving token spend versus separate passes.

  Batch API: 50% off. Nothing here is latency-sensitive.

  Prompt caching on the rubric. The taxonomy is a fixed ~1.5k-token prefix on
  every request, so it is marked cache_control and billed at ~10% after the first
  write. Verify it is working via cache_read_input_tokens in docs/token_log.csv --
  if that column is zero across thousands of calls, something is invalidating the
  prefix and you are paying full price.

  NO EMBEDDINGS TIER. The original design had lexicon -> embeddings -> LLM to
  route spend. At ~4k documents a full Haiku pass costs ~£2 and the embedding tier
  would save ~£1 for an hour of code. Cut deliberately; it becomes essential at
  600k docs/month (see the scaling memo). Sizing the problem before engineering
  for it is the point.

THE ASPECT-SENTIMENT INSTRUCTION IS THE CRUX. Sentiment is directed at AI's impact
ON THIS COMPANY, not at the document's mood and not at whether AI sounds exciting.
Coverage can be enthusiastic about generative AI while being bearish on Adobe,
because the same technology threatens its moat. Document-level sentiment scores
that exactly backwards -- which is the whole reason the brief asks for aspect-based.

"Judge only from the text provided" is load-bearing: without it the model answers
from its own knowledge of who won, which is a look-ahead leak dressed as sentiment.
validate.py's name-masking test checks whether it held.

    python -m src.classify.llm_classify --sample     # 5 docs, live, ~10s
    python -m src.classify.llm_classify --submit     # submit the batch
    python -m src.classify.llm_classify --collect    # poll + write results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

from src.utils import config, paths
from src.utils.tokenlog import record
from src.utils import paths

MODEL = "claude-haiku-4-5"
BATCH_ID_FILE = paths.BATCH_ID
OUT = paths.CLASSIFIED

SCHEMA = {
    "type": "object",
    "properties": {
        "ai_relevance": {"type": "string", "enum": ["none", "incidental", "substantive"]},
        "ai_role": {"type": "string", "enum": ["A_partnership", "B_costcut",
                                               "C_positive_exposure", "D_negative_exposure", "E_none"]},
        "materiality": {"type": "string", "enum": ["low", "medium", "high"]},
        # strict:true accepts enum/const/anyOf but NOT minimum/maximum/multipleOf --
        # the API rejects them with "for integer type, properties maximum, minimum are
        # not supported". The SDK strips unsupported constraints only when the schema
        # comes from a Pydantic model or @beta_tool; a raw dict like this one is sent
        # verbatim, so the subset has to be respected here.
        #
        # An enum is the better shape regardless: valence is a 5-point scale defined by
        # config/taxonomy.yaml, and this pins the output to exactly those values instead
        # of any integer that happens to land in range.
        "valence_ai": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
        # Continuous, so no enum fits. Range moves into the description and is clamped
        # client-side in _parse.
        "confidence": {"type": "number",
                       "description": "Confidence in this classification, 0.0 to 1.0."},
        "evidence_span": {"type": "string"},
    },
    "required": ["ai_relevance", "ai_role", "materiality", "valence_ai",
                 "confidence", "evidence_span"],
    "additionalProperties": False,
}

TOOL = {
    "name": "classify",
    "description": "Record the classification and AI-aspect sentiment for this document.",
    "input_schema": SCHEMA,
    "strict": True,
}


def rubric() -> str:
    """Built from config/taxonomy.yaml so the rubric and the code cannot drift apart.

    Must be BYTE-STABLE across requests or the prompt cache never hits.
    """
    t = config.load("taxonomy")
    lines = [
        "You classify a single document for its relevance to the AI theme in relation to",
        "one specific company, and score the direction of AI's impact on that company.",
        "",
        "Judge ONLY from the text provided. Do not use anything you know about the company",
        "from outside this document, and do not use hindsight about how events turned out.",
        "",
        "ai_relevance:",
    ]
    for k, v in t["ai_relevance"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("ai_role:")
    for k, v in t["ai_role"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("materiality:")
    for k, v in t["materiality"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("valence_ai -- the crux. Score AI's impact ON THIS COMPANY, not the")
    lines.append("document's overall tone and not whether AI in general sounds promising.")
    lines.append("A document can be enthusiastic about AI while being bearish on this company,")
    lines.append("because the same technology threatens its business. Those are opposite signs.")
    for k in ("-2", "-1", "0", "+1", "+2"):
        lines.append(f"  {k}: {t['valence_ai'][k]}")
    lines.append("")
    lines.append("evidence_span must be an exact quote from the document. If you cannot quote")
    lines.append("something that justifies the call, the answer is ai_relevance=none.")
    return "\n".join(lines)


def user_block(row: pd.Series, mask_name: bool = False) -> str:
    text = str(row.get("text") or row.get("title") or "")
    name = "[COMPANY]" if mask_name else row["ticker"]
    if mask_name:
        for alias in _aliases(row["ticker"]):
            text = text.replace(alias, "[COMPANY]")
    return (f"Company: {name}\nDate: {row['date']}\n"
            f"Source type: {row['source_category']}\n\n---\n{text[:4000]}\n---")


def _aliases(ticker: str) -> list[str]:
    for c in config.universe():
        if c["ticker"] == ticker:
            return c["aliases"] + [ticker]
    return [ticker]


def _params(row: pd.Series, model: str, mask_name: bool = False) -> dict:
    return {
        "model": model,
        "max_tokens": 512,
        # cache_control on the rubric: fixed prefix, billed at ~10% after the write.
        "system": [{"type": "text", "text": rubric(),
                    "cache_control": {"type": "ephemeral"}}],
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "classify"},
        "messages": [{"role": "user", "content": user_block(row, mask_name)}],
    }


def _parse(message) -> dict | None:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use":
            out = dict(block.input)
            # confidence lost its schema bounds (strict:true rejects minimum/maximum),
            # so enforce the range here rather than trusting the model to stay in it.
            try:
                out["confidence"] = min(1.0, max(0.0, float(out.get("confidence", 0.0))))
            except (TypeError, ValueError):
                out["confidence"] = 0.0
            return out
    return None


# --------------------------------------------------------------------------- #

def sample(n: int = 5, model: str = MODEL, gate: str = "lexicon") -> pd.DataFrame:
    """Live, non-batched, a handful of documents. Verify before spending.

    Samples the GATED rows by default -- the ones the batch will actually send. Drawing
    from the whole corpus mostly returns non-AI text, which only exercises the easy
    path: the classifier saying "none" to a dividend announcement proves little. The
    call worth checking before spending is whether valence tracks AI's impact ON THE
    COMPANY on text that genuinely discusses AI. Pass --gate none to sample everything.
    """
    docs = _gated(pd.read_parquet(paths.CORPUS), gate)
    docs = docs.sample(min(n, len(docs)), random_state=0)
    client = Anthropic()
    rows = []
    for _, r in docs.iterrows():
        msg = client.messages.create(**_params(r, model))
        record("classify:sample", model, msg.usage)
        out = _parse(msg) or {}
        rows.append({**{"doc_id": r["doc_id"], "ticker": r["ticker"]}, **out})
        print(f"\n[{r['ticker']}] {str(r['title'])[:80]}")
        print(f"  terms: {r.get('lexicon_terms', '')}")
        print(f"  -> {out.get('ai_relevance')} / {out.get('ai_role')} / "
              f"mat={out.get('materiality')} / valence={out.get('valence_ai')} "
              f"/ conf={out.get('confidence')}")
        print(f"  evidence: {str(out.get('evidence_span'))[:120]}")
    return pd.DataFrame(rows)


def _gated(docs: pd.DataFrame, gate: str) -> pd.DataFrame:
    """Which rows actually go to the model.

    Chunking took the corpus from ~3k paragraphs to ~15k chunks, because a 10-K is now
    represented whole rather than by its AI-keyword paragraphs. Classifying all of it
    would spend most of the budget on segment tables and accounting policy notes that
    no taxonomy class applies to.

    The lexicon gate sends only chunks containing an AI term. Everything else is scored
    E_none / ai_relevance=none by construction, which is what the classifier would have
    said anyway -- the lexicon is high-recall on this taxonomy, since "substantive AI
    coverage" that never uses an AI word is close to a null set. Crucially the skipped
    rows stay in corpus.parquet, so salience keeps its true denominator.

    --gate none classifies everything, for the cost of doing so. Worth running once on
    a sample to measure the gate's false-negative rate, which is a validate.py job.
    """
    if gate == "none":
        return docs
    if "lexicon_hit" not in docs.columns:
        raise SystemExit("corpus.parquet has no lexicon_hit -- run prefilter first")
    keep = docs[docs["lexicon_hit"]]
    print(f"gate={gate}: {len(keep):,} of {len(docs):,} units "
          f"({len(keep)/max(len(docs),1):.0%}); the rest default to ai_relevance=none")
    return keep


def submit(model: str = MODEL, mask_name: bool = False, limit: int | None = None,
           gate: str = "lexicon") -> str:
    docs = _gated(pd.read_parquet(paths.CORPUS), gate)
    if limit:
        docs = docs.head(limit)
    est = int(docs["text"].str.len().sum() // 4)
    print(f"~{est:,} input tokens before the cached rubric; "
          f"Haiku batch input runs about ${est/1e6*0.5:.2f}")
    client = Anthropic()
    requests = [{"custom_id": r["doc_id"], "params": _params(r, model, mask_name)}
                for _, r in docs.iterrows()]
    print(f"submitting {len(requests):,} requests to the Batch API ({model})")
    batch = client.messages.batches.create(requests=requests)
    BATCH_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_ID_FILE.write_text(batch.id)
    print(f"batch id: {batch.id}  (saved to {BATCH_ID_FILE})")
    print("Batches usually finish well inside an hour. Collect with --collect.")
    return batch.id


def collect(batch_id: str | None = None, model: str = MODEL, wait: bool = True) -> pd.DataFrame:
    batch_id = batch_id or BATCH_ID_FILE.read_text().strip()
    client = Anthropic()

    while True:
        b = client.messages.batches.retrieve(batch_id)
        print(f"  status={b.processing_status}  counts={b.request_counts}", flush=True)
        if b.processing_status == "ended" or not wait:
            break
        time.sleep(30)
    if b.processing_status != "ended":
        return pd.DataFrame()

    rows, errors = [], 0
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            errors += 1
            continue
        msg = res.result.message
        record("classify:bulk", model, msg.usage, batch=True)
        parsed = _parse(msg)
        if parsed is None:
            errors += 1
            continue
        # Results arrive in ANY order -- key by custom_id, never by position.
        rows.append({"doc_id": res.custom_id, **parsed})

    out = pd.DataFrame(rows)
    docs = pd.read_parquet(paths.CORPUS)
    merged = docs.merge(out, on="doc_id", how="left")

    # Gated-out rows were never sent. They are not missing data -- they are the
    # denominator, and leaving them NaN would let a later dropna() silently inflate
    # every salience score. Fill them with the answer the gate already implies.
    if "lexicon_hit" in merged.columns:
        skipped = merged["ai_relevance"].isna() & ~merged["lexicon_hit"]
        merged.loc[skipped, ["ai_relevance", "ai_role", "materiality"]] = \
            ["none", "E_none", "low"]
        merged.loc[skipped, ["valence_ai", "confidence"]] = [0, 1.0]
        merged.loc[skipped, "evidence_span"] = ""
        merged.loc[skipped, "gated_out"] = True
        merged["gated_out"] = merged.get("gated_out", False).fillna(False)
        print(f"{int(skipped.sum()):,} gated-out units defaulted to ai_relevance=none")

    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out):,} classified, {errors} failed, "
          f"{merged['ai_relevance'].isna().sum()} unmatched)")

    print("\nai_relevance distribution:")
    print(merged["ai_relevance"].value_counts(dropna=False).to_string())
    print("\nsubstantive share by ticker (this is raw salience, pre-shrinkage):")
    m = merged.dropna(subset=["ai_relevance"])
    print(m.assign(sub=m["ai_relevance"].eq("substantive"))
           .groupby(["source_category", "ticker"])["sub"].mean().round(3).unstack(0).to_string())
    print("\nmean valence among substantive docs (expect ADBE/TEAM negative):")
    print(m[m["ai_relevance"].eq("substantive")].groupby("ticker")["valence_ai"]
          .agg(["mean", "count"]).round(2).to_string())
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, nargs="?", const=5, help="live run on N docs")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mask-name", action="store_true", help="for the look-ahead leak test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gate", choices=["lexicon", "none"], default="lexicon",
                    help="lexicon (default): only units containing an AI term. "
                         "none: classify everything, and pay for it.")
    a = ap.parse_args()

    if a.sample:
        sample(a.sample, a.model, a.gate)
    elif a.submit:
        submit(a.model, a.mask_name, a.limit, a.gate)
    elif a.collect:
        collect(model=a.model)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
