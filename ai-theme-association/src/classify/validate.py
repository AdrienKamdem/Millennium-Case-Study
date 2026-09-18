"""Classifier diagnostics. Task 2 asks for quality on a reviewed sample, and for
the documents the classifier gets WRONG -- shown, not summarised.

Everything here runs offline against classified.parquet. No API spend, no waiting on a
batch, which is why it is worth doing even on the last day.

WHAT THIS CAN AND CANNOT TELL YOU. The buzzword rate and the contradiction checks are
measurements. The spot-review agreement rate is a judgement you supply by reading forty
documents -- the script only stratifies the sample and gives you a column to fill in.
Neither is accuracy against ground truth, because no ground truth exists here; they
bound it from different sides. Say that in the write-up rather than presenting the
agreement rate as precision.

    python -m src.classify.validate
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import paths

IN = paths.CLASSIFIED
OUTDIR = paths.TABLES

BUZZWORD_CLASSES = ("none", "incidental")


def _sent(c: pd.DataFrame) -> pd.DataFrame:
    """Rows the model actually saw. Gated-out rows were labelled by rule, not judged."""
    gated = c["gated_out"].fillna(False) if "gated_out" in c else pd.Series(False, index=c.index)
    return c[~gated & c["ai_relevance"].notna()]


# --------------------------------------------------------------------------- #

def spot_review_sample(classified: pd.DataFrame, n: int = 40,
                       seed: int = 1) -> pd.DataFrame:
    """Stratified by predicted class. Write to outputs/ as a CSV you fill in by hand.

    Stratified rather than random because a random 40 from this corpus is ~35 `none`
    rows, which tells you nothing about the classes that matter. Equal-ish weight per
    class means the sample is informative about `substantive` precision, at the cost of
    not being a population estimate -- so report it as "agreement on a stratified
    sample", never as overall accuracy.
    """
    s = _sent(classified)
    per = max(1, n // s["ai_relevance"].nunique())
    picks = [g.sample(min(per, len(g)), random_state=seed)
             for _, g in s.groupby("ai_relevance")]
    out = pd.concat(picks).sample(frac=1, random_state=seed)

    cols = [c for c in ("doc_id", "ticker", "source_category", "section_name",
                        "ai_relevance", "ai_role", "materiality", "valence_ai",
                        "confidence", "evidence_span", "text") if c in out.columns]
    out = out[cols].copy()
    # classified.parquet ships without the text column (see README: the labels are the
    # result, the bodies are bulk). Fall back to the sample, which keeps it.
    if "text" in out.columns:
        out["text"] = out["text"].str.slice(0, 700)
    # The column you fill in. y / n / ? -- and the reason when it is n.
    out["agree"] = ""
    out["note"] = ""
    return out


def buzzword_trap(classified: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Documents containing an AI term that the model declined to call substantive.

    This IS the buzzword population. The lexicon is deliberately high-recall -- every one
    of these rows contains "AI", "machine learning", "GPU" or similar -- so the model
    demoting them is exactly the separation Task 2 asks to be demonstrated. The clearest
    case in this corpus is safe-harbour boilerplate: "our business, strategy, artificial
    intelligence ('AI') and innovation momentum" is an AI mention and carries no AI
    information.
    """
    s = _sent(classified)
    flagged = s[s["lexicon_hit"]] if "lexicon_hit" in s else s
    trap = flagged[flagged["ai_relevance"].isin(BUZZWORD_CLASSES)]
    stats = {
        "lexicon_flagged_and_sent": len(flagged),
        "demoted_to_none_or_incidental": len(trap),
        "separation_rate": round(len(trap) / len(flagged), 3) if len(flagged) else 0.0,
        "kept_substantive": int((flagged["ai_relevance"] == "substantive").sum()),
    }
    cols = [c for c in ("ticker", "source_category", "section_name", "ai_relevance",
                        "lexicon_terms", "evidence_span") if c in trap.columns]
    return stats, trap[cols]


def representative_by_class(classified: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Highest-confidence example(s) per (ai_relevance, ai_role). Task 2, literally."""
    s = _sent(classified)
    cols = [c for c in ("ticker", "section_name", "materiality", "valence_ai",
                        "confidence", "evidence_span") if c in s.columns]
    return (s.sort_values("confidence", ascending=False)
             .groupby(["ai_relevance", "ai_role"], group_keys=True)
             .head(k)
             .set_index(["ai_relevance", "ai_role"])[cols]
             .sort_index())


def likely_errors(classified: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """The documents the classifier probably got wrong -- found, not hoped for.

    Three of these are self-contradictions: the model returned a combination its own
    rubric forbids, so no human judgement is needed to call them errors. The fourth is
    a disagreement with the lexicon, which is not automatically a mistake -- the model
    finding AI content where no keyword appears may be a genuine catch -- but it is
    always worth reading.
    """
    s = _sent(classified).copy()
    s["why"] = ""

    def mark(mask, label):
        s.loc[mask & (s["why"] == ""), "why"] = label

    mark((s["ai_role"] == "D_negative_exposure") & (s["valence_ai"] > 0),
         "negative role, positive valence")
    mark((s["ai_relevance"] == "none")
         & ((s["ai_role"] != "E_none") | (s["valence_ai"] != 0)),
         "relevance none but role/valence set")
    mark((s["ai_relevance"] == "substantive") & (s["ai_role"] == "E_none"),
         "substantive but no role")
    if "lexicon_hit" in s:
        mark((s["ai_relevance"] == "substantive") & (~s["lexicon_hit"]),
             "substantive with no AI term in the text")

    flagged = s[s["why"] != ""]
    # Top up with the least-confident substantive calls: not contradictions, but the
    # rows most worth a human look.
    low = (s[(s["why"] == "") & (s["ai_relevance"] == "substantive")]
           .nsmallest(max(0, k - len(flagged)), "confidence")
           .assign(why="lowest-confidence substantive"))

    cols = [c for c in ("why", "ticker", "source_category", "section_name",
                        "ai_relevance", "ai_role", "valence_ai", "confidence",
                        "evidence_span") if c in s.columns]
    # Every contradiction is returned. k caps only the low-confidence top-up, because
    # truncating the error list to a round number would hide errors from the artifact
    # whose whole purpose is showing them.
    return pd.concat([flagged, low])[cols]


def cross_model_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Cohen's kappa per field between two classified frames, joined on doc_id.

    No API call -- this compares results you already have. Produce `b` by re-running
    llm_classify with --model claude-sonnet-5 over a sample, then pass both frames.
    Kappa measures CONSISTENCY between two models, not correctness; a high value means
    the task is well specified, not that the answers are right.
    """
    from sklearn.metrics import cohen_kappa_score

    j = a.merge(b, on="doc_id", suffixes=("_a", "_b"))
    out = {"n": len(j)}
    for f in ("ai_relevance", "ai_role", "materiality"):
        ca, cb = f"{f}_a", f"{f}_b"
        if ca in j and cb in j:
            m = j[[ca, cb]].dropna()
            out[f] = round(cohen_kappa_score(m[ca], m[cb]), 3) if len(m) else None
    if "valence_ai_a" in j:
        m = j[["valence_ai_a", "valence_ai_b"]].dropna()
        out["valence_corr"] = round(m.corr().iloc[0, 1], 3) if len(m) > 2 else None
        out["valence_exact_match"] = round((m.iloc[:, 0] == m.iloc[:, 1]).mean(), 3) if len(m) else None
    return out


def name_masking_test(docs: pd.DataFrame, n: int = 100) -> dict:
    """Not run. Requires live API calls, which were out of scope on submission day.

    The mechanism exists: `llm_classify --mask-name` replaces every company alias with
    [COMPANY] before sending. Re-run a sample with it and compare valence against the
    unmasked run; material movement means the model is scoring its own priors about who
    won rather than the text in front of it, which is a look-ahead leak dressed as
    sentiment. Descoped deliberately, recorded here rather than omitted.
    """
    raise NotImplementedError(
        "name-masking needs a live run: python -m src.classify.llm_classify "
        "--sample 100 --mask-name, then compare valence against classified.parquet")


# --------------------------------------------------------------------------- #

def main() -> int:
    if not IN.exists():
        print(f"{IN} not found -- run `make collect` first")
        return 1
    c = pd.read_parquet(IN)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = _sent(c)

    print("=" * 72)
    print("WHAT WAS JUDGED")
    print("=" * 72)
    gated = len(c) - len(s) - int(c["ai_relevance"].isna().sum())
    print(f"  corpus           {len(c):,}")
    print(f"  judged by model  {len(s):,}")
    print(f"  labelled by gate {gated:,}  (no AI term -> none by construction)")
    print(f"  failed           {int(c['ai_relevance'].isna().sum()):,}")
    print("\n" + s["ai_relevance"].value_counts().to_string())

    print("\n" + "=" * 72)
    print("BUZZWORD SEPARATION  (Task 2: genuine AI coverage vs buzzword mentions)")
    print("=" * 72)
    stats, trap = buzzword_trap(c)
    for k, v in stats.items():
        print(f"  {k:32s} {v}")
    print(f"\n  {stats['separation_rate']:.0%} of documents containing an AI term were "
          f"demoted below 'substantive'.")
    print("  The lexicon is high-recall by design, so this set is the buzzword population.")
    trap.to_csv(OUTDIR / "buzzword_trap.csv", index=False)
    print(f"\n  wrote {OUTDIR / 'buzzword_trap.csv'}  ({len(trap):,} rows)")
    print("\n  examples:")
    for _, r in trap.head(4).iterrows():
        print(f"    [{r.get('ticker','')} {r.get('ai_relevance','')}] "
              f"{str(r.get('evidence_span',''))[:120]}")

    print("\n" + "=" * 72)
    print("REPRESENTATIVE DOCUMENTS PER CLASS")
    print("=" * 72)
    rep = representative_by_class(c, k=1)
    for (rel, role), r in rep.iterrows():
        print(f"\n  {rel} / {role}  (conf {r.get('confidence', 0):.2f}, "
              f"valence {r.get('valence_ai', 0):+.0f})")
        print(f"    {str(r.get('evidence_span',''))[:150]}")
    representative_by_class(c, k=3).to_csv(OUTDIR / "representative_by_class.csv")

    print("\n" + "=" * 72)
    print("LIKELY ERRORS  (Task 2: 'including the documents your classifier gets wrong')")
    print("=" * 72)
    err = likely_errors(c)
    if err.empty:
        print("  none flagged -- check the rules are not too narrow before believing this")
    else:
        print(err["why"].value_counts().to_string())
        err.to_csv(OUTDIR / "likely_errors.csv", index=False)
        print(f"\n  wrote {OUTDIR / 'likely_errors.csv'}")
        for _, r in err.head(5).iterrows():
            print(f"\n    [{r['why']}] {r.get('ticker','')} "
                  f"{r.get('ai_relevance','')}/{r.get('ai_role','')} "
                  f"v={r.get('valence_ai',0):+.0f}")
            print(f"      {str(r.get('evidence_span',''))[:130]}")

    print("\n" + "=" * 72)
    print("SPOT REVIEW  (the part only a human can do)")
    print("=" * 72)
    sample = spot_review_sample(c)
    p = OUTDIR / "validation_sample.csv"
    sample.to_csv(p, index=False)
    print(f"  wrote {p}  ({len(sample)} rows, stratified by predicted class)")
    print("  Open it, fill the `agree` column with y/n/?, and note a reason on every n.")
    print("  The resulting rate is the headline number for Task 2 -- report it as")
    print("  agreement on a stratified sample, not as overall accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
