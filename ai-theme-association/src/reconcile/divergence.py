"""Where the two methods disagree, and why. Task 06's core judgment test.

The brief: "identify the 2-3 largest disagreements and give each a specific, testable
hypothesis. This is the core judgment test." It also says the disagreement is the
interesting result -- signal, not a bug to smooth away.

THREE-WAY TRIAGE. Every gap is one of:

  MEASUREMENT   one side is mis-measured. Thin coverage, a section the extractor could
                not isolate, a beta with a t-stat near zero or a VIF that makes it
                uninterpretable. Check this FIRST -- it is the boring explanation and
                it is often right.
  CHANNEL       both measurements are fine and they are measuring different things.
                The text says a company is engaged with AI, but its earnings exposure
                runs through a channel the AI factor does not span (power, cooling,
                logistics, construction).
  MISPRICING    both measurements are fine, the channel is spanned, and the market
                simply has not repriced the disclosure yet. The tradeable case, and
                the one to claim least often.

This module computes the evidence for each; the call is yours to write.

    python -m src.reconcile.divergence
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import paths

RECON = paths.RECONCILIATION
INDEX = paths.AIA_INDEX
CLASSIFIED = paths.CLASSIFIED

# Below this the beta is not distinguishable from zero, so a "disagreement" with it is
# a disagreement with noise.
T_WEAK = 1.5
# EB shrinkage leaves a cell mostly prior below roughly this many substantive units.
THIN_SUBSTANTIVE = 20


def largest_gaps(df: pd.DataFrame | None = None, n: int = 3) -> pd.DataFrame:
    df = pd.read_parquet(RECON) if df is None else df
    return df.reindex(df["gap"].abs().sort_values(ascending=False).index).head(n)


def triage(df: pd.DataFrame | None = None, n: int = 3) -> pd.DataFrame:
    """Attach the measurement-quality evidence to each disagreement.

    Does not decide -- flags what would have to be true for MEASUREMENT to be the
    explanation, so you can rule it out before reaching for MISPRICING.
    """
    df = pd.read_parquet(RECON) if df is None else df
    g = largest_gaps(df, n).copy()

    g["direction"] = g["gap"].apply(lambda x: "text > market" if x > 0 else "market > text")
    if "ai_t" in g:
        g["beta_weak"] = g["ai_t"].abs() < T_WEAK
    if "max_vif" in g:
        g["beta_collinear"] = g["max_vif"] > 10
    g["semantic_thin"] = g["n_substantive"] < THIN_SUBSTANTIVE
    g["likely_measurement"] = (g.get("beta_weak", False)
                               | g.get("beta_collinear", False)
                               | g["semantic_thin"])
    return g


def evidence(ticker: str, k: int = 4) -> pd.DataFrame:
    """The actual sentences behind a company's score. Put these on the slide.

    A divergence claim is only as good as the text under it, and a reviewer will ask.
    """
    m = pd.read_parquet(CLASSIFIED)
    s = m[(m["ticker"] == ticker) & (m["ai_relevance"] == "substantive")
          & (m["source_category"] == "sec_filing")]
    if s.empty:
        return s
    cols = [c for c in ("period", "section_name", "ai_role", "materiality",
                        "valence_ai", "evidence_span") if c in s.columns]
    top = pd.concat([s.nlargest(k, "valence_ai"), s.nsmallest(k, "valence_ai")])
    return top[cols].drop_duplicates()


def trajectory(ticker: str) -> pd.DataFrame:
    """Has the semantic score MOVED? A rising score against a static full-sample beta
    is the shape that motivates the Task 07 lead-lag design."""
    idx = pd.read_parquet(INDEX)
    return (idx[idx["ticker"] == ticker]
            [["period", "salience_shrunk", "valence_shrunk", "aia", "n_substantive"]]
            .set_index("period").round(3))


def main() -> int:
    df = pd.read_parquet(RECON)
    g = triage(df)

    print("=== the largest disagreements ===")
    cols = [c for c in ("semantic_z", "beta_z", "gap", "direction", "ai_t", "max_vif",
                        "n_substantive", "beta_weak", "beta_collinear", "semantic_thin",
                        "likely_measurement") if c in g.columns]
    print(g[cols].round(3).to_string())

    for t in g.index:
        r = g.loc[t]
        print("\n" + "=" * 72)
        print(f"{t}   gap {r['gap']:+.2f} sd   ({r['direction']})")
        print("=" * 72)
        if r.get("likely_measurement", False):
            why = []
            if r.get("beta_weak", False):
                why.append(f"beta t={r.get('ai_t', float('nan')):+.2f} -- not distinguishable from zero")
            if r.get("beta_collinear", False):
                why.append(f"max VIF={r.get('max_vif', float('nan')):.1f} -- beta not cleanly identified")
            if r.get("semantic_thin", False):
                why.append(f"only {int(r['n_substantive'])} substantive units -- score is mostly prior")
            print("  MEASUREMENT is live: " + "; ".join(why))
            print("  Rule this out before claiming a channel or a mispricing.")
        else:
            print("  Measurement looks sound on both sides: beta is identified and the")
            print("  semantic score rests on enough text. So this is CHANNEL or MISPRICING.")

        print("\n  semantic trajectory:")
        print(trajectory(t).tail(6).to_string())
        ev = evidence(t, k=2)
        if len(ev):
            print("\n  what the filings actually say:")
            for _, e in ev.iterrows():
                print(f"    [{e.get('period','')} {e.get('section_name','')} "
                      f"{e.get('valence_ai', 0):+.0f}] {str(e.get('evidence_span',''))[:130]}")

    print("\n" + "=" * 72)
    print("Write one testable hypothesis per name. A hypothesis names what you would")
    print("measure next and what would falsify it -- not 'the market has not caught up'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
