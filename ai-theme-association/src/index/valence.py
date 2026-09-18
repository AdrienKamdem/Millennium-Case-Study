"""Valence: how positive the AI coverage is, conditioned on the AI aspect.

Computed over SUBSTANTIVE units only. Averaging valence across the whole corpus would
drown the signal in the ~80% of chunks that have no AI content and score 0 by
construction -- the mean would measure the AI share, not the direction.

This is the aspect-based half of Task 03 and it is doing real work. Measured on the
full corpus:

    risk_factors          -0.83   (415 chunks)
    financial_statements  +1.19
    mdna                  +1.47
    business              +1.68   (160 chunks)

A 2.5-point spread inside the same filings. Document-level sentiment cannot produce
that -- a 10-K's overall tone does not flip between Item 1A and Item 1. ADBE and TEAM
score -0.98 and -0.97 in risk factors while sitting positive overall, which is the
company disclosing AI as a competitive threat in the one section where it must.

SHRINKAGE TOWARD ZERO, not toward the peer mean. A single substantive chunk scoring +2
is not evidence a company's AI narrative is positive; the honest prior for "we know
almost nothing" is neutral. lambda = n/(n+k) on the SUBSTANTIVE count -- a different n
from salience, which uses total units, because valence is only informed by the chunks
that actually discuss AI. PG has 15 substantive units across four years and WM has 25,
so both land close to 0 whatever their raw mean says.

    python -m src.index.valence
"""

from __future__ import annotations

import pandas as pd

from src.utils import config, paths
from src.utils import paths

IN = paths.CLASSIFIED
DEFAULT_K = 10


def _params() -> tuple[dict, int, float]:
    t = config.load("taxonomy")
    sh = t.get("shrinkage", {})
    return (t.get("materiality_weights", {}),
            int(sh.get("k", DEFAULT_K)),
            float(sh.get("valence_target", 0.0)))


def raw_valence(docs: pd.DataFrame, sources: tuple[str, ...] = ("sec_filing",),
                weight_by_materiality: bool = True) -> pd.DataFrame:
    """Materiality-weighted mean valence over substantive units, per ticker-period.

    Weighting by materiality means a high-materiality product launch counts for five
    times a low-materiality passing mention (1.0 vs 0.2 in taxonomy.yaml), which is the
    same judgement salience applies to its numerator. Set False for an unweighted mean
    if you want to show the two side by side in the appendix.
    """
    mw, _, _ = _params()
    d = docs[(docs["source_category"].isin(sources))
             & (docs["ai_relevance"] == "substantive")].copy()
    if d.empty:
        return pd.DataFrame(columns=["ticker", "period", "n_substantive", "valence_raw"])

    d["w"] = d["materiality"].map(mw).fillna(0.0) if weight_by_materiality else 1.0
    g = d.groupby(["ticker", "period"], observed=True)
    out = pd.DataFrame({
        "n_substantive": g.size(),
        "wsum": g["w"].sum(),
        "vsum": g.apply(lambda x: (x["valence_ai"] * x["w"]).sum(), include_groups=False),
    }).reset_index()
    # wsum can be 0 if every unit in a cell is low-materiality and weights are 0.
    out["valence_raw"] = (out["vsum"] / out["wsum"].replace(0, pd.NA)).fillna(0.0)
    return out.sort_values(["ticker", "period"]).reset_index(drop=True)


def shrink(val: pd.DataFrame, k: int | None = None, target: float | None = None) -> pd.DataFrame:
    """Shrink toward neutral. n is the SUBSTANTIVE count, not the unit count."""
    if k is None or target is None:
        _, k_cfg, t_cfg = _params()
        k = k_cfg if k is None else k
        target = t_cfg if target is None else target

    out = val.copy()
    out["lam_v"] = out["n_substantive"] / (out["n_substantive"] + k)
    out["valence_shrunk"] = out["lam_v"] * out["valence_raw"] + (1 - out["lam_v"]) * target
    return out


def section_evidence(docs: pd.DataFrame) -> pd.DataFrame:
    """Valence by section -- the evidence that the conditioning worked.

    Task 03 asks you to state the validation approach. This table IS it: if sentiment
    were reading document tone rather than the AI aspect, risk factors and business
    could not differ by 2.5 points inside the same filings.
    """
    s = docs[(docs["ai_relevance"] == "substantive")
             & (docs["source_category"] == "sec_filing")
             & (docs["section_name"] != "")]
    t = s.groupby("section_name")["valence_ai"].agg(["mean", "count"])
    return t[t["count"] >= 15].sort_values("mean").round(2)


def main() -> int:
    docs = pd.read_parquet(IN)
    val = shrink(raw_valence(docs))
    val.to_parquet(paths.VALENCE, index=False)
    print(f"wrote data/processed/valence.parquet  ({len(val)} ticker-quarters)")

    print("\nvalence by section -- the aspect-conditioning evidence:")
    print(section_evidence(docs).to_string())

    print("\nmean shrunk valence by ticker:")
    print(val.groupby("ticker")[["valence_raw", "valence_shrunk", "n_substantive"]]
             .agg({"valence_raw": "mean", "valence_shrunk": "mean", "n_substantive": "sum"})
             .sort_values("valence_shrunk").round(3).to_string())
    print("\nThin names shrink hardest toward 0 -- that is the point, not a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
