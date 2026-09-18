"""Put the semantic score and the factor beta on a comparable footing.

You cannot compare a raw AIA of 0.24 to a raw AI beta of 0.31 -- different units,
different scales, different generating processes. Both sides become cross-sectional
z-scores over the same nine names, which is the brief's "e.g. cross-sectional z-scores
or percentile ranks per period".

THE SEMANTIC SIDE IS A TRAILING MEAN, NOT THE LAST QUARTER. A 10-K lands in one
quarter, so single-quarter salience tracks the filing calendar rather than AI exposure
-- MU shows 4 units in 2026Q3 against NVDA's 79. Four quarters covers one 10-K plus
three 10-Qs for every name and removes that artifact.

THE BETA SIDE IS FULL-SAMPLE. regressions.py fits one beta per ticker over the whole
window, so it is already a single number. Mixing a full-sample beta against a
trailing-4-quarter semantic score is a real mismatch and belongs in the limitations:
the semantic side can move within the window and the beta cannot.

RANK CORRELATION IS DESCRIPTIVE ONLY. n=9. The brief says so explicitly -- "at this
universe size, treat the reconciliation as case-based, not statistical". The bootstrap
CI is reported to show how wide it is, not to claim significance.

    python -m src.reconcile.align
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths
from scipy import stats

INDEX = paths.AIA_INDEX
# regressions.py writes to data/processed; the repo also carries a data/FACTOR copy.
BETA_PATHS = [paths.AI_BETAS]
OUT = paths.RECONCILIATION

TRAILING_QUARTERS = 4


def _betas() -> pd.DataFrame:
    for p in BETA_PATHS:
        if p.exists():
            b = pd.read_parquet(p)
            if b.index.name != "ticker" and "ticker" in b.columns:
                b = b.set_index("ticker")
            print(f"betas from {p}")
            return b
    raise FileNotFoundError(f"no ai_betas.parquet in {[str(p) for p in BETA_PATHS]} "
                            "-- run `make factor` first")


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


def align(quarters: int = TRAILING_QUARTERS) -> pd.DataFrame:
    """One row per ticker: both scores, both z-scores, and the gap between them."""
    idx = pd.read_parquet(INDEX)
    periods = sorted(idx["period"].unique())[-quarters:]
    sem = (idx[idx["period"].isin(periods)]
           .groupby("ticker")
           .agg(salience=("salience_shrunk", "mean"),
                valence=("valence_shrunk", "mean"),
                aia=("aia", "mean"),
                n_substantive=("n_substantive", "sum")))

    b = _betas()
    keep = [c for c in ("ai_beta", "ai_se", "ai_t", "ai_p", "max_vif") if c in b.columns]
    df = sem.join(b[keep], how="inner")

    df["semantic_z"] = _z(df["aia"])
    df["beta_z"] = _z(df["ai_beta"])
    # Positive gap: the text says more about AI than the market prices into the stock.
    df["gap"] = df["semantic_z"] - df["beta_z"]
    df["abs_gap"] = df["gap"].abs()
    df.attrs["periods"] = periods
    return df.sort_values("semantic_z", ascending=False)


def rank_correlation(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """Spearman rho with a bootstrap CI. Descriptive at n=9 -- read the width, not the p."""
    rho, p = stats.spearmanr(df["semantic_z"], df["beta_z"])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(df), replace=True)
        if len(set(s)) < 3:
            continue
        r, _p = stats.spearmanr(df["semantic_z"].values[s], df["beta_z"].values[s])
        if not np.isnan(r):
            boots.append(r)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return {"rho": rho, "p": p, "ci_low": lo, "ci_high": hi, "n": len(df)}


def main() -> int:
    df = align()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"wrote {OUT}  ({len(df)} tickers, semantic side = "
          f"{df.attrs['periods'][0]}..{df.attrs['periods'][-1]})\n")

    cols = ["aia", "semantic_z", "ai_beta", "beta_z", "gap", "n_substantive"]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string())

    rc = rank_correlation(df)
    print(f"\nSpearman rho = {rc['rho']:+.3f}  (p={rc['p']:.3f}, n={rc['n']})")
    print(f"bootstrap 95% CI: [{rc['ci_low']:+.3f}, {rc['ci_high']:+.3f}]")
    print("\nDESCRIPTIVE ONLY. At n=9 that interval is wide enough to contain almost")
    print("any story; the brief asks for case-based reconciliation, not a statistic.")
    print("The individual disagreements below are the result, not this number.")

    print("\nlargest disagreements (|gap|):")
    print(df.nlargest(3, "abs_gap")[["semantic_z", "beta_z", "gap"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
