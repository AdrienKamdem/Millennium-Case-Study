"""Salience: what share of a company's coverage is AI-related.

A SHARE, not a count. NVDA files more pages than WM and is written about more often;
counting AI mentions would rank on size. Dividing by the company's own total volume in
the same period is what the brief means by "controlled for total coverage volume so
that a company mentioned more simply because it is larger does not appear to be
improving".

  numerator   sum over SUBSTANTIVE units of source_weight x materiality_weight
  denominator sum over ALL units of source_weight

Materiality weights the numerator only, so the measure is bounded by [0, 1] but reaches
1 only if every unit is substantive AND high-materiality. That asymmetry is deliberate:
a quarter of low-materiality AI mentions should not score like a quarter of product
launches.

Gated-out units are in the denominator. They were never sent to the model because they
contain no AI term, which is the classification `none` by construction -- dropping them
would shrink the denominator and inflate every score, the exact failure prefilter's
docstring warns about.

SHRINKAGE. Coverage is thin: 27 of 135 ticker-quarters carry fewer than 5 substantive
chunks, and CAT has exactly 1 in eight separate quarters. A raw share over 1 chunk is
noise. Empirical-Bayes with lambda = n/(n+k) pulls thin cells toward a prior; k=10 from
config/taxonomy.yaml means a cell needs 10 units to get half its own weight.

The prior is the COMPANY's own trailing mean, not the cross-sectional mean. That matters
here: CAT's true salience really is near zero, and shrinking it toward the nine-name
average would manufacture AI exposure it does not have.

    python -m src.index.salience
"""

from __future__ import annotations

import pandas as pd

from src.utils import config, paths
from src.utils import paths

IN = paths.CLASSIFIED
DEFAULT_K = 10
TRAILING_QUARTERS = 4          # "trailing_12m_company_mean" in taxonomy.yaml


def _weights() -> tuple[dict, dict, int]:
    t = config.load("taxonomy")
    return (t.get("source_weights", {}), t.get("materiality_weights", {}),
            int(t.get("shrinkage", {}).get("k", DEFAULT_K)))


def raw_salience(docs: pd.DataFrame, sources: tuple[str, ...] = ("sec_filing",)) -> pd.DataFrame:
    """Weighted substantive share per ticker-period.

    `sources` defaults to SEC only. The news half of the corpus covers 2026 alone
    (Alpha Vantage returns the most recent N regardless of time_from), so including it
    would compare a four-year SEC series against a one-quarter news series. Pass
    ("sec_filing", "news") for the 2026 cross-section, never for the time series.
    """
    sw, mw, _ = _weights()
    d = docs[docs["source_category"].isin(sources)].copy()

    d["w_src"] = d["source_category"].map(sw).fillna(1.0)
    d["w_mat"] = d["materiality"].map(mw).fillna(0.0)
    d["is_sub"] = d["ai_relevance"].eq("substantive")
    # A failed request is not evidence of absence. Excluding it from both sides keeps
    # the share honest; at 0.28% it moves nothing either way.
    d = d[d["ai_relevance"].notna()]

    g = d.groupby(["ticker", "period"], observed=True)
    out = pd.DataFrame({
        "n_units": g.size(),
        "n_substantive": g["is_sub"].sum(),
        "den": g["w_src"].sum(),
        "num": g.apply(lambda x: (x["w_src"] * x["w_mat"] * x["is_sub"]).sum(),
                       include_groups=False),
    }).reset_index()
    out["salience_raw"] = (out["num"] / out["den"]).fillna(0.0)
    return out.sort_values(["ticker", "period"]).reset_index(drop=True)


def shrink(sal: pd.DataFrame, k: int | None = None,
           trailing: int = TRAILING_QUARTERS) -> pd.DataFrame:
    """Empirical-Bayes shrinkage toward the company's own trailing mean.

    lambda = n/(n+k) on the unit count, so a 1-chunk quarter keeps ~9% of its own
    reading and a 100-chunk quarter keeps ~91%. Before a company has any history the
    prior falls back to its full-sample mean -- still its own level, never the peer
    average, which would invent exposure for CAT, PG and WM.
    """
    if k is None:
        _, _, k = _weights()
    out = sal.sort_values(["ticker", "period"]).copy()

    prior = (out.groupby("ticker", observed=True)["salience_raw"]
                .transform(lambda s: s.shift(1).rolling(trailing, min_periods=1).mean()))
    full = out.groupby("ticker", observed=True)["salience_raw"].transform("mean")
    out["prior"] = prior.fillna(full)

    out["lam"] = out["n_units"] / (out["n_units"] + k)
    out["salience_shrunk"] = out["lam"] * out["salience_raw"] + (1 - out["lam"]) * out["prior"]
    return out


def coverage_check(sal: pd.DataFrame) -> None:
    """How much of the panel is thin enough that shrinkage is doing the work."""
    thin = sal[sal["n_units"] < 10]
    print(f"\nticker-quarters: {len(sal)}  |  under 10 units: {len(thin)} "
          f"({len(thin)/max(len(sal),1):.0%}) -- these are mostly prior, not data")
    moved = (sal["salience_shrunk"] - sal["salience_raw"]).abs()
    print(f"median |shrinkage move|: {moved.median():.4f}  max: {moved.max():.4f}")


def main() -> int:
    docs = pd.read_parquet(IN)
    sal = shrink(raw_salience(docs))
    sal.to_parquet(paths.SALIENCE, index=False)
    print(f"wrote data/processed/salience.parquet  ({len(sal)} ticker-quarters)")
    coverage_check(sal)
    print("\nmean shrunk salience by ticker:")
    print(sal.groupby("ticker")["salience_shrunk"].mean().sort_values(ascending=False)
             .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
