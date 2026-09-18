"""The AI-Association Index. Task 04 -- the central structural call of Part 1.

    AIA = salience_shrunk x valence_shrunk

Two normalisations, because they answer different questions and the brief asks for both
a comparable-over-time index and a cross-sectional ranking:

  aia_z            cross-sectional z-score within each period. Removes any quarter when
                   the whole universe talks about AI more, so a company only moves if it
                   moves RELATIVE TO ITS PEERS. This is the series to compare against
                   factor betas in Part 3 -- both sides then live in the same units.

  excess_salience  salience minus the cross-sectional median that period. The "index
                   against a baseline and/or peer set" the brief asks for, kept on the
                   raw scale so it reads in percentage points.

WHY A PRODUCT, AND WHY THE PAIR IS ALSO REPORTED. Multiplying makes the index signed:
heavy negative AI coverage (ADBE's risk factors) and heavy positive coverage (NVDA's
MD&A) are opposite exposures, not the same "high AI association". But the product
collapses two facts into one number -- AIA = 0 means either no AI coverage or perfectly
balanced coverage, and those are very different companies. So salience and valence are
carried through as columns and reported as a pair, per README.md.

SCOPE. SEC filings only for the time series. The news half covers 2026 alone, because
Alpha Vantage returns the most recent N articles regardless of time_from and quarterly
re-querying returned empty feeds before 2026. `--with-news` produces the 2026
cross-section separately; it must not be blended into the series.

    python -m src.index.aia_index
    python -m src.index.aia_index --with-news     # 2026 cross-section only
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils import paths

from src.index import salience as sal_mod
from src.index import valence as val_mod

IN = paths.CLASSIFIED
OUT = paths.AIA_INDEX
RANKING = paths.AIA_RANKING


def cross_sectional_z(df: pd.DataFrame, col: str, out: str | None = None) -> pd.DataFrame:
    """z-score within each period, across the nine names.

    Population std (ddof=0): these nine ARE the universe, not a sample from one. With a
    single name in a period, or zero dispersion, z is 0 rather than NaN -- a lone
    observation is not an outlier.
    """
    out = out or f"{col}_z"
    g = df.groupby("period", observed=True)[col]
    df[out] = ((df[col] - g.transform("mean"))
               / g.transform(lambda s: s.std(ddof=0)).replace(0, pd.NA)).fillna(0.0)
    return df


def excess_salience(df: pd.DataFrame) -> pd.DataFrame:
    """Salience relative to the peer median that period, on the raw scale."""
    med = df.groupby("period", observed=True)["salience_shrunk"].transform("median")
    df["excess_salience"] = df["salience_shrunk"] - med
    return df


def build(docs: pd.DataFrame, sources: tuple[str, ...] = ("sec_filing",)) -> pd.DataFrame:
    s = sal_mod.shrink(sal_mod.raw_salience(docs, sources))
    v = val_mod.shrink(val_mod.raw_valence(docs, sources))

    df = s.merge(v.drop(columns=["n_substantive"]), on=["ticker", "period"], how="left")
    # A ticker-quarter with coverage but nothing substantive has no valence to average.
    # Neutral is the correct reading, and it is what full shrinkage would give anyway.
    df["valence_raw"] = df["valence_raw"].fillna(0.0)
    df["valence_shrunk"] = df["valence_shrunk"].fillna(0.0)
    df["lam_v"] = df["lam_v"].fillna(0.0)

    df["aia"] = df["salience_shrunk"] * df["valence_shrunk"]
    df = cross_sectional_z(df, "aia")
    df = cross_sectional_z(df, "salience_shrunk", "salience_z")
    df = excess_salience(df)
    return df.sort_values(["ticker", "period"]).reset_index(drop=True)


def current_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Latest period, ranked. The cross-sectional answer the brief asks for."""
    last = df["period"].max()
    cur = df[df["period"] == last].copy()
    cols = ["ticker", "salience_shrunk", "valence_shrunk", "aia", "aia_z",
            "excess_salience", "n_units", "n_substantive"]
    return cur[cols].sort_values("aia", ascending=False).reset_index(drop=True), last


def trailing_ranking(df: pd.DataFrame, quarters: int = 4) -> pd.DataFrame:
    """Mean over the last N quarters.

    More honest than a single quarter as "the current reading": one 10-K lands in one
    quarter, so a company's Q-on-Q salience swings with its filing calendar rather than
    with its AI exposure. Four quarters covers exactly one 10-K plus three 10-Qs for
    every name.
    """
    periods = sorted(df["period"].unique())[-quarters:]
    w = df[df["period"].isin(periods)]
    return (w.groupby("ticker")
             .agg(salience=("salience_shrunk", "mean"), valence=("valence_shrunk", "mean"),
                  aia=("aia", "mean"), aia_z=("aia_z", "mean"),
                  n_substantive=("n_substantive", "sum"))
             .sort_values("aia", ascending=False).round(4)), periods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-news", action="store_true",
                    help="include news; valid for the 2026 cross-section only")
    a = ap.parse_args()

    docs = pd.read_parquet(IN)
    sources = ("sec_filing", "news") if a.with_news else ("sec_filing",)
    if a.with_news:
        print("*** news included -- 2026 only, NOT a time series. Cross-section only. ***\n")

    df = build(docs, sources)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} ticker-quarters, {df.period.nunique()} periods)")

    cur, last = current_ranking(df)
    print(f"\n=== cross-sectional ranking, {last} ===")
    print(cur.round(4).to_string(index=False))

    tr, periods = trailing_ranking(df)
    print(f"\n=== trailing 4 quarters ({periods[0]}..{periods[-1]}) -- "
          f"the headline ranking ===")
    print(tr.to_string())

    import pathlib
    pathlib.Path(RANKING).parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(RANKING)
    print(f"\nwrote {RANKING}")

    print("\n=== salience time series (shrunk) ===")
    print(df.pivot_table(index="period", columns="ticker", values="salience_shrunk")
            .round(3).to_string())

    print("\n=== the pair, trailing 4q -- read these together, not just AIA ===")
    print("high salience + positive valence = AI tailwind priced as opportunity")
    print("high salience + negative valence = AI discussed as a threat to the business")
    print("low salience                     = the company barely engages the theme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
