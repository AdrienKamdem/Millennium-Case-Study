# """AI factor construction. The central structural call of Part 2.

# AIF-1 (primary):  mean(AI pure-play basket) - XLK, then residualised against
#                   MKT / SMB / HML / RMW / CMA / MOM / XLK.

#   Why long-short vs XLK: it pre-empts the obvious objection that your "AI factor"
#   is just tech beta wearing a hat.

#   Why residualise: if the factor still contains market and sector movement, every
#   tech stock loads on it and you have measured nothing. The residual is the part
#   of AI-basket performance that market, style and sector CANNOT explain. Loading
#   on that is a real statement.

#   HARD CONSTRAINT: none of the 9 test names in any leg. Put NVDA in the basket and
#   then read NVDA's beta off it and the result is circular by construction.

#   Report the first-stage R^2 honestly. If residualising strips 85% of the variance,
#   say so -- the betas are then estimated off a thin slice and the standard errors
#   should reflect it.

# AIF-2 (robustness): a thematic ETF, TAKEN APART rather than used blind. The brief
#   says examine what your ingredient is actually made of, so compute AIQ's
#   correlation with XLK and SPY and its R^2 on MKT+XLK. A high R^2 means the ETF is
#   largely a mega-cap tech repackage with little independent variance -- a negative
#   result, and a better slide than any coefficient.

# CIRCULARITY: every factor here is PRICE-derived, never text-derived. That is what
# keeps Part 1 and Part 2 independent instruments and makes the reconciliation mean
# something. State it as the reason a text-intensity factor was rejected.

#     python -m src.factor.ai_factor
# """

# from __future__ import annotations

# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm

# from src.utils import config, paths

# OUT = paths.AI_FACTOR
# STYLE = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


# def build_aif1(rets: pd.DataFrame, basket: list[str] | None = None) -> pd.Series:
#     """Equal-weighted AI pure-plays minus XLK, daily.

#     Basket comes from config/ai_basket.yaml -- single source of truth, and it
#     means changing the factor legs never requires touching code.
#     """
#     basket = basket or config.load("ai_basket")["aif1"]["long"]
#     leak = set(basket) & set(config.tickers())
#     if leak:
#         raise ValueError(f"CIRCULARITY: test names in the factor basket: {leak}")

#     available = [t for t in basket if t in rets.columns and rets[t].notna().sum() > 100]
#     dropped = set(basket) - set(available)
#     if dropped:
#         print(f"  basket names dropped for insufficient history: {sorted(dropped)}")
#     print(f"  long leg ({len(available)}): {available}")

#     # Names like ARM and PLTR list mid-window; mean over available names each day
#     # keeps the series continuous rather than starting it at the last IPO.
#     long_leg = rets[available].mean(axis=1, skipna=True)
#     return (long_leg - rets["XLK"]).rename("AIF_raw").dropna()


# def orthogonalize(factor: pd.Series, controls: pd.DataFrame) -> tuple[pd.Series, float, pd.Series]:
#     """Residualise the factor. Returns (residual, first_stage_R2, loadings)."""
#     df = pd.concat([factor, controls], axis=1).dropna()
#     y, X = df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])
#     fit = sm.OLS(y, X).fit()
#     resid = fit.resid.rename("AIF_orth")
#     return resid, float(fit.rsquared), fit.params.drop("const")


# def orthogonalize_sectors(rets: pd.DataFrame, ff: pd.DataFrame,
#                           etfs: list[str]) -> pd.DataFrame:
#     """Sector ETF excess returns residualised against MKT + styles.

#     Done so the sector control does not absorb market beta, which would leave the
#     market coefficient meaningless and distort everything downstream.
#     """
#     out = {}
#     for e in etfs:
#         excess = (rets[e] - ff["RF"]).dropna()
#         df = pd.concat([excess.rename("y"), ff[STYLE]], axis=1).dropna()
#         fit = sm.OLS(df["y"], sm.add_constant(df[STYLE])).fit()
#         out[f"{e}_orth"] = fit.resid
#         print(f"  {e}: R^2 on market+styles = {fit.rsquared:.3f} "
#               f"-> {1 - fit.rsquared:.0%} of its variance is sector-specific")
#     return pd.DataFrame(out)


# def teardown_etf(rets: pd.DataFrame, ff: pd.DataFrame, etf: str = "AIQ") -> dict:
#     """'Examine what your chosen ingredient is actually made of.'"""
#     if etf not in rets.columns:
#         return {}
#     print(f"\n--- {etf} teardown ---")
#     cols = [c for c in (etf, "XLK", "SPY") if c in rets.columns]
#     corr = rets[cols].corr()
#     print(f"  correlations:\n{corr.round(3).to_string()}")

#     df = pd.concat([(rets[etf] - ff["RF"]).rename("y"),
#                     ff[STYLE], (rets["XLK"] - ff["RF"]).rename("XLK_ex")], axis=1).dropna()
#     fit = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit()
#     print(f"  R^2 of {etf} on market+styles+XLK = {fit.rsquared:.3f}")
#     verdict = ("mostly a tech/market repackage with little independent variance"
#                if fit.rsquared > 0.85 else "retains meaningful independent variance")
#     print(f"  -> {verdict}")
#     return {"corr": corr, "r2_on_mkt_xlk": float(fit.rsquared), "verdict": verdict}


# def build() -> pd.DataFrame:
#     from src.factor.ff_factors import load as load_ff
#     start, end = config.window()
#     rets = pd.read_parquet(paths.PRICES)
#     rets.index = pd.to_datetime(rets.index)
#     if getattr(rets.index, "tz", None) is not None:
#         rets.index = rets.index.tz_localize(None)
#     ff = load_ff(start, end)
#     ff.index = pd.to_datetime(ff.index)

#     print("\n--- AIF-1: pure-play basket minus XLK ---")
#     raw = build_aif1(rets)

#     controls = pd.concat([ff[STYLE], (rets["XLK"] - ff["RF"]).rename("XLK_ex")], axis=1)
#     orth, r2, loadings = orthogonalize(raw, controls)
#     print(f"  first-stage R^2 = {r2:.3f}  -> {1 - r2:.0%} of the AI factor's variance "
#           f"survives as AI-specific")
#     if r2 > 0.85:
#         print("  NOTE: most variance is explained away. Betas come off a thin residual; "
#               "say so and let the standard errors speak.")
#     print(f"  loadings:\n{loadings.round(3).to_string()}")

#     print("\n--- sector controls ---")
#     sectors = orthogonalize_sectors(rets, ff, ["XLK", "XLI", "XLP"])

#     teardown_etf(rets, ff, "AIQ")

#     out = pd.concat([raw, orth, ff[STYLE + ["RF"]], sectors], axis=1).dropna()
#     OUT.parent.mkdir(parents=True, exist_ok=True)
#     out.to_parquet(OUT)
#     print(f"\nwrote {OUT}  {out.shape[0]} days x {list(out.columns)}")
#     print(f"\nAIF_orth annualised vol: {out['AIF_orth'].std() * np.sqrt(252):.1%}")
#     return out


# if __name__ == "__main__":
#     build()
#     sys.exit(0)

"""AI factor construction. The central structural call of Part 2.

AIF-1 (primary):  mean(AI pure-play basket) - XLK, then residualised against
                  MKT / SMB / HML / RMW / CMA / MOM / XLK.

  Why long-short vs XLK: it pre-empts the obvious objection that your "AI factor"
  is just tech beta wearing a hat.

  Why residualise: if the factor still contains market and sector movement, every
  tech stock loads on it and you have measured nothing. The residual is the part
  of AI-basket performance that market, style and sector CANNOT explain. Loading
  on that is a real statement.

  HARD CONSTRAINT: none of the 9 test names in any leg. Put NVDA in the basket and
  then read NVDA's beta off it and the result is circular by construction.

  Report the first-stage R^2 honestly. If residualising strips 85% of the variance,
  say so -- the betas are then estimated off a thin slice and the standard errors
  should reflect it.

AIF-2 (robustness): a thematic ETF, TAKEN APART rather than used blind. The brief
  says examine what your ingredient is actually made of, so compute AIQ's
  correlation with XLK and SPY and its R^2 on MKT+XLK. A high R^2 means the ETF is
  largely a mega-cap tech repackage with little independent variance -- a negative
  result, and a better slide than any coefficient.

CIRCULARITY: every factor here is PRICE-derived, never text-derived. That is what
keeps Part 1 and Part 2 independent instruments and makes the reconciliation mean
something. State it as the reason a text-intensity factor was rejected.

    python -m src.factor.ai_factor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils import config
from src.utils import paths

OUT = paths.AI_FACTOR
OUT_TXT = paths.AI_FACTOR_TXT

STYLE = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def build_aif1(rets: pd.DataFrame, basket: list[str] | None = None) -> pd.Series:
    """Equal-weighted AI pure-plays minus XLK, daily.

    Basket comes from config/ai_basket.yaml -- single source of truth, and it
    means changing the factor legs never requires touching code.
    """
    basket = basket or config.load("ai_basket")["aif1"]["long"]

    leak = set(basket) & set(config.tickers())

    if leak:
        raise ValueError(
            f"CIRCULARITY: test names in the factor basket: {leak}"
        )

    available = [
        t
        for t in basket
        if t in rets.columns and rets[t].notna().sum() > 100
    ]

    dropped = set(basket) - set(available)

    if dropped:
        print(
            f"  basket names dropped for insufficient history: "
            f"{sorted(dropped)}"
        )

    print(
        f"  long leg ({len(available)}): {available}"
    )

    # Names like ARM and PLTR list mid-window; mean over available names each day
    # keeps the series continuous rather than starting it at the last IPO.
    long_leg = rets[available].mean(
        axis=1,
        skipna=True,
    )

    return (
        long_leg - rets["XLK"]
    ).rename("AIF_raw").dropna()


def orthogonalize(
    factor: pd.Series,
    controls: pd.DataFrame,
) -> tuple[pd.Series, float, pd.Series]:
    """Residualise the factor. Returns (residual, first_stage_R2, loadings)."""

    df = pd.concat(
        [factor, controls],
        axis=1,
    ).dropna()

    y = df.iloc[:, 0]

    X = sm.add_constant(
        df.iloc[:, 1:]
    )

    fit = sm.OLS(
        y,
        X,
    ).fit()

    resid = fit.resid.rename(
        "AIF_orth"
    )

    return (
        resid,
        float(fit.rsquared),
        fit.params.drop("const"),
    )


def orthogonalize_sectors(
    rets: pd.DataFrame,
    ff: pd.DataFrame,
    etfs: list[str],
) -> pd.DataFrame:
    """Sector ETF excess returns residualised against MKT + styles.

    Done so the sector control does not absorb market beta, which would leave the
    market coefficient meaningless and distort everything downstream.
    """

    out = {}

    for e in etfs:

        excess = (
            rets[e] - ff["RF"]
        ).dropna()

        df = pd.concat(
            [
                excess.rename("y"),
                ff[STYLE],
            ],
            axis=1,
        ).dropna()

        fit = sm.OLS(
            df["y"],
            sm.add_constant(df[STYLE]),
        ).fit()

        out[f"{e}_orth"] = fit.resid

        print(
            f"  {e}: R^2 on market+styles = "
            f"{fit.rsquared:.3f} "
            f"-> {1 - fit.rsquared:.0%} "
            f"of its variance is sector-specific"
        )

    return pd.DataFrame(out)


def teardown_etf(
    rets: pd.DataFrame,
    ff: pd.DataFrame,
    etf: str = "AIQ",
) -> dict:
    """'Examine what your chosen ingredient is actually made of.'"""

    if etf not in rets.columns:
        return {}

    print(
        f"\n--- {etf} teardown ---"
    )

    cols = [
        c
        for c in (etf, "XLK", "SPY")
        if c in rets.columns
    ]

    corr = rets[cols].corr()

    print(
        f"  correlations:\n"
        f"{corr.round(3).to_string()}"
    )

    df = pd.concat(
        [
            (rets[etf] - ff["RF"]).rename("y"),
            ff[STYLE],
            (rets["XLK"] - ff["RF"]).rename("XLK_ex"),
        ],
        axis=1,
    ).dropna()

    fit = sm.OLS(
        df["y"],
        sm.add_constant(
            df.drop(columns="y")
        ),
    ).fit()

    print(
        f"  R^2 of {etf} on market+styles+XLK = "
        f"{fit.rsquared:.3f}"
    )

    verdict = (
        "mostly a tech/market repackage with little independent variance"
        if fit.rsquared > 0.85
        else "retains meaningful independent variance"
    )

    print(
        f"  -> {verdict}"
    )

    return {
        "corr": corr,
        "r2_on_mkt_xlk": float(fit.rsquared),
        "verdict": verdict,
    }


def build() -> pd.DataFrame:

    from src.factor.ff_factors import load as load_ff

    start, end = config.window()

    rets = pd.read_parquet(
        paths.PRICES
    )

    rets.index = pd.to_datetime(
        rets.index
    )

    if getattr(rets.index, "tz", None) is not None:
        rets.index = rets.index.tz_localize(None)

    ff = load_ff(
        start,
        end,
    )

    ff.index = pd.to_datetime(
        ff.index
    )

    print(
        "\n--- AIF-1: pure-play basket minus XLK ---"
    )

    raw = build_aif1(
        rets
    )

    controls = pd.concat(
        [
            ff[STYLE],
            (
                rets["XLK"] - ff["RF"]
            ).rename("XLK_ex"),
        ],
        axis=1,
    )

    orth, r2, loadings = orthogonalize(
        raw,
        controls,
    )

    print(
        f"  first-stage R^2 = {r2:.3f}  "
        f"-> {1 - r2:.0%} of the AI factor's variance "
        f"survives as AI-specific"
    )

    if r2 > 0.85:
        print(
            "  NOTE: most variance is explained away. "
            "Betas come off a thin residual; "
            "say so and let the standard errors speak."
        )

    print(
        f"  loadings:\n"
        f"{loadings.round(3).to_string()}"
    )

    print(
        "\n--- sector controls ---"
    )

    sectors = orthogonalize_sectors(
        rets,
        ff,
        ["XLK", "XLI", "XLP"],
    )

    teardown_etf(
        rets,
        ff,
        "AIQ",
    )

    out = pd.concat(
        [
            raw,
            orth,
            ff[STYLE + ["RF"]],
            sectors,
        ],
        axis=1,
    ).dropna()

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Machine-readable version used by the pipeline.
    out.to_parquet(
        OUT
    )

    # Human-readable version of the same dataset.
    out.to_csv(
        OUT_TXT,
        sep="\t",
        index=True,
        index_label="Date",
        float_format="%.8f",
    )

    print(
        f"\nwrote {OUT}  "
        f"{out.shape[0]} days x {list(out.columns)}"
    )

    print(
        f"wrote {OUT_TXT}"
    )

    print(
        f"\nAIF_orth annualised vol: "
        f"{out['AIF_orth'].std() * np.sqrt(252):.1%}"
    )

    return out


if __name__ == "__main__":
    build()
    sys.exit(0)