# """Per-stock AI betas.

#     r_it - rf = a + b_MKT*MKT + b_SMB*SMB + b_HML*HML + b_RMW*RMW + b_CMA*CMA
#                 + b_MOM*MOM + g*SECTOR_orth + d*AIF_orth + e

# d is the AI BETA. Newey-West (HAC, 5 lags) standard errors, because daily equity
# residuals are heteroskedastic and mildly autocorrelated; plain OLS errors would
# overstate significance.

# RUN THE SANITY GATE FIRST. NVDA strongly positive; PG and WM indistinguishable
# from zero. The brief states this test explicitly. If it fails your factor is
# broken -- report that rather than shipping betas you do not believe. "My null
# names were not null, and here is what I think that means" is a better answer than
# a clean-looking table you cannot defend.

# VIFs are reported because NVDA will be collinear across MOM / XLK / AIF
# simultaneously. That is a finding, not an embarrassment. Note that d means
# "loading on the AI-specific residual", not "total AI sensitivity".

# ROLLING BETAS matter: a single full-sample number cannot be reconciled against a
# time-varying semantic index, and cannot feed the Task 7 lead-lag design.

#     python -m src.factor.regressions
# """

# from __future__ import annotations

# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# from src.utils import config, paths

# STYLE = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
# HAC_LAGS = 5
# ROLL = 126
# OUT = paths.AI_BETAS


# def _design(ticker: str, rets: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
#     sector = config.sector_map()[ticker]
#     df = pd.concat([
#         (rets[ticker] - fac["RF"]).rename("excess"),
#         fac[STYLE],
#         fac[f"{sector}_orth"].rename("SECTOR"),
#         fac["AIF_orth"].rename("AIF"),
#     ], axis=1).dropna()
#     return df


# def run(rets: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
#     rows = []
#     for t in config.tickers():
#         if t not in rets.columns:
#             print(f"  {t}: MISSING from price data")
#             continue
#         df = _design(t, rets, fac)
#         X = sm.add_constant(df.drop(columns="excess"))
#         fit = sm.OLS(df["excess"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

#         vifs = {X.columns[i]: variance_inflation_factor(X.values, i)
#                 for i in range(1, X.shape[1])}
#         rows.append({
#             "ticker": t,
#             "ai_beta": fit.params["AIF"],
#             "ai_se": fit.bse["AIF"],
#             "ai_t": fit.tvalues["AIF"],
#             "ai_p": fit.pvalues["AIF"],
#             "ai_ci_lo": fit.conf_int().loc["AIF", 0],
#             "ai_ci_hi": fit.conf_int().loc["AIF", 1],
#             "mkt_beta": fit.params["Mkt-RF"],
#             "sector_beta": fit.params["SECTOR"],
#             "mom_beta": fit.params["MOM"],
#             "alpha_ann": fit.params["const"] * 252,
#             "r2": fit.rsquared,
#             "n_obs": int(fit.nobs),
#             "max_vif": max(vifs.values()),
#             "vif_AIF": vifs["AIF"],
#         })
#     return pd.DataFrame(rows).set_index("ticker")


# def sanity_gate(res: pd.DataFrame) -> dict:
#     """The gate the brief specifies. Print whatever it says, pass or fail."""
#     gate = config.load("ai_basket")["sanity_gate"]
#     out, ok = {}, True
#     print("\n--- SANITY GATE ---")
#     for t in gate["must_load_positive"]:
#         if t not in res.index:
#             continue
#         r = res.loc[t]
#         p = bool(r.ai_beta > 0 and r.ai_p < 0.05)
#         ok &= p
#         out[f"{t}_positive"] = p
#         print(f"  {t} loads positive : beta={r.ai_beta:+.3f} t={r.ai_t:+.2f} "
#               f"p={r.ai_p:.4f}  {'PASS' if p else 'FAIL'}")
#     for t in gate["must_be_indistinguishable_from_zero"]:
#         if t not in res.index:
#             continue
#         r = res.loc[t]
#         p = bool(r.ai_p > 0.05)
#         ok &= p
#         out[f"{t}_null"] = p
#         print(f"  {t} is null        : beta={r.ai_beta:+.3f} t={r.ai_t:+.2f} "
#               f"p={r.ai_p:.4f}  {'PASS' if p else 'FAIL'}")
#     out["all_passed"] = ok
#     print(f"  => {'GATE PASSED' if ok else 'GATE FAILED - report this, do not hide it'}")
#     return out


# def rolling_beta(rets: pd.DataFrame, fac: pd.DataFrame, window: int = ROLL) -> pd.DataFrame:
#     """126-day rolling AI beta per ticker.

#     Overlapping windows are mechanically autocorrelated -- fine for describing
#     how exposure evolves, but do NOT run naive significance tests on the series
#     (see the Task 7 design note).
#     """
#     out = {}
#     for t in config.tickers():
#         if t not in rets.columns:
#             continue
#         df = _design(t, rets, fac)
#         betas = {}
#         for i in range(window, len(df) + 1):
#             w = df.iloc[i - window:i]
#             fit = sm.OLS(w["excess"], sm.add_constant(w.drop(columns="excess"))).fit()
#             betas[df.index[i - 1]] = fit.params["AIF"]
#         out[t] = pd.Series(betas)
#         print(f"  {t}: {len(betas)} rolling estimates", flush=True)
#     return pd.DataFrame(out)


# def main() -> int:
#     rets = pd.read_parquet(paths.PRICES)
#     rets.index = pd.to_datetime(rets.index)
#     if getattr(rets.index, "tz", None) is not None:
#         rets.index = rets.index.tz_localize(None)
#     fac = pd.read_parquet(paths.AI_FACTOR)
#     fac.index = pd.to_datetime(fac.index)

#     res = run(rets, fac).sort_values("ai_beta", ascending=False)
#     print("\n--- AI BETAS ---")
#     print(res[["ai_beta", "ai_se", "ai_t", "ai_p", "mkt_beta", "sector_beta",
#                "r2", "vif_AIF"]].round(3).to_string())

#     sanity_gate(res)

#     hi = res[res.max_vif > 10]
#     if len(hi):
#         print(f"\nVIF > 10 for: {list(hi.index)} -- collinear with momentum/sector. "
#               f"Expected for mega-cap AI names; report it rather than dropping controls.")

#     OUT.parent.mkdir(parents=True, exist_ok=True)
#     res.to_parquet(OUT)
#     print(f"\nwrote {OUT}")

#     print(f"\n--- rolling {ROLL}-day betas ---")
#     roll = rolling_beta(rets, fac)
#     roll.to_parquet(paths.AI_BETAS_ROLLING)
#     print("wrote data/FACTOR/ai_betas_rolling.parquet")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())

"""Per-stock AI betas.

    r_it - rf = a + b_MKT*MKT + b_SMB*SMB + b_HML*HML + b_RMW*RMW + b_CMA*CMA
                + b_MOM*MOM + g*SECTOR_orth + d*AIF_orth + e

d is the AI BETA. Newey-West (HAC, 5 lags) standard errors, because daily equity
residuals are heteroskedastic and mildly autocorrelated; plain OLS errors would
overstate significance.

RUN THE SANITY GATE FIRST. NVDA strongly positive; PG and WM indistinguishable
from zero. The brief states this test explicitly. If it fails your factor is
broken -- report that rather than shipping betas you do not believe. "My null
names were not null, and here is what I think that means" is a better answer than
a clean-looking table you cannot defend.

VIFs are reported because NVDA will be collinear across MOM / XLK / AIF
simultaneously. That is a finding, not an embarrassment. Note that d means
"loading on the AI-specific residual", not "total AI sensitivity".

ROLLING BETAS matter: a single full-sample number cannot be reconciled against a
time-varying semantic index, and cannot feed the Task 7 lead-lag design.

    python -m src.factor.regressions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils import config
from src.utils import paths

STYLE = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]

HAC_LAGS = 5
ROLL = 126

OUT = paths.AI_BETAS
OUT_TXT = paths.AI_BETAS_TXT

ROLLING_OUT = paths.AI_BETAS_ROLLING
ROLLING_OUT_TXT = paths.AI_BETAS_ROLLING_TXT


def _design(
    ticker: str,
    rets: pd.DataFrame,
    fac: pd.DataFrame,
) -> pd.DataFrame:

    sector = config.sector_map()[ticker]

    df = pd.concat(
        [
            (rets[ticker] - fac["RF"]).rename("excess"),
            fac[STYLE],
            fac[f"{sector}_orth"].rename("SECTOR"),
            fac["AIF_orth"].rename("AIF"),
        ],
        axis=1,
    ).dropna()

    return df


def run(
    rets: pd.DataFrame,
    fac: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for t in config.tickers():

        if t not in rets.columns:
            print(
                f"  {t}: MISSING from price data"
            )
            continue

        df = _design(
            t,
            rets,
            fac,
        )

        X = sm.add_constant(
            df.drop(columns="excess")
        )

        fit = sm.OLS(
            df["excess"],
            X,
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": HAC_LAGS
            },
        )

        vifs = {
            X.columns[i]: variance_inflation_factor(
                X.values,
                i,
            )
            for i in range(1, X.shape[1])
        }

        rows.append(
            {
                "ticker": t,
                "ai_beta": fit.params["AIF"],
                "ai_se": fit.bse["AIF"],
                "ai_t": fit.tvalues["AIF"],
                "ai_p": fit.pvalues["AIF"],
                "ai_ci_lo": fit.conf_int().loc["AIF", 0],
                "ai_ci_hi": fit.conf_int().loc["AIF", 1],
                "mkt_beta": fit.params["Mkt-RF"],
                "sector_beta": fit.params["SECTOR"],
                "mom_beta": fit.params["MOM"],
                "alpha_ann": fit.params["const"] * 252,
                "r2": fit.rsquared,
                "n_obs": int(fit.nobs),
                "max_vif": max(vifs.values()),
                "vif_AIF": vifs["AIF"],
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("ticker")
    )


def sanity_gate(
    res: pd.DataFrame,
) -> dict:
    """The gate the brief specifies. Print whatever it says, pass or fail."""

    gate = config.load(
        "ai_basket"
    )["sanity_gate"]

    out = {}
    ok = True

    print(
        "\n--- SANITY GATE ---"
    )

    for t in gate["must_load_positive"]:

        if t not in res.index:
            continue

        r = res.loc[t]

        p = bool(
            r.ai_beta > 0
            and r.ai_p < 0.05
        )

        ok &= p

        out[f"{t}_positive"] = p

        print(
            f"  {t} loads positive : "
            f"beta={r.ai_beta:+.3f} "
            f"t={r.ai_t:+.2f} "
            f"p={r.ai_p:.4f}  "
            f"{'PASS' if p else 'FAIL'}"
        )

    for t in gate["must_be_indistinguishable_from_zero"]:

        if t not in res.index:
            continue

        r = res.loc[t]

        p = bool(
            r.ai_p > 0.05
        )

        ok &= p

        out[f"{t}_null"] = p

        print(
            f"  {t} is null        : "
            f"beta={r.ai_beta:+.3f} "
            f"t={r.ai_t:+.2f} "
            f"p={r.ai_p:.4f}  "
            f"{'PASS' if p else 'FAIL'}"
        )

    out["all_passed"] = ok

    print(
        f"  => "
        f"{'GATE PASSED' if ok else 'GATE FAILED - report this, do not hide it'}"
    )

    return out


def rolling_beta(
    rets: pd.DataFrame,
    fac: pd.DataFrame,
    window: int = ROLL,
) -> pd.DataFrame:
    """126-day rolling AI beta per ticker.

    Overlapping windows are mechanically autocorrelated -- fine for describing
    how exposure evolves, but do NOT run naive significance tests on the series
    (see the Task 7 design note).
    """

    out = {}

    for t in config.tickers():

        if t not in rets.columns:
            continue

        df = _design(
            t,
            rets,
            fac,
        )

        betas = {}

        for i in range(
            window,
            len(df) + 1,
        ):

            w = df.iloc[
                i - window:i
            ]

            fit = sm.OLS(
                w["excess"],
                sm.add_constant(
                    w.drop(columns="excess")
                ),
            ).fit()

            betas[
                df.index[i - 1]
            ] = fit.params["AIF"]

        out[t] = pd.Series(
            betas
        )

        print(
            f"  {t}: "
            f"{len(betas)} rolling estimates",
            flush=True,
        )

    return pd.DataFrame(out)


def main() -> int:

    # ---------------------------------------------------------
    # Load price returns
    # ---------------------------------------------------------

    rets = pd.read_parquet(
        paths.PRICES
    )

    rets.index = pd.to_datetime(
        rets.index
    )

    if getattr(
        rets.index,
        "tz",
        None,
    ) is not None:

        rets.index = (
            rets.index.tz_localize(None)
        )

    # ---------------------------------------------------------
    # Load AI factor
    # ---------------------------------------------------------

    fac = pd.read_parquet(
        paths.AI_FACTOR
    )

    fac.index = pd.to_datetime(
        fac.index
    )

    # ---------------------------------------------------------
    # Full-sample regressions
    # ---------------------------------------------------------

    res = run(
        rets,
        fac,
    ).sort_values(
        "ai_beta",
        ascending=False,
    )

    print(
        "\n--- AI BETAS ---"
    )

    print(
        res[
            [
                "ai_beta",
                "ai_se",
                "ai_t",
                "ai_p",
                "mkt_beta",
                "sector_beta",
                "r2",
                "vif_AIF",
            ]
        ]
        .round(3)
        .to_string()
    )

    # ---------------------------------------------------------
    # Sanity gate
    # ---------------------------------------------------------

    sanity_gate(
        res
    )

    # ---------------------------------------------------------
    # VIF diagnostics
    # ---------------------------------------------------------

    hi = res[
        res.max_vif > 10
    ]

    if len(hi):

        print(
            f"\nVIF > 10 for: "
            f"{list(hi.index)} -- "
            f"collinear with momentum/sector. "
            f"Expected for mega-cap AI names; "
            f"report it rather than dropping controls."
        )

    # ---------------------------------------------------------
    # Save full-sample beta results
    # ---------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Machine-readable version.
    res.to_parquet(
        OUT
    )

    # Human-readable version of the same dataset.
    res.to_csv(
        OUT_TXT,
        sep="\t",
        index=True,
        index_label="ticker",
        float_format="%.8f",
    )

    print(
        f"\nwrote {OUT}"
    )

    print(
        f"wrote {OUT_TXT}"
    )

    # ---------------------------------------------------------
    # Rolling betas
    # ---------------------------------------------------------

    print(
        f"\n--- rolling {ROLL}-day betas ---"
    )

    roll = rolling_beta(
        rets,
        fac,
    )

    # Machine-readable rolling beta dataset.
    roll.to_parquet(
        ROLLING_OUT
    )

    # Human-readable rolling beta dataset.
    roll.to_csv(
        ROLLING_OUT_TXT,
        sep="\t",
        index=True,
        index_label="Date",
        float_format="%.8f",
    )

    print(
        f"wrote {ROLLING_OUT}"
    )

    print(
        f"wrote {ROLLING_OUT_TXT}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
