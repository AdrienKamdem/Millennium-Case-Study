# """Daily prices, with validation. No corpus dependency.

# The brief says endpoint reliability is your problem, so validate rather than
# assume. Four checks, and PRINT them -- "here are the validation checks I ran and
# what they returned" is an appendix paragraph you get for free.

# yfinance gotcha: recent versions default auto_adjust=True, which silently changes
# what Close means. Set it explicitly.

#     python -m src.factor.prices
# """

# from __future__ import annotations

# import sys
# from pathlib import Path

# import pandas as pd
# import yfinance as yf

# from src.utils import config, paths

# OUT = paths.PRICES

# SECTOR_ETFS = ["XLK", "XLI", "XLP"]
# EXTRA = ["AIQ", "SPY"]      # AIQ for the ETF teardown, SPY as a correlation yardstick


# def all_tickers() -> list[str]:
#     """Universe + factor basket + controls. Basket read from config, not duplicated."""
#     basket = config.load("ai_basket")["aif1"]["long"]
#     return sorted(set(config.tickers() + basket + SECTOR_ETFS + EXTRA))


# def download(tickers: list[str], start: str, end: str, force: bool = False) -> pd.DataFrame:
#     """Daily simple returns, one column per ticker. Cached to parquet."""
#     if OUT.exists() and not force:
#         print(f"using cached {OUT}")
#         return pd.read_parquet(OUT)

#     raw = yf.download(tickers, start=start, end=end,
#                       auto_adjust=True, progress=False, group_by="column")

#     # yfinance returns a MultiIndex for a list and a flat frame for one ticker.
#     close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
#     close = close.dropna(how="all").sort_index()

#     missing = [t for t in tickers if t not in close.columns or close[t].notna().sum() == 0]
#     if missing:
#         print(f"WARNING: no data for {missing} -- check tickers / listing dates")

#     rets = close.pct_change().iloc[1:]
#     OUT.parent.mkdir(parents=True, exist_ok=True)
#     rets.to_parquet(OUT)
#     print(f"wrote {OUT}  {rets.shape[0]} days x {rets.shape[1]} tickers")
#     return rets


# def validate(rets: pd.DataFrame) -> dict:
#     """Four checks. Returns pass/fail per check; always print the detail."""
#     res = {}
#     idx = rets.index

#     gaps = idx.to_series().diff().dt.days.fillna(0)
#     big = gaps[gaps > 5]
#     res["no_gap_over_5_days"] = big.empty
#     print(f"  gaps > 5 calendar days : {'PASS' if big.empty else f'FAIL {list(big.index[:3])}'}")

#     # ~252 trading days/year; flag if we are materially short.
#     years = (idx[-1] - idx[0]).days / 365.25
#     expected = years * 252
#     ratio = len(idx) / expected
#     res["row_count_plausible"] = 0.95 <= ratio <= 1.05
#     print(f"  row count vs calendar  : {len(idx)} rows, {expected:.0f} expected "
#           f"({ratio:.1%})  {'PASS' if res['row_count_plausible'] else 'CHECK'}")

#     extreme = {}
#     for t in rets.columns:
#         s = rets[t].dropna()
#         hits = s[s.abs() > 0.25]
#         if len(hits):
#             extreme[t] = [(str(d.date()), round(v, 3)) for d, v in hits.items()][:3]
#     res["extreme_moves"] = extreme
#     print(f"  |daily return| > 25%   : {len(extreme)} tickers affected")
#     for t, hits in list(extreme.items())[:5]:
#         print(f"      {t}: {hits}")
#     print("      ^ confirm each against a real event (earnings, guidance) before trusting")

#     # NVDA 10:1 split, June 2024. A -90% day means the adjustment is broken.
#     ok_split = True
#     if "NVDA" in rets.columns:
#         worst = rets["NVDA"].min()
#         ok_split = worst > -0.50
#         print(f"  NVDA split continuity  : worst day {worst:.1%}  "
#               f"{'PASS' if ok_split else 'FAIL - adjustment broken'}")
#     res["split_continuity"] = ok_split
#     return res


# def main() -> int:
#     start, end = config.window()
#     rets = download(all_tickers(), start, end)
#     print("\nVALIDATION")
#     validate(rets)
#     print(f"\nrange: {rets.index[0].date()} -> {rets.index[-1].date()}")
#     print(f"tickers: {list(rets.columns)}")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())


"""Daily prices, with validation. No corpus dependency.

The brief says endpoint reliability is your problem, so validate rather than
assume. Four checks, and PRINT them -- "here are the validation checks I ran and
what they returned" is an appendix paragraph you get for free.

yfinance gotcha: recent versions default auto_adjust=True, which silently changes
what Close means. Set it explicitly.

    python -m src.factor.prices
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils import config
from src.utils import paths

OUT = paths.PRICES
VALIDATION_OUT = paths.PRICES_VALIDATION_TXT

SECTOR_ETFS = ["XLK", "XLI", "XLP"]
EXTRA = ["AIQ", "SPY"]      # AIQ for the ETF teardown, SPY as a correlation yardstick


def all_tickers() -> list[str]:
    """Universe + factor basket + controls. Basket read from config, not duplicated."""
    basket = config.load("ai_basket")["aif1"]["long"]
    return sorted(set(config.tickers() + basket + SECTOR_ETFS + EXTRA))


def download(tickers: list[str], start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Daily simple returns, one column per ticker. Cached to parquet."""
    if OUT.exists() and not force:
        print(f"using cached {OUT}")
        return pd.read_parquet(OUT)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns a MultiIndex for a list and a flat frame for one ticker.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all").sort_index()

    missing = [
        t
        for t in tickers
        if t not in close.columns or close[t].notna().sum() == 0
    ]

    if missing:
        print(
            f"WARNING: no data for {missing} -- "
            f"check tickers / listing dates"
        )

    rets = close.pct_change().iloc[1:]

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rets.to_parquet(OUT)

    print(
        f"wrote {OUT}  "
        f"{rets.shape[0]} days x {rets.shape[1]} tickers"
    )

    return rets


def validate(rets: pd.DataFrame) -> dict:
    """Four checks. Returns pass/fail per check; always print the detail."""
    res = {}
    idx = rets.index

    gaps = idx.to_series().diff().dt.days.fillna(0)
    big = gaps[gaps > 5]

    res["no_gap_over_5_days"] = big.empty

    print(
        f"  gaps > 5 calendar days : "
        f"{'PASS' if big.empty else f'FAIL {list(big.index[:3])}'}"
    )

    # ~252 trading days/year; flag if we are materially short.
    years = (idx[-1] - idx[0]).days / 365.25
    expected = years * 252
    ratio = len(idx) / expected

    res["row_count_plausible"] = 0.95 <= ratio <= 1.05

    print(
        f"  row count vs calendar  : "
        f"{len(idx)} rows, {expected:.0f} expected "
        f"({ratio:.1%})  "
        f"{'PASS' if res['row_count_plausible'] else 'CHECK'}"
    )

    extreme = {}

    for t in rets.columns:
        s = rets[t].dropna()
        hits = s[s.abs() > 0.25]

        if len(hits):
            extreme[t] = [
                (str(d.date()), round(v, 3))
                for d, v in hits.items()
            ][:3]

    res["extreme_moves"] = extreme

    print(
        f"  |daily return| > 25%   : "
        f"{len(extreme)} tickers affected"
    )

    for t, hits in list(extreme.items())[:5]:
        print(f"      {t}: {hits}")

    print(
        "      ^ confirm each against a real event "
        "(earnings, guidance) before trusting"
    )

    # NVDA 10:1 split, June 2024. A -90% day means the adjustment is broken.
    ok_split = True

    if "NVDA" in rets.columns:
        worst = rets["NVDA"].min()
        ok_split = worst > -0.50

        print(
            f"  NVDA split continuity  : "
            f"worst day {worst:.1%}  "
            f"{'PASS' if ok_split else 'FAIL - adjustment broken'}"
        )

    res["split_continuity"] = ok_split

    return res


def main() -> int:
    start, end = config.window()

    rets = download(
        all_tickers(),
        start,
        end,
    )

    print("\nVALIDATION")

    validation = validate(rets)

    print(
        f"\nrange: "
        f"{rets.index[0].date()} -> {rets.index[-1].date()}"
    )

    print(
        f"tickers: {list(rets.columns)}"
    )

    # ---------------------------------------------------------
    # Save validation results to TXT
    # ---------------------------------------------------------

    VALIDATION_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gaps = rets.index.to_series().diff().dt.days.fillna(0)
    big = gaps[gaps > 5]

    years = (
        rets.index[-1] - rets.index[0]
    ).days / 365.25

    expected = years * 252
    ratio = len(rets.index) / expected

    with open(
        VALIDATION_OUT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "DAILY PRICE DATA VALIDATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Date range: "
            f"{rets.index[0].date()} -> "
            f"{rets.index[-1].date()}\n"
        )

        file.write(
            f"Number of observations: "
            f"{len(rets.index)}\n"
        )

        file.write(
            f"Number of tickers: "
            f"{len(rets.columns)}\n"
        )

        file.write(
            f"Tickers: "
            f"{list(rets.columns)}\n\n"
        )

        # -----------------------------------------------------
        # Check 1
        # -----------------------------------------------------

        file.write(
            "CHECK 1 - DATA GAPS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Result: "
            f"{'PASS' if validation['no_gap_over_5_days'] else 'FAIL'}\n"
        )

        file.write(
            f"Gaps greater than 5 calendar days: "
            f"{len(big)}\n"
        )

        if not big.empty:
            file.write(
                f"Dates: "
                f"{list(big.index)}\n"
            )

        file.write("\n")

        # -----------------------------------------------------
        # Check 2
        # -----------------------------------------------------

        file.write(
            "CHECK 2 - ROW COUNT PLAUSIBILITY\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Result: "
            f"{'PASS' if validation['row_count_plausible'] else 'CHECK'}\n"
        )

        file.write(
            f"Actual rows: "
            f"{len(rets.index)}\n"
        )

        file.write(
            f"Expected rows (~252/year): "
            f"{expected:.0f}\n"
        )

        file.write(
            f"Actual / expected ratio: "
            f"{ratio:.1%}\n\n"
        )

        # -----------------------------------------------------
        # Check 3
        # -----------------------------------------------------

        file.write(
            "CHECK 3 - EXTREME DAILY RETURNS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        extreme = validation["extreme_moves"]

        file.write(
            f"Tickers with |daily return| > 25%: "
            f"{len(extreme)}\n"
        )

        if extreme:

            for ticker, hits in extreme.items():

                file.write(
                    f"{ticker}: {hits}\n"
                )

            file.write(
                "\nExtreme observations should be confirmed "
                "against genuine market events such as earnings "
                "announcements or guidance revisions.\n"
            )

        else:

            file.write(
                "No extreme daily returns detected.\n"
            )

        file.write("\n")

        # -----------------------------------------------------
        # Check 4
        # -----------------------------------------------------

        file.write(
            "CHECK 4 - NVDA SPLIT CONTINUITY\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Result: "
            f"{'PASS' if validation['split_continuity'] else 'FAIL'}\n"
        )

        if "NVDA" in rets.columns:

            worst = rets["NVDA"].min()

            file.write(
                f"Worst NVDA daily return: "
                f"{worst:.2%}\n"
            )

            file.write(
                "A return below -50% would indicate that "
                "the June 2024 10:1 stock split may not have "
                "been adjusted correctly.\n"
            )

        file.write("\n")

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        file.write(
            "VALIDATION SUMMARY\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"No gap > 5 days: "
            f"{validation['no_gap_over_5_days']}\n"
        )

        file.write(
            f"Row count plausible: "
            f"{validation['row_count_plausible']}\n"
        )

        file.write(
            f"Split continuity: "
            f"{validation['split_continuity']}\n"
        )

        file.write(
            f"Extreme-return tickers requiring review: "
            f"{len(validation['extreme_moves'])}\n"
        )

    print(
        f"\nwrote validation results to "
        f"{VALIDATION_OUT}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

