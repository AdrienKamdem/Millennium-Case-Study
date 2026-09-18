# """Fama-French 5 + Momentum, daily, from the Kenneth French Data Library.

# THE trap: the library publishes values in PERCENT. Join them to decimal returns
# without dividing by 100 and every beta is 100x wrong -- and it will not look
# obviously wrong, which is why it bites. This module divides, and asserts.

#     python -m src.factor.ff_factors
# """

# from __future__ import annotations

# import sys
# from pathlib import Path

# import pandas as pd

# OUT = paths.FF_FACTORS
# OUT_TXT = paths.FF_FACTORS_TXT
# FF5 = "F-F_Research_Data_5_Factors_2x3_daily"
# MOM = "F-F_Momentum_Factor_daily"


# def load(start: str, end: str, force: bool = False) -> pd.DataFrame:
#     """Columns: Mkt-RF, SMB, HML, RMW, CMA, MOM, RF -- as DECIMALS."""
#     if OUT.exists() and not force:
#         print(f"using cached {OUT}")
#         return pd.read_parquet(OUT)

#     # Imported lazily so a cached run needs no network library -- `make all`
#     # should work offline from the committed cache.
#     from pandas_datareader.famafrench import FamaFrenchReader

#     ff = FamaFrenchReader(FF5, start=start, end=end).read()[0]
#     mom = FamaFrenchReader(MOM, start=start, end=end).read()[0]
#     mom.columns = [c.strip() for c in mom.columns]
#     mom = mom.rename(columns={"Mom": "MOM"})

#     df = ff.join(mom, how="inner")

#     # Percent -> decimal. Sanity: daily market moves are ~1%, so the max absolute
#     # value should be single-digit BEFORE dividing and tiny after.
#     mx = df.abs().max().max()
#     if mx < 0.5:
#         print(f"  values already look like decimals (max {mx:.4f}) -- NOT dividing")
#     else:
#         print(f"  max |value| = {mx:.2f} -> percent. Dividing by 100.")
#         df = df / 100.0
#     assert df.abs().max().max() < 0.5, "still too large after conversion -- inspect manually"

#     if isinstance(df.index, pd.PeriodIndex):
#         df.index = df.index.to_timestamp()
#     else:
#         df.index = pd.to_datetime(df.index)
#     OUT.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(OUT)
#     print(f"wrote {OUT}  {df.shape[0]} days x {list(df.columns)}")
#     return df


# def main() -> int:
#     from src.utils import config, paths
#     start, end = config.window()
#     df = load(start, end)
#     print(f"\nrange: {df.index[0].date()} -> {df.index[-1].date()}")
#     print(f"\nannualised mean (sanity: Mkt-RF should be roughly 5-15%):")
#     print((df.mean() * 252).round(4).to_string())
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())

"""Fama-French 5 + Momentum, daily, from the Kenneth French Data Library.

THE trap: the library publishes values in PERCENT. Join them to decimal returns
without dividing by 100 and every beta is 100x wrong -- and it will not look
obviously wrong, which is why it bites. This module divides, and asserts.

    python -m src.factor.ff_factors
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from src.utils import paths

OUT = paths.FF_FACTORS
OUT_TXT = paths.FF_FACTORS_TXT

FF5 = "F-F_Research_Data_5_Factors_2x3_daily"
MOM = "F-F_Momentum_Factor_daily"


def load(start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Columns: Mkt-RF, SMB, HML, RMW, CMA, MOM, RF -- as DECIMALS."""

    if OUT.exists() and not force:
        print(f"using cached {OUT}")

        df = pd.read_parquet(OUT)

        # Create human-readable TXT version if it does not already exist.
        if not OUT_TXT.exists():
            df.to_csv(
                OUT_TXT,
                sep="\t",
                index=True,
                index_label="Date",
                float_format="%.8f",
            )
            print(f"wrote {OUT_TXT}")

        return df

    # Imported lazily so a cached run needs no network library -- `make all`
    # should work offline from the committed cache.
    from pandas_datareader.famafrench import FamaFrenchReader

    ff = FamaFrenchReader(
        FF5,
        start=start,
        end=end,
    ).read()[0]

    mom = FamaFrenchReader(
        MOM,
        start=start,
        end=end,
    ).read()[0]

    mom.columns = [
        c.strip()
        for c in mom.columns
    ]

    mom = mom.rename(
        columns={"Mom": "MOM"}
    )

    df = ff.join(
        mom,
        how="inner",
    )

    # Percent -> decimal.
    # Sanity: daily market moves are ~1%, so the max absolute
    # value should be single-digit BEFORE dividing and tiny after.
    mx = df.abs().max().max()

    if mx < 0.5:
        print(
            f"  values already look like decimals "
            f"(max {mx:.4f}) -- NOT dividing"
        )
    else:
        print(
            f"  max |value| = {mx:.2f} "
            f"-> percent. Dividing by 100."
        )

        df = df / 100.0

    assert df.abs().max().max() < 0.5, (
        "still too large after conversion -- inspect manually"
    )

    # pandas_datareader may return a PeriodIndex depending on version.
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index)

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Machine-readable version used by the pipeline.
    df.to_parquet(OUT)

    # Human-readable version for inspection / appendix.
    df.to_csv(
        OUT_TXT,
        sep="\t",
        index=True,
        index_label="Date",
        float_format="%.8f",
    )

    print(
        f"wrote {OUT}  "
        f"{df.shape[0]} days x {list(df.columns)}"
    )

    print(
        f"wrote {OUT_TXT}"
    )

    return df


def main() -> int:

    from src.utils import config

    start, end = config.window()

    df = load(
        start,
        end,
    )

    print(
        f"\nrange: "
        f"{df.index[0].date()} -> "
        f"{df.index[-1].date()}"
    )

    print(
        "\nannualised mean "
        "(sanity: Mkt-RF should be roughly 5-15%):"
    )

    print(
        (df.mean() * 252)
        .round(4)
        .to_string()
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
