"""Evening-1 source probe. Run this BEFORE writing any pipeline code.

One question: which sources actually return January-2023 data for free? Most free
news APIs serve only the last 30 days to 12 months, and the window here starts
Jan 2023. The answer determines scope, so it is worth knowing on Monday rather
than discovering it on Thursday.

Everything here is throwaway and deliberately unsophisticated. Record the answers
in config/sources.yaml — including the failures, which are Task 8 evidence — and
move on.

    python -m scripts.probe_sources
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import requests

TIMEOUT = 30
PROBE_START = "20230115000000"
PROBE_END = "20230122000000"


def _ok(label: str, detail: str = "") -> None:
    print(f"  PASS  {label}  {detail}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  FAIL  {label}  {detail}")


# ---------------------------------------------------------------------------

def probe_gdelt() -> None:
    """The critical one. A 200 is not a pass — check the returned DATES."""
    print("\n[1] GDELT DOC API — historical depth")
    for query in ['"Nvidia"', '"Atlassian"', '"Waste Management"']:
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": f"{query} sourcelang:english",
                    "mode": "ArtList",
                    "maxrecords": 250,
                    "format": "json",
                    "startdatetime": PROBE_START,
                    "enddatetime": PROBE_END,
                },
                timeout=TIMEOUT,
            )
            # GDELT returns HTML, not JSON, on some error paths.
            try:
                payload = r.json()
            except ValueError:
                _fail(query, f"http {r.status_code}, non-JSON body: {r.text[:120]!r}")
                continue

            arts = payload.get("articles", [])
            if not arts:
                _fail(query, f"http {r.status_code}, 0 articles")
                continue

            # The actual test: did the date filter work, or did it silently
            # return recent articles?
            #
            # Do NOT assume the seendate format — strip to digits first, so an
            # ISO-style date does not read as a failure. A false FAIL here would
            # cost you the news backbone for no reason.
            raw = [str(a.get("seendate", "")) for a in arts]
            digits = ["".join(ch for ch in d if ch.isdigit()) for d in raw]
            in_window = [d for d in digits if d.startswith("202301")]
            yyyymm = sorted({d[:6] for d in digits if len(d) >= 6})

            verdict = f"{len(arts)} articles, {len(in_window)} in Jan 2023"
            (_ok if in_window else _fail)(query, verdict)
            print(f"        raw seendate sample : {raw[:2]}")
            print(f"        months returned     : {yyyymm[:8]}")
            if not in_window and yyyymm:
                print("        ^^ dates are OUTSIDE the requested window: the date filter")
                print("           was ignored. Treat as shallow and use the fallback.")
            print(f"        title sample        : {arts[0].get('title', '')[:85]}")
        except Exception as e:
            _fail(query, repr(e))

    print("      -> If dates are NOT in Jan 2023, the DOC API is shallow. Fall back to")
    print("         GDELT BigQuery (dry-run first), or truncate the news window and say so.")


def probe_sec(email_ua: str) -> dict[str, str]:
    """Resolve CIKs and confirm filing history reaches 2023. This will work."""
    print("\n[2] SEC EDGAR — CIK resolution + filing history")
    tickers = {"ADBE", "CAT", "DELL", "INTC", "MU", "NVDA", "PG", "TEAM", "WM"}
    ciks: dict[str, str] = {}
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": email_ua},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        for row in r.json().values():
            t = row["ticker"].upper()
            if t in tickers:
                ciks[t] = str(row["cik_str"]).zfill(10)
        missing = tickers - ciks.keys()
        (_ok if not missing else _fail)("company_tickers.json", f"resolved {len(ciks)}/9, missing {missing or 'none'}")
        print(f"        {json.dumps(ciks, indent=8)}")
    except Exception as e:
        _fail("company_tickers.json", repr(e))
        return {}

    if "NVDA" in ciks:
        try:
            r = requests.get(
                f"https://data.sec.gov/submissions/CIK{ciks['NVDA']}.json",
                headers={"User-Agent": email_ua},
                timeout=TIMEOUT,
            )
            recent = r.json()["filings"]["recent"]
            oldest = min(recent["filingDate"])
            _ok("submissions API (NVDA)", f"{len(recent['form'])} recent filings, oldest {oldest}")
            print("        note: 'recent' is capped ~1000 filings; older ones are in the")
            print("        additional files listed under filings.files — page through them.")
        except Exception as e:
            _fail("submissions API", repr(e))
    return ciks


def probe_hn() -> None:
    print("\n[3] Hacker News (Algolia) — full archive, free, no key")
    start = int(dt.datetime(2023, 1, 1).timestamp())
    end = int(dt.datetime(2023, 2, 1).timestamp())
    try:
        r = requests.get(
            "http://hn.algolia.com/api/v1/search_by_date",
            params={"query": "Nvidia", "tags": "story",
                    "numericFilters": f"created_at_i>{start},created_at_i<{end}"},
            timeout=TIMEOUT,
        )
        hits = r.json().get("hits", [])
        (_ok if hits else _fail)("Jan 2023 stories", f"{len(hits)} hits")
        if hits:
            print(f"        sample: {hits[0].get('title', '')[:90]}")
    except Exception as e:
        _fail("hn algolia", repr(e))


def probe_prices() -> None:
    print("\n[4] Prices + Fama-French")
    try:
        import yfinance as yf
        df = yf.download("NVDA", start="2023-01-01", end="2023-02-01",
                         auto_adjust=True, progress=False)
        (_ok if len(df) > 15 else _fail)("yfinance NVDA Jan 2023", f"{len(df)} rows")
    except Exception as e:
        _fail("yfinance", repr(e))

    # FF5 and Momentum are SEPARATE datasets. The regression needs both, so test
    # both now rather than discovering a bad dataset name mid-week.
    try:
        from pandas_datareader.famafrench import FamaFrenchReader
    except ImportError as e:
        _fail("Ken French", f"pandas_datareader not installed ({e})")
        return
    for label, dataset in [
        ("Ken French FF5 daily", "F-F_Research_Data_5_Factors_2x3_daily"),
        ("Ken French MOM daily", "F-F_Momentum_Factor_daily"),
    ]:
        try:
            df = FamaFrenchReader(dataset, start="2023-01-01", end="2023-02-01").read()[0]
            _ok(label, f"{len(df)} rows, cols {list(df.columns)}")
            print(f"        max |value| = {df.abs().max().max():.3f}  "
                  f"<- if ~1-3 these are PERCENT; divide by 100 or every beta is 100x wrong")
        except Exception as e:
            _fail(label, repr(e))
            print("        -> browse https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
            print("           for the exact dataset name, or use get_available_datasets()")


def main() -> int:
    import os
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        print("SEC_USER_AGENT unset — copy env.example to .env and fill it in first.")
        print("SEC throttles or blocks requests without a real contact address.")
        return 1

    print("=" * 72)
    print("SOURCE PROBE — can we reach January 2023?")
    print("=" * 72)

    probe_gdelt()
    ciks = probe_sec(ua)
    probe_hn()
    probe_prices()

    print("\n" + "=" * 72)
    print("Next: write every result into config/sources.yaml, including failures.")
    if ciks:
        print("Paste the resolved CIKs into config/universe.yaml.")
    print("Not probed automatically — do these by hand:")
    print("  - GDELT BigQuery dry-run (needs a GCP project)")
    print("  - Transcript API free-tier limits (FMP / API Ninjas) -> decide buy vs SEC-substitute")
    print("  - Alias false-positive rate for INTC / MU / WM (eyeball 20 headlines each)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
