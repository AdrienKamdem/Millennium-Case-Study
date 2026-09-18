"""Verify the EDGAR RETRIEVAL chain, not just discovery.

The main probe confirmed CIK resolution and the submissions index. That proves you
can LIST filings. This proves you can FETCH them, which is a different set of
endpoints and the one Tuesday actually depends on:

    submissions.json  ->  accession number
    accession index   ->  which file inside the filing is the document / EX-99.1
    Archives fetch    ->  the text itself

Also reports form types per company across the window, which catches the two
name-specific traps: TEAM may have filed 20-F early in the window (it redomiciled
from Australia in 2022), and DELL's fiscal year is offset from the calendar.

    python -m scripts.probe_edgar
"""

from __future__ import annotations

import collections
import os
import sys
import time

import requests

BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
WINDOW_START = "2023-01-01"
PAUSE = 0.2  # SEC caps at 10 req/s; stay well under

CIK = {"ADBE": "0000796343", "CAT": "0000018230", "DELL": "0001571996",
       "INTC": "0000050863", "MU": "0000723125", "NVDA": "0001045810",
       "PG": "0000080424", "TEAM": "0001650372", "WM": "0000823768"}


def get(url: str, hdr: dict):
    time.sleep(PAUSE)
    r = requests.get(url, headers=hdr, timeout=60)
    r.raise_for_status()
    return r


def form_census(hdr: dict) -> dict[str, list]:
    """What forms does each company actually file in the window? Catches 20-F, 10-K405, etc."""
    print("\n[1] Form types filed since", WINDOW_START)
    out = {}
    for tic, cik in sorted(CIK.items()):
        try:
            rec = get(f"{BASE}/submissions/CIK{cik}.json", hdr).json()["filings"]["recent"]
            rows = [(f, d, a, p) for f, d, a, p in zip(
                rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"])
                if d >= WINDOW_START]
            counts = collections.Counter(f for f, _, _, _ in rows)
            interesting = {k: v for k, v in counts.items() if k in
                           ("10-K", "10-Q", "8-K", "20-F", "40-F", "6-K")}
            print(f"  {tic:5} {dict(sorted(interesting.items()))}")
            if "20-F" in counts or "6-K" in counts:
                print(f"        ^^ {tic} files foreign-issuer forms — different structure, handle separately")
            if not interesting.get("10-K"):
                print(f"        ^^ NO 10-K in window for {tic} — check fiscal year / form type")
            out[tic] = rows
        except Exception as e:
            print(f"  {tic:5} FAIL {e!r}")
    return out


def fetch_one_10k(rows: list, cik: str, hdr: dict) -> None:
    """Prove the Archives fetch works end to end."""
    print("\n[2] Retrieve one 10-K end to end (NVDA)")
    tenk = [r for r in rows if r[0] == "10-K"]
    if not tenk:
        print("  FAIL  no 10-K found in window")
        return
    form, date, acc, doc = tenk[0]
    acc_nodash = acc.replace("-", "")
    url = f"{ARCHIVES}/{int(cik)}/{acc_nodash}/{doc}"
    try:
        text = get(url, hdr).text
        ai = text.lower().count("artificial intelligence")
        print(f"  PASS  {form} {date}  {len(text):,} chars")
        print(f"        url: {url}")
        print(f"        'artificial intelligence' appears {ai}x  <- sanity: should be >0 for NVDA")
    except Exception as e:
        print(f"  FAIL  {url}\n        {e!r}")


def find_ex99(rows: list, cik: str, hdr: dict) -> None:
    """8-K EX-99.1 is a separate file inside the filing directory. Prove you can locate it."""
    print("\n[3] Locate 8-K Exhibit 99.1 (the earnings release)")
    eights = [r for r in rows if r[0] == "8-K"][:4]
    if not eights:
        print("  FAIL  no 8-K in window")
        return
    for form, date, acc, _ in eights:
        acc_nodash = acc.replace("-", "")
        idx_url = f"{ARCHIVES}/{int(cik)}/{acc_nodash}/index.json"
        try:
            items = get(idx_url, hdr).json()["directory"]["item"]
            names = [i["name"] for i in items]
            ex99 = [n for n in names if n.lower().startswith("ex-99") or "ex99" in n.lower()]
            if ex99:
                url = f"{ARCHIVES}/{int(cik)}/{acc_nodash}/{ex99[0]}"
                body = get(url, hdr).text
                print(f"  PASS  8-K {date} -> {ex99[0]}  ({len(body):,} chars)")
                print(f"        files in filing: {names[:6]}")
                return
            print(f"  ....  8-K {date}: no EX-99 ({names[:5]})")
        except Exception as e:
            print(f"  FAIL  {idx_url}\n        {e!r}")
    print("  NOTE  no EX-99.1 in the first few 8-Ks — most 8-Ks are not earnings releases.")
    print("        Filter on the 8-K 'items' field (2.02 = Results of Operations) instead of scanning.")


def main() -> int:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        print("SEC_USER_AGENT unset. `set -a && source .env && set +a` first.")
        return 1
    hdr = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    print("=" * 72)
    print("EDGAR RETRIEVAL PROBE — can we fetch, not just list?")
    print("=" * 72)

    rows = form_census(hdr)
    if "NVDA" in rows:
        fetch_one_10k(rows["NVDA"], CIK["NVDA"], hdr)
        find_ex99(rows["NVDA"], CIK["NVDA"], hdr)

    print("\n" + "=" * 72)
    print("If [2] and [3] passed, Tuesday's EDGAR work is de-risked.")
    print("Check [1] for TEAM (20-F?) and DELL (offset fiscal year) before bucketing by quarter.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
