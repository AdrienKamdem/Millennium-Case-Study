"""EDGAR filing index: which documents exist, and where.

Rebuilds data/SEC_Filing/sec_filings_2023_present.csv from the submissions API and
then goes one level deeper -- to the individual documents inside each filing, which
is where the text actually lives.

Two traps this module exists to avoid:

SHARD MERGING. filings.recent caps at 1000 entries, and Form 4 / 144 filings crowd
out everything else. For DELL it reaches back only to 2023-10-02 and for TEAM only
to 2024-09-24. Reading `recent` alone loses TEAM's entire Jan-2023-to-Sep-2024
history: 32 narrative filings instead of 57. src/corpus/edgar.py only warns about
this. Here the shards in filings.files are always merged.

THE 8-K PRIMARY DOCUMENT IS A COVER PAGE. Measured across a sample of ten, 8-K
primary documents run 3.3k-14k chars of registrant address, checkboxes and
signature block. The substance is in Exhibit 99.1 -- NVDA's Q1 FY27 earnings
release is 20,790 chars against 3,925 for the 8-K carrying it. 428 of the 562
filings in the window are 8-Ks, so indexing primary documents only would leave
three-quarters of the SEC corpus as boilerplate.

    python -m src.corpus.sec_index               # rebuild + diff against the frozen CSV
    python -m src.corpus.sec_index --documents   # also resolve documents per filing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import lxml.html
import pandas as pd

from src.utils import config, paths
from src.utils.cache import cached_get
from src.utils import paths

log = logging.getLogger(__name__)

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SHARD = "https://data.sec.gov/submissions/{name}"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

MIN_INTERVAL = 0.12                  # ~8 req/s, under SEC's 10 req/s cap
FORMS = ("10-K", "10-Q", "8-K")
SINCE = "2023-01-01"

# The frozen index stops here. Filings keep arriving, so an unpinned rebuild is not
# reproducible -- and reproducibility is the whole point of the diff.
AS_OF = "2026-09-10"
FROZEN_CSV = paths.FROZEN_FILING_INDEX

CSV_COLUMNS = ["ticker", "company", "cik", "form", "filingDate", "reportDate",
               "accessionNumber", "primaryDocument", "doc_url"]

# Exhibit types worth reading. EX-31/EX-32 are certifications, EX-101.* is the XBRL
# taxonomy, GRAPHIC/XML/ZIP are not text. EX-99 is press releases and presentations.
EXHIBIT_PREFIX = "EX-99"


def _hdr() -> dict:
    return config.sec_headers()


def doc_url(cik: int, accession: str, document: str) -> str:
    return f"{ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/{document}"


def fetch_submissions(cik: int) -> dict[str, list]:
    """Company submissions with the filings.files shards merged into recent."""
    data = cached_get("edgar", SUBMISSIONS.format(cik=cik), headers=_hdr(),
                      parse="json", min_interval=MIN_INTERVAL)

    merged = {k: list(v) for k, v in data["filings"]["recent"].items()}
    for shard in data["filings"].get("files", []):
        extra = cached_get("edgar", SHARD.format(name=shard["name"]), headers=_hdr(),
                           parse="json", min_interval=MIN_INTERVAL)
        # Shards are parallel arrays like `recent`, but a sparse column can be absent
        # entirely. Pad from the row count rather than indexing a key that may not exist.
        n = len(extra.get("form", []))
        for k in merged:
            merged[k].extend(extra.get(k) or [""] * n)

    merged["_entityName"] = data.get("name", "")
    return merged


def list_filings(as_of: str = AS_OF, since: str = SINCE,
                 forms: tuple[str, ...] = FORMS) -> pd.DataFrame:
    """One row per filing across the 9-name universe, matching the frozen CSV's schema."""
    rows = []
    for company in config.universe():
        cik = company.get("cik")
        if not cik:
            log.warning("%s: no cik in universe.yaml", company["ticker"])
            continue

        sub = fetch_submissions(int(cik))
        name = sub.pop("_entityName")
        n = len(sub["form"])
        for i in range(n):
            if sub["form"][i] not in forms:
                continue
            filed = sub["filingDate"][i]
            if not (since <= filed <= as_of):
                continue
            acc = sub["accessionNumber"][i]
            rows.append({
                "ticker": company["ticker"],
                "company": name,
                "cik": int(cik),
                "form": sub["form"][i],
                "filingDate": filed,
                "reportDate": sub["reportDate"][i],
                "accessionNumber": acc,
                "primaryDocument": sub["primaryDocument"][i],
                "doc_url": doc_url(int(cik), acc, sub["primaryDocument"][i]),
                "items": sub["items"][i],
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["ticker", "filingDate"], ascending=[True, False]).reset_index(drop=True)


def list_documents(cik: int, accession: str, primary_document: str) -> list[dict]:
    """Primary document plus every EX-99.* exhibit, from the filing's index page.

    The Document Format Files table carries Seq | Description | Document | Type | Size.
    Matching on Type is what makes this reliable: exhibit *filenames* are filer-specific
    (NVDA ships q2fy27pr.htm, not something containing "ex-99"), which is why
    src/corpus/edgar.py had to guess from an exclusion list instead.

    Type values are not uniform either -- EX-99.1, EX-99.2, and PG's "EX-99.1 CHARTER"
    all occur -- so this is a prefix test, never equality.
    """
    url = f"{ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/{accession}-index.htm"
    try:
        page = cached_get("edgar", url, headers=_hdr(), parse="text",
                          min_interval=MIN_INTERVAL)
    except Exception as e:
        log.warning("index page failed for %s: %s", accession, e)
        return []

    docs, seen = [], set()
    for tr in lxml.html.fromstring(page).xpath("//tr"):
        cells = [" ".join(td.text_content().split()) for td in tr.xpath("./td")]
        if len(cells) != 5 or not cells[0].isdigit():
            continue
        seq, _desc, document, doc_type, _size = cells
        document = document.split(" ")[0]                    # trailing "iXBRL" marker
        if not document.lower().endswith((".htm", ".html")) or document in seen:
            continue

        if document == primary_document:
            role = "primary"
        elif doc_type.upper().startswith(EXHIBIT_PREFIX):
            role = "exhibit"
        else:
            continue

        seen.add(document)
        docs.append({"seq": int(seq), "doc_type": doc_type, "doc_role": role,
                     "filename": document,
                     "url": doc_url(cik, accession, document)})

    if not any(d["doc_role"] == "primary" for d in docs):
        # Some older filings do not list the primary document in the table; it is
        # still fetchable, and dropping the filing entirely would be worse.
        docs.insert(0, {"seq": 1, "doc_type": "", "doc_role": "primary",
                        "filename": primary_document,
                        "url": doc_url(cik, accession, primary_document)})
        log.info("%s: primary document not in index table, added by hand", accession)

    return sorted(docs, key=lambda d: d["seq"])


def diff_against_frozen(df: pd.DataFrame, csv_path: str = FROZEN_CSV) -> pd.DataFrame:
    """Row-level diff of a rebuilt index against the committed CSV.

    Evidence for Task 8 that the corpus is reproducible from public sources, and a
    tripwire for the shard-merge regression: if `recent` is ever read on its own,
    TEAM and DELL lose rows here and the diff says so.
    """
    frozen = pd.read_csv(csv_path, dtype={"cik": int})
    left = df[CSV_COLUMNS].copy()

    a = set(frozen["accessionNumber"])
    b = set(left["accessionNumber"])
    out = pd.concat([
        frozen[frozen["accessionNumber"].isin(a - b)].assign(_diff="missing_from_rebuild"),
        left[left["accessionNumber"].isin(b - a)].assign(_diff="new_in_rebuild"),
    ], ignore_index=True)

    print(f"frozen {len(frozen):,} rows | rebuilt {len(left):,} rows | differing {len(out):,}")
    if len(out):
        print(out[["_diff", "ticker", "form", "filingDate", "accessionNumber"]].to_string(index=False))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=AS_OF, help="freeze the index at this filing date")
    ap.add_argument("--documents", action="store_true",
                    help="also resolve documents (primary + EX-99) per filing")
    ap.add_argument("--no-diff", action="store_true")
    args = ap.parse_args()

    df = list_filings(as_of=args.as_of)
    if df.empty:
        print("NO FILINGS -- check SEC_USER_AGENT and config/universe.yaml")
        return 1

    out = paths.FILING_INDEX
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df):,} filings)")
    print(pd.crosstab(df["ticker"], df["form"]).to_string())

    if not args.no_diff and Path(FROZEN_CSV).exists():
        print(f"\ndiff vs {FROZEN_CSV}:")
        diff_against_frozen(df)

    if args.documents:
        docs = []
        for _, f in df.iterrows():
            for d in list_documents(f["cik"], f["accessionNumber"], f["primaryDocument"]):
                docs.append({**f.to_dict(), **d})
        dd = pd.DataFrame(docs)
        p = paths.DOCUMENT_INDEX
        dd.to_parquet(p, index=False)
        print(f"\nwrote {p}  ({len(dd):,} documents)")
        print(dd["doc_role"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
