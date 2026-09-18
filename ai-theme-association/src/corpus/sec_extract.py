"""EDGAR HTML -> clean text. Pure functions over bytes: no network, no file I/O.

Everything that touches sec.gov lives in sec_index.py / sec_documents.py. This module
is deliberately offline so that cleaning can be iterated against cached fixtures
without re-fetching a 2 MB filing every time a regex changes.

What the input actually looks like, measured on NVDA's FY2026 10-K (1.97 MB):

  <ix:header> is 125 KB and contributes ~20k chars of "iso4217:USD xbrli:shares"
  noise if it survives to the text stage. 398 display:none blocks. 64 tables, 12.3%
  of text volume, 26% of cells numeric. "Table of Contents" repeats as a running
  header on every page. Every "Item 1A." appears twice -- once in the TOC, once for
  real. &#9746; and friends survive naive tag-stripping.

Two non-obvious decisions:

TABLES ARE KEPT, as pipe markdown, with a char-offset span and a numeric density
recorded per table. Dropping them outright (what src/corpus/edgar.py does) is
cheaper but silently deletes narrative, because EDGAR filers routinely lay prose
out in tables. Keeping the spans lets the chunking step apply a drop-or-keep policy
later without re-parsing anything.

THE WALK IS BLOCK-AWARE. lxml's .text_content() and bs4's .get_text() with no
separator glue the last word of one block to the first word of the next. That
corruption is invisible in a character count and poisons every downstream keyword
match, so blocks emit their own newlines here.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

import lxml.html

# Inline XBRL has to go before parsing: lxml's HTML parser mangles the ix: namespace,
# so a post-parse xpath for it is unreliable.
IX_BLOCK = re.compile(r"<ix:(header|hidden)\b.*?</ix:\1>", re.IGNORECASE | re.DOTALL)
HIDDEN_STYLE = re.compile(r"(display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)

DROP_TAGS = {"script", "style", "head", "noscript", "meta", "link", "title"}
BLOCK_TAGS = {
    "p", "div", "tr", "li", "table", "blockquote", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "ul", "ol", "dl", "dd", "dt",
}

# A cell counts as numeric if it holds nothing but digits, currency and punctuation.
NUMERIC_CELL = re.compile(r"^[\s$()\-+.,%0-9]+$")
NUMERIC_TABLE_THRESHOLD = 0.40

# Sentinels stand in for rendered tables while the text is normalised line by line,
# so that page-artifact stripping cannot chew up a table row that happens to look
# like a page number. Substituted back at the very end, which is also how the
# offsets end up correct.
_SENT_OPEN, _SENT_CLOSE = "\x00", "\x01"
_SENTINEL = re.compile(r"\x00(\d+)\x01")

TOC_LINE = re.compile(r"^table\s+of\s+contents\.?$", re.IGNORECASE)
PAGE_LINE = re.compile(r"^(page\s+)?\d{1,4}(\s*\|\s*page)?$", re.IGNORECASE)
RUNNING_HEADER_MIN_REPEATS = 5
RUNNING_HEADER_MAX_LEN = 80

# The leading "|" matters. Plenty of filers lay their Item headings out in a table, and
# since tables render as markdown the line arrives as "| Item 1. | Business |". Anchoring
# on "^[ \t]*Item" alone missed every one: INTC scored 0 of 15 periodic filings and PG
# 4 of 15, while filers using plain paragraphs scored 15 of 15.
ITEM_RE = re.compile(r"(?m)^[ \t|]*Item[ \t]+(\d{1,2}[A-C]?)[ \t]*[.:|–—-]", re.IGNORECASE)
PART_RE = re.compile(r"(?m)^[ \t|]*PART[ \t]+(I{1,3}V?|IV)\b", re.IGNORECASE)
MIN_SECTIONS = {"10-K": 5, "10-Q": 2}
# Minimum share of the document that the Item spans must cover once the largest is
# excluded. Below this the headings are a cross-reference index, not content.
CROSS_REF_COVERAGE = 0.05

EXTRACTOR_VERSION = "sec_extract/1.3"

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...", " ": " ",
})


@dataclass
class TableSpan:
    start: int
    end: int
    n_rows: int
    n_cols: int
    numeric_frac: float


@dataclass
class SectionSpan:
    item: str
    start: int
    end: int


@dataclass
class Extracted:
    text: str
    tables: list[TableSpan] = field(default_factory=list)
    sections: list[SectionSpan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_tables_numeric(self) -> int:
        return sum(t.numeric_frac > NUMERIC_TABLE_THRESHOLD for t in self.tables)


def extract(raw: bytes, *, doc_type: str = "") -> Extracted:
    """Clean text plus table and section offsets for one EDGAR document.

    `doc_type` is the filing-index Type ("10-K", "EX-99.1", ...). Only 10-K and 10-Q
    get Item-boundary detection; everything else returns an empty `sections`.
    """
    warnings: list[str] = []

    text_html = _decode(raw, warnings)
    text_html = _strip_sgml_wrapper(text_html)
    text_html = IX_BLOCK.sub(" ", text_html)

    if not text_html.strip():
        return Extracted(text="", warnings=warnings + ["empty_after_prestrip"])

    try:
        root = lxml.html.fromstring(text_html)
    except Exception as first:
        # Belt and braces. str input is rejected outright for a whole class of inputs
        # (an encoding declaration anywhere lxml notices it), and returning an empty
        # document for those is a silent, corpus-wide failure. Bytes input has no such
        # restriction, so retry there before giving up.
        try:
            root = lxml.html.fromstring(text_html.encode("utf-8", "replace"))
            warnings.append(f"parse_retried_as_bytes:{type(first).__name__}")
        except Exception as e:                               # malformed beyond recovery
            return Extracted(text="", warnings=warnings + [f"parse_failed:{type(e).__name__}"])

    _drop_hidden(root)

    parts: list[str] = []
    tables: list[tuple[str, int, int, float]] = []           # markdown, rows, cols, numeric
    _walk(root, parts, tables)

    body = "".join(parts)
    body = html.unescape(body)
    body = _normalise(body)
    body, dropped = _strip_page_artifacts(body)
    if dropped:
        # "info:" rather than a bare warning -- this fires on nearly every filing, and
        # a flag that is always on tells you nothing about the ones that went wrong.
        warnings.append(f"info:page_artifacts_dropped:{len(dropped)}")

    text, spans = _splice_tables(body, tables)

    sections = _find_item_sections(text, doc_type, warnings)

    return Extracted(text=text, tables=spans, sections=sections, warnings=warnings)


# ---------------------------------------------------------------- decoding

_META_CHARSET = re.compile(rb"""charset=["']?\s*([A-Za-z0-9_\-]+)""", re.IGNORECASE)


def _decode(raw: bytes, warnings: list[str]) -> str:
    """Decode with the declared charset, falling back utf-8 -> cp1252.

    errors="replace" rather than "ignore": a U+FFFD is something the QC harness can
    count, whereas silently dropped bytes are indistinguishable from clean text.
    """
    m = _META_CHARSET.search(raw[:4096])
    candidates = ([m.group(1).decode("ascii", "ignore")] if m else []) + ["utf-8", "cp1252"]
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    warnings.append("decode_lossy")
    return raw.decode("utf-8", errors="replace")


# EDGAR wraps each document in an SGML envelope. bs4 and lxml both render those tags
# as text, which is where the stray "EX-99.1 2 foo.htm EX-99.1 EARNINGS RELEASE
# Document" prefix on exhibits comes from.
_SGML_HEAD = re.compile(r"^.*?<TEXT>", re.IGNORECASE | re.DOTALL)
_SGML_TAIL = re.compile(r"</TEXT>.*$", re.IGNORECASE | re.DOTALL)
_DOC_START = re.compile(r"<(!DOCTYPE|html|\?xml)", re.IGNORECASE)

# Modern iXBRL primary documents are XHTML and open with
# <?xml version="1.0" encoding="UTF-8"?>. lxml.html.fromstring() REFUSES a str that
# carries an encoding declaration -- "Unicode strings with encoding declaration are
# not supported" -- so this line, left in, kills every 10-K, 10-Q and 8-K primary
# document while EX-99 exhibits (plain HTML, no declaration) sail through.
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>\s*", re.IGNORECASE)


def _strip_sgml_wrapper(s: str) -> str:
    head = s[:8192]
    if re.search(r"<TEXT>", head, re.IGNORECASE):
        s = _SGML_HEAD.sub("", s, count=1)
        s = _SGML_TAIL.sub("", s, count=1)
    elif re.search(r"<(TYPE|SEQUENCE|FILENAME|DESCRIPTION)>", head, re.IGNORECASE):
        m = _DOC_START.search(s)
        if m:
            s = s[m.start():]
    # After the wrapper comes off, the XML declaration is at the front. Drop it rather
    # than re-encoding: the declaration may name a charset other than the one we
    # decoded with, and handing lxml mismatched bytes would be worse than no header.
    return _XML_DECL.sub("", s, count=1)


# ---------------------------------------------------------------- tree -> text

def _drop_hidden(root) -> None:
    for el in root.xpath("//*[@style]"):
        if HIDDEN_STYLE.search(el.get("style") or ""):
            el.drop_tree()
    for el in root.xpath("//*[@hidden]"):
        el.drop_tree()


def _walk(el, out: list[str], tables: list) -> None:
    tag = el.tag
    if not isinstance(tag, str):                             # comment, PI
        return
    tag = tag.lower()
    if tag in DROP_TAGS:
        return

    if tag == "table":
        # Emitting a placeholder and not descending is what keeps nested layout
        # tables from being rendered twice: an inner table flattens into its
        # parent's cell text instead of becoming a second markdown block.
        md, rows, cols, numeric = _table_to_markdown(el)
        if md:
            out.append(f"{_SENT_OPEN}{len(tables)}{_SENT_CLOSE}")
            tables.append((md, rows, cols, numeric))
        out.append("\n\n")
        return

    if tag == "br":
        out.append("\n")

    if el.text:
        out.append(el.text)
    for child in el:
        _walk(child, out, tables)
        if child.tail:
            out.append(child.tail)

    if tag in BLOCK_TAGS:
        out.append("\n\n")


def _own_rows(table) -> list:
    """Rows belonging to this table, not to a table nested inside it."""
    return [tr for tr in table.iter("tr")
            if (anc := tr.xpath("ancestor::table[1]")) and anc[0] is table]


def _table_to_markdown(table) -> tuple[str, int, int, float]:
    """Render one table as a pipe table; also return shape and numeric-cell share."""
    grid: list[list[str]] = []
    numeric = total = 0
    for tr in _own_rows(table):
        cells = [c for c in tr if isinstance(c.tag, str) and c.tag.lower() in ("td", "th")]
        if not cells:
            continue
        row = []
        for c in cells:
            v = re.sub(r"\s+", " ", c.text_content()).strip().replace("|", r"\|")
            row.append(v)
            if v:
                total += 1
                numeric += bool(NUMERIC_CELL.match(v))
        grid.append(row)

    grid = [r for r in grid if any(r)]
    if not grid:
        return "", 0, 0, 0.0

    n_cols = max(len(r) for r in grid)
    grid = [r + [""] * (n_cols - len(r)) for r in grid]

    lines = ["| " + " | ".join(grid[0]) + " |",
             "| " + " | ".join(["---"] * n_cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines), len(grid), n_cols, (numeric / total if total else 0.0)


# ---------------------------------------------------------------- normalisation

def _normalise(s: str) -> str:
    """NFKC, fold punctuation, strip zero-widths, collapse whitespace.

    Punctuation folding is not cosmetic: it makes content_sha256 stable across filers
    who differ only in curly-vs-straight quotes, which is what makes cross-filing
    duplicate detection work.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_ZERO_WIDTH).translate(_PUNCT_FOLD)
    s = re.sub(r"[^\S\n]+", " ", s)                          # horizontal runs -> one space
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_page_artifacts(s: str) -> tuple[str, list[str]]:
    """Remove running headers, "Table of Contents" and bare page numbers.

    Returns the dropped lines as well as the text -- the QC report shows them, because
    a cleaning rule that deletes content silently is worse than no cleaning rule.
    """
    lines = s.split("\n")

    counts: dict[str, int] = {}
    for ln in lines:
        t = ln.strip()
        if t and len(t) <= RUNNING_HEADER_MAX_LEN and not _SENTINEL.search(t):
            counts[t] = counts.get(t, 0) + 1
    repeated = {t for t, n in counts.items() if n >= RUNNING_HEADER_MIN_REPEATS}

    kept, dropped = [], []
    for ln in lines:
        t = ln.strip()
        if t and (TOC_LINE.match(t) or PAGE_LINE.match(t) or t in repeated):
            dropped.append(t)
            continue
        kept.append(ln)

    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return out, dropped


def _splice_tables(body: str, tables: list) -> tuple[str, list[TableSpan]]:
    """Replace sentinels with rendered markdown, recording each table's char span."""
    spans: list[TableSpan] = []
    out: list[str] = []
    pos = 0
    cursor = 0
    for m in _SENTINEL.finditer(body):
        out.append(body[pos:m.start()])
        cursor += m.start() - pos
        md, rows, cols, numeric = tables[int(m.group(1))]
        out.append(md)
        spans.append(TableSpan(cursor, cursor + len(md), rows, cols, round(numeric, 3)))
        cursor += len(md)
        pos = m.end()
    out.append(body[pos:])
    return "".join(out), spans


# ---------------------------------------------------------------- sections

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def _item_rank(item: str) -> tuple[int, str]:
    m = re.match(r"(\d{1,2})([A-C]?)", item.upper())
    return (int(m.group(1)), m.group(2)) if m else (99, "")


def _find_item_sections(text: str, doc_type: str, warnings: list[str]) -> list[SectionSpan]:
    """Item boundaries for 10-K/10-Q, or [] when detection is not confident.

    Three things make this harder than the regex suggests, and all three are why
    src/corpus/edgar.py skipped it:

      1. Every item appears in the table of contents before it appears for real.
      2. Items are cross-referenced from inside other items ("see Item 1A above").
      3. A 10-Q reuses item numbers across Part I and Part II -- Item 1 is both
         Financial Statements and Legal Proceedings.

    (1) and (2) are handled by keeping, per label, the occurrence that opens the
    longest span, then forcing the survivors into increasing order. (3) by prefixing
    the part. Below the per-form minimum, return [] rather than wrong offsets: the
    chunker is expected to fall back to paragraph splitting.
    """
    if doc_type not in MIN_SECTIONS:
        return []

    marks = [(m.start(), "part", m.group(1).upper()) for m in PART_RE.finditer(text)]
    marks += [(m.start(), "item", m.group(1).upper()) for m in ITEM_RE.finditer(text)]
    marks.sort()
    if not marks:
        warnings.append("sections_none_found")
        return []

    part = ""
    cands: list[tuple[str, int, int]] = []                   # label, start, gap
    for i, (pos, kind, val) in enumerate(marks):
        if kind == "part":
            part = val
            continue
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        cands.append((f"{part}.{val}" if part else val, pos, nxt - pos))

    # Per label keep the occurrence opening the longest span: the TOC entry and any
    # cross-reference are both followed almost immediately by the next marker.
    best: dict[str, tuple[int, int]] = {}
    for label, pos, gap in cands:
        if label not in best or gap > best[label][1]:
            best[label] = (pos, gap)

    ordered = sorted((pos, label) for label, (pos, _) in best.items())
    ordered = _increasing_by_item(ordered)

    if len(ordered) < MIN_SECTIONS[doc_type]:
        warnings.append(f"sections_low_confidence:{len(ordered)}")
        return []

    spans = []
    for i, (pos, label) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        spans.append(SectionSpan(label, pos, end))

    # Cross-reference filers. SEC permits answering the Items by pointing elsewhere in
    # the document -- Intel's 10-K opens with an index reading "Item 1A. Risk Factors
    # Pages 37-51" and puts the actual content under its own headings further down. The
    # spans are then real Item headings sitting on 40-char index rows, which passes every
    # count- and label-based test and is still useless: chunking by them yields twenty
    # fragments plus one span swallowing the rest of the filing.
    #
    # Spans always tile to the end of the document, so total coverage is no help. Drop
    # the largest and ask how much the REST cover: Intel scores 0.3%, NVDA 69%, PG 85%.
    sizes = [s.end - s.start for s in spans]
    if text and (sum(sizes) - max(sizes)) / len(text) < CROSS_REF_COVERAGE:
        warnings.append("sections_cross_reference_index")
        return []

    return spans


def _increasing_by_item(ordered: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Longest subsequence whose item ranks increase with position.

    A stray match that lands out of order -- Item 7 appearing between Item 2 and
    Item 3 -- is a cross-reference, not a heading. Dropping it is cheaper and safer
    than trying to decide which of the two is real.
    """
    if not ordered:
        return []

    def key(lab: str) -> tuple[int, int, str]:
        pt, _, it = lab.rpartition(".")
        num, suf = _item_rank(it)
        return (_ROMAN.get(pt, 0), num, suf)

    keys = [key(lab) for _, lab in ordered]
    n = len(ordered)
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if keys[j] < keys[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(n), key=lambda k: best[k])
    out = []
    while i >= 0:
        out.append(ordered[i])
        i = prev[i]
    return out[::-1]
