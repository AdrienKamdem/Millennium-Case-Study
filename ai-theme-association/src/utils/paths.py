"""Every artifact path in one place.

Before this module the same string appeared in a dozen files -- `classified.parquet`
alone was hardcoded in six -- so moving an output meant grepping and hoping. Worse, the
duplication had already drifted: `ai_betas.parquet` was written to `data/FACTOR/` and
read from `data/processed/` in different modules, and only a fallback in
reconcile/align.py hid it.

Rules:
  - a module never writes a path literal; it imports the constant
  - every path is absolute, derived from ROOT, so scripts work from any directory
    (previously every module assumed you had cd'd to the repo root)
  - a new artifact gets a constant here first

LAYOUT
  data/raw/<source>/    cached HTTP responses, keyed by request hash. Regenerable,
                        gitignored, and the reason the pipeline re-runs offline.
  data/interim/         per-ticker partials and the batch id. Crash insurance.
  data/processed/       the artifacts the analysis reads.
  data/FACTOR/          Part 2 outputs. Uppercase is legacy -- left alone because the
                        files are committed under it; the inconsistency now lives on
                        this one line instead of in fourteen call sites.
  data/SEC_Filing/      the frozen filing index, committed, the input the pipeline
                        reproduces and diffs against.
  outputs/              tables and figures for the deck.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FACTOR = DATA / "FACTOR"
SEC_FILING = DATA / "SEC_Filing"
NEWS_API = DATA / "NEWS_API"

OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
DOCS = ROOT / "docs"

# --- corpus -----------------------------------------------------------------
FROZEN_FILING_INDEX = SEC_FILING / "sec_filings_2023_present.csv"
FILING_INDEX = INTERIM / "sec_filing_index.parquet"
DOCUMENT_INDEX = INTERIM / "sec_document_index.parquet"
SEC_DOCUMENTS = PROCESSED / "sec_documents.parquet"
CHUNKS = PROCESSED / "chunks.parquet"
AV_NEWS = PROCESSED / "av_news.parquet"
AV_NEWS_COVERAGE = PROCESSED / "av_news_coverage.csv"
AV_CELL_AUDIT = PROCESSED / "av_cell_audit.parquet"
NEWS_RAW_GLOB = str(NEWS_API / "alpha_vantage_news_*.json")

# --- classify ---------------------------------------------------------------
CORPUS = PROCESSED / "corpus.parquet"
CLASSIFIED = PROCESSED / "classified.parquet"
BATCH_ID = INTERIM / "batch_id.txt"
TOKEN_LOG = DOCS / "token_log.csv"

# --- index ------------------------------------------------------------------
SALIENCE = PROCESSED / "salience.parquet"
VALENCE = PROCESSED / "valence.parquet"
AIA_INDEX = PROCESSED / "aia_index.parquet"

# --- factor (Part 2) --------------------------------------------------------
PRICES = FACTOR / "prices.parquet"
FF_FACTORS = FACTOR / "ff_factors.parquet"
AI_FACTOR = FACTOR / "ai_factor.parquet"
AI_BETAS = FACTOR / "ai_betas.parquet"
AI_BETAS_ROLLING = FACTOR / "ai_betas_rolling.parquet"
# Human-readable companions the factor modules write alongside each parquet.
PRICES_VALIDATION_TXT = FACTOR / "prices_validation.txt"
FF_FACTORS_TXT = FACTOR / "ff_factors.txt"
AI_FACTOR_TXT = FACTOR / "ai_factor.txt"
AI_BETAS_TXT = FACTOR / "ai_betas.txt"
AI_BETAS_ROLLING_TXT = FACTOR / "ai_betas_rolling.txt"

# --- reconcile --------------------------------------------------------------
RECONCILIATION = PROCESSED / "reconciliation.parquet"

# --- outputs ----------------------------------------------------------------
EXTRACTION_QC = TABLES / "extraction_qc.csv"
EXTRACTION_SAMPLES = TABLES / "extraction_samples"
AIA_RANKING = TABLES / "aia_ranking.csv"
VALIDATION_SAMPLE = TABLES / "validation_sample.csv"
BUZZWORD_TRAP = TABLES / "buzzword_trap.csv"
LIKELY_ERRORS = TABLES / "likely_errors.csv"
REPRESENTATIVE = TABLES / "representative_by_class.csv"
RECONCILIATION_SCATTER = FIGURES / "reconciliation_scatter.png"
SALIENCE_VALENCE_2X2 = FIGURES / "salience_valence_2x2.png"

_WRITABLE = (RAW, INTERIM, PROCESSED, FACTOR, TABLES, FIGURES, DOCS)


def ensure_dirs() -> None:
    """Create every directory a module might write to. Cheap and idempotent."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)
