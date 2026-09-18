# AI-Theme Association: from semantic signal to factor exposure

Two independent estimates of how strongly nine companies are associated with the AI theme
— one built from what they **write**, one from how their **shares trade** — and a
reconciliation of where the two disagree.

**Universe:** ADBE, CAT, DELL, INTC, MU, NVDA, PG, TEAM, WM · **Window:** Jan 2023 – Sep 2026

- **Part 1** derives a text-based AI-association score per company from public disclosure.
- **Part 2** runs a factor model isolating an AI factor's contribution to each company's returns.
- **Part 3** reconciles the two and interrogates the disagreements.

The two sides are deliberately independent — the semantic score never sees a price, the
factor model never sees the text — so the reconciliation is not circular.

---

## Headline numbers

| | |
|---|---|
| Corpus | **872 SEC documents**, 48.0M chars -> 15,289 chunks |
| Classified | 4,576 by LLM, 18,528 by lexicon rule, 64 failed (0.28%) |
| Cost | **$6.29** against a GBP 100 budget |
| Aspect sentiment works | `risk_factors` **-0.83** vs `business` **+1.68**, inside the same filings |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env          # SEC_USER_AGENT required; ANTHROPIC_API_KEY only for `classify`
set -a; source .env; set +a

make help                    # every target, one line each
```

### What is in this repo

Results are committed, so **nothing needs to be run** to review the work.

| Committed | |
|---|---|
| `data/processed/classified.parquet` | all 23,168 units with their labels. The `text` column is dropped — the labels are the result, the bodies are 20 MB of bulk. Every figure in this README is verifiable from it. |
| `data/processed/sample/classified_sample.parquet` | 600 units stratified by class, **with** text, for reading |
| `data/processed/{aia_index,salience,valence,reconciliation}.parquet` | the index and the reconciliation |
| `data/processed/chunks.parquet` | 15,289 chunks with section labels and table offsets |
| `data/FACTOR/` | Part 2: prices, factors, the AI factor, per-stock betas |
| `data/SEC_Filing/` | the frozen filing index the pipeline reproduces and diffs against |
| `outputs/tables/` | extraction QC, buzzword trap, likely errors, validation sample, ranking |
| `outputs/figures/` | the reconciliation scatter and the salience-valence quadrant |
| `docs/` | sources log and the full token log |

**Not committed, regenerable:** `data/raw/` (408 MB of cached EDGAR HTML — `make corpus`,
~12 min), `sec_documents.parquet` (the pre-chunk full text), `corpus.parquet` (prefilter
output, seconds), and the raw Alpha Vantage JSON (`av_news.parquet` is its processed
form).

## Running

```bash
make prices        # Part 2 inputs: yfinance + Fama-French          free
make corpus        # harvest -> extract -> chunk -> news            ~12 min, network
make sec-qc        # extraction quality + cross-extractor check     free
make classify      # prefilter -> Batch API -> collect              COSTS MONEY (~$6)
make validate      # classifier diagnostics                         free
make index         # salience, valence, AI-association index        free
make factor        # AI factor, orthogonalisation, betas            free
make reconcile     # align, divergence triage, charts               free
```

`make all` runs the lot. Every network response is cached under `data/raw/` keyed by
request hash, so re-runs are offline and free. Only `corpus` (first run) and `classify`
touch the network.

---

## How it works

```
SEC EDGAR --> sec_index --> sec_documents --> chunk -----+
              what exists   full text        3k chars    |
                                                         +--> prefilter --> llm_classify
Alpha Vantage --> news_normalise ------------------------+    lexicon gate   Haiku, batch
                                                                                  |
                                            salience --+                          |
                                            valence  --+--> aia_index <-----------+
                                                       |         |
prices --> ff_factors --> ai_factor --> regressions ---+---------+
           Part 2, price-only, never sees the text     |         |
                                                       +--> reconcile --> scatter
```

### Decisions worth defending

**Full text, not keyword paragraphs.** An earlier version kept only paragraphs containing
an AI term, which makes salience a count of hits over a count of hits. The corpus is whole
documents, so the denominator is real.

**Exhibits are documents.** 428 of 562 filings are 8-Ks, whose primary document is a ~4k
cover page — NVDA's Q1 FY27 earnings release runs 20,790 chars against 3,925 for the 8-K
carrying it. 310 EX-99 exhibits are indexed separately.

**Chunks are 3,000 chars.** `llm_classify` clips its user block at 4,000 and a 10-K is 402k
chars at the median, so whole documents would have been silently truncated to ~1% and still
scored confidently.

**Sections resolve across forms.** 10-K Part I Item 2 is Properties; 10-Q Part I Item 2 is
MD&A. `section_name` maps both to canonical names so `risk_factors` is a single filter. 119
of 134 periodic filings carry boundaries; Intel files by cross-reference, is detected, and
falls back to paragraph chunking rather than being parsed wrongly.

**A lexicon gate, not an embeddings tier.** 23,168 units -> 4,640 sent to the model. A chunk
with no AI term is `ai_relevance=none` by construction; sending it would cost ~$12 to have
the model agree. Gated rows stay in the corpus as the denominator. The embeddings tier in
the original design was cut — at this size it saves ~GBP 1 for an hour of work.

**Shrinkage priors differ by measure.** Salience shrinks toward the company's own trailing
mean (CAT's salience really is 0.005; a peer average would invent exposure it lacks).
Valence shrinks toward zero (PG has 15 substantive units in four years; neutral is the
honest prior). lambda = n/(n+k), k = 10, from `config/taxonomy.yaml`.

---

## Selected results

**Aspect conditioning works.** Mean valence by section, SEC filings:

```
risk_factors          -0.83   (415 chunks)
financial_statements  +1.19
mdna                  +1.47
business              +1.68   (160 chunks)
```

A 2.5-point spread inside the same documents — document-level sentiment cannot produce
that. ADBE and TEAM score -0.98 and -0.97 in risk factors while positive overall: the
companies disclosing AI as a competitive threat in the one section where they must.

**Disclosure versus coverage.** Substantive AI share:

```
        news    sec
NVDA   0.194  0.285    the only name that discloses more than it is covered
CAT    0.110  0.005    narrated as an AI story it barely tells about itself
PG     0.010  0.004
```

---

## Known limitations

**News covers 2026 only.** Alpha Vantage returns the most recent N regardless of
`time_from`; quarter-sliced re-querying returned empty feeds before 2026, confirming
provider retention rather than a query error. Premium prices requests-per-minute, not
archive depth, and starts at $49.99/month. GDELT reaches 2023 free but resolves companies
by text match, unsafe for INTC, MU and WM specifically. **The index time series is
SEC-only**; news is used cross-sectionally. Full reasoning in `docs/sources_log.csv`.

**SEC-only measures what a company chooses to disclose**, quarterly, with DELL and TEAM on
off-calendar fiscal years. It captures what a company *says*, not what is *said about it*.

**Prompt caching never engaged.** `cache_control` is set correctly, but the rubric is 572
tokens against Haiku 4.5's 4,096-token minimum cacheable prefix, so it silently never
cached and the prefix was billed in full on all 4,576 calls — roughly $1.20 of the $6.16
batch. Visible as `cache_read = 0` in `docs/token_log.csv`. A rounding error here; not at
scale.

**Intel has no section boundaries.** Its 15 periodic filings use SEC's permitted
cross-reference format, so its risk-factor language cannot be isolated.

**n = 9.** Rank correlation between the two sides is reported as descriptive only. The
individual disagreements are the result, not the statistic.

**Descoped deliberately:** earnings-call transcripts (no free source covering the window),
IR newsroom scrapers, cross-model agreement, the name-masking look-ahead test (the
mechanism exists as `llm_classify --mask-name`), and rewriting git history to drop the
cached HTML that was committed early.

---

## Layout

```
config/         universe, taxonomy, AI basket, source probe results — the tunables
src/utils/      paths.py (every artifact path), config, cache, tokenlog
src/corpus/     sec_index -> sec_documents -> sec_extract -> chunk; news_normalise; quality
src/classify/   prefilter (lexicon gate) -> llm_classify (Haiku batch) -> validate
src/index/      salience -> valence -> aia_index
src/factor/     prices -> ff_factors -> ai_factor -> regressions        Part 2
src/reconcile/  align -> divergence -> scatter                          Part 3
checks/         standalone audit scripts: cp x.txt x.py && python x.py
docs/           sources log, token log
```

**To move an artifact,** edit `src/utils/paths.py` — nothing else hardcodes a path.
**To retune the index,** edit `config/taxonomy.yaml` — source and materiality weights,
shrinkage k, and the classifier rubric are all generated from it.
**To change chunking,** the four constants at the top of `src/corpus/chunk.py`, each
carrying the measurement that set it.
