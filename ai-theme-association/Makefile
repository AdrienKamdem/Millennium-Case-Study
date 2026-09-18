.PHONY: help probe prices sec-index sec-corpus sec-qc chunk news news-repull submit collect corpus classify index factor reconcile all clean-cache probe-edgar

PY := python -m

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

probe:       ## Probe which sources actually reach Jan 2023. Results recorded in config/sources.yaml.
	$(PY) scripts.probe_sources

prices:      ## Download + validate prices and Fama-French factors. No corpus dependency.
	$(PY) src.factor.prices
	$(PY) src.factor.ff_factors

sec-index:   ## Rebuild the EDGAR filing index and diff it against the frozen CSV. ~5 min.
	$(PY) src.corpus.sec_index

sec-corpus:  ## Fetch + extract full text for every SEC document. Cold run ~10 min; re-runs ~1 min.
	$(PY) src.corpus.sec_documents

sec-qc:      ## Extraction quality report + cross-extractor check. Read this before classifying.
	$(PY) src.corpus.quality --dump 5

chunk:       ## Split filings into classifier-sized chunks. Offline, ~1 min.
	$(PY) src.corpus.chunk

news:        ## Normalise the Alpha Vantage dump + print the coverage audit. Offline.
	$(PY) src.corpus.news_normalise

news-repull: ## Re-pull news sliced by quarter (fixes the 1000-most-recent truncation). ~4 min.
	$(PY) src.corpus.alphavantage

submit:      ## Prefilter -> corpus.parquet -> submit the classification batch. COSTS MONEY.
	$(PY) src.classify.prefilter
	$(PY) src.classify.llm_classify --submit

collect:     ## Poll the batch and write classified.parquet.
	$(PY) src.classify.llm_classify --collect

corpus: sec-corpus chunk news   ## The whole corpus: harvest -> extract -> chunk -> news. First run needs network; later runs serve from data/raw/.

classify:    ## Prefilter -> LLM classification + aspect sentiment. Costs money. See `submit`/`collect`.
	$(PY) src.classify.prefilter
	$(PY) src.classify.llm_classify --submit
	$(PY) src.classify.llm_classify --collect

validate:    ## Classifier diagnostics: cross-model agreement, spot review, buzzword trap, name masking.
	$(PY) src.classify.validate

index:       ## Build the AI-association index from classified documents.
	$(PY) src.index.salience
	$(PY) src.index.valence
	$(PY) src.index.aia_index

factor:      ## Construct the AI factor, orthogonalize, run regressions, read betas.
	$(PY) src.factor.ai_factor
	$(PY) src.factor.regressions

reconcile:   ## Align both signals, scatter, divergence analysis.
	$(PY) src.reconcile.align
	$(PY) src.reconcile.scatter
	$(PY) src.reconcile.divergence

all: prices corpus classify index factor reconcile   ## Full pipeline. Runs from cache; no network needed.

clean-cache: ## Delete cached network responses. Forces a full re-harvest; rarely what you want.
	rm -rf data/raw/* data/interim/*

probe-edgar: ## Verify the EDGAR retrieval chain end to end (~5 min).
	$(PY) scripts.probe_edgar
