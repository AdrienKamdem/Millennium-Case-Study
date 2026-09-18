# Appendix
---

# Task 07 — Lead–lag test design

**Design only. Not implemented, by instruction.**

## The question

Does a change in a company's semantic AI association **precede** a change in its
factor-implied AI exposure? If disclosure leads the beta, the text is an early indicator.
If the beta leads the text, companies are writing about AI *after* the market rewards it —
which is a narrative-following effect, not a signal.

**Both directions must be tested.** Testing only the first is how you manufacture a result.

## What I would measure

| Side | Measure | Frequency | Why |
|---|---|---|---|
| Text | Δ `salience_z`, quarter on quarter | quarterly | Filing-driven. A 10-K lands in one quarter; anything finer measures the filing calendar, not the company. |
| Market | Δ rolling AI beta, 126-day window, sampled at quarter end | quarterly | `ai_betas_rolling.parquet` already computes this. 126 days is two quarters — long enough to estimate, short enough to move. |

Both as **cross-sectional z-scores per period**, so a quarter when the whole universe
becomes more AI-exposed nets out. Only relative movement counts.

## Specification

```
Δβ(i, t+1) = a + b·Δsalience_z(i, t) + c·Δβ(i, t) + ε        lead test
Δsalience_z(i, t+1) = a + b·Δβ(i, t) + c·Δsalience_z(i, t) + ε   reverse test
```

Company fixed effects. Driscoll–Kraay or two-way clustered errors — the nine names load on
a common factor, so residuals are cross-sectionally correlated and plain OLS errors would
be far too small. Lags of 1 and 2 quarters; anything longer exhausts the window.

**A result is only interesting if the lead coefficient is significant and the reverse one is not.**

## Causal mechanism

A 10-K or 10-Q is a **scheduled, discrete, legally-reviewed information release.** When a
company materially changes its AI language — particularly in Item 1A, where it is obliged to
disclose threats it might prefer not to volunteer — it is releasing information on a
timetable the market knows about but does not fully read. Filings are demonstrably
under-processed relative to earnings calls and press releases. The proposed lag is the
analyst revision cycle: disclosure → coverage → repricing, which plausibly runs one to two
quarters.

That is the mechanism I would claim. I would **not** claim generic "AI adoption" — it is
unfalsifiable and the brief warns against it.

## Statistical power — the honest part

- 9 names × 15 quarters = 135 observations, minus lags and first differences → **~117**.
- But the nine are cross-sectionally correlated by construction. Effective independent
  observations are closer to **15–20**, not 117.
- Δβ has a standard deviation of roughly 0.3 across the panel. A plausible economic effect —
  one standard deviation of disclosure change moving beta by ~0.1 — is **~0.3σ**.
- At effective n ≈ 20, detecting 0.3σ at 80% power needs roughly n = 90.

**This design can detect a 1σ effect. It cannot detect a plausible one.** A null result
would be uninformative, and I would report it as such rather than as evidence of no
relationship.

**What would make it powered:** the S&P 500 cross-section the brief offers as optional.
~400 names × 15 quarters, with genuine cross-sectional variation, takes effective n into
the hundreds. That is the version worth building, and the reason I would not build the
nine-name version at all.

## What would falsify it

- Reverse coefficient significant and lead coefficient not → companies follow the narrative.
- Both significant → contemporaneous common driver, most likely the AI factor itself moving.
- Lead significant only for names with large Δsalience → a few outliers, not a relationship.

---

# Task 08 — Scaling memo

**2,000+ global companies, refreshed monthly.**

## What breaks first, and it is not the LLM

**1. Prompt caching, which is already broken and currently invisible.**
`cache_control` is set, but the rubric is 572 tokens against Haiku 4.5's 4,096-token
minimum cacheable prefix, so it silently never caches. Cost here: ~$1.20 of $6.16. At
2,000 companies monthly — roughly 400k chunks, ~100k gated — the rubric alone is **~57M
tokens per month billed at full rate.** Fix: pad the prefix past the minimum or move to a
model with a 1,024-token threshold. **Biggest single lever, and it is a one-line change.**

**2. The lexicon gate stops being high-recall.**
It works here because nine tech-adjacent names make "GPU", "inference" and "accelerator"
near-unambiguous. Across 2,000 global companies those terms hit semiconductors, particle
physics, business incubators and statistics. The 20% gate rate would rise and its precision
would fall, so cost rises *and* quality falls together. This is where the embeddings tier I
cut becomes necessary rather than optional — it saves ~£1 at this scale and becomes the
dominant cost control at 400k chunks.

**3. Shrinkage priors have no history for new names.**
Salience shrinks toward the company's own trailing mean, which is correct here and
impossible for a name in its first quarter of coverage. Needs a hierarchical prior:
company → sector → universe, with the weight on each set by how much history exists.

**4. Section parsing degrades unpredictably.**
Intel's cross-reference format defeats Item-boundary detection — 1 of 9 names, 11%. Across
2,000 filers, formats multiply and 11% is optimistic. Section-aware chunking has to become
best-effort with a measured fallback rate reported per filer, not a binary.

**5. Human QC does not scale and is currently load-bearing.**
Four of the five bugs found in this build were caught by reading output, not by an
assertion. At 24,000 filings a month nobody reads anything. Every check in
`src/corpus/quality.py` and `src/classify/validate.py` has to become a gate that fails the
run, with thresholds tuned on this corpus as the baseline.

## What only works because the universe is small

- **Case-based reconciliation.** Three hand-read divergences is the right method at n=9 and
  meaningless at n=2,000. It becomes a screen: rank by |gap|, filter on measurement quality
  (beta t-stat, VIF, substantive-unit count), surface the top decile.
- **One AI factor.** A single semis-and-infrastructure basket is why TEAM shows a negative
  beta. At scale you need at least three — compute, software, and energy/infrastructure —
  and each name assigned by its actual channel.
- **Committing the corpus.** 872 documents fit in a repo. 24,000 a month does not; it needs
  object storage with content-addressed keys, which the cache layer already uses.

## The three changes that matter most

1. **Fix the cache prefix.** One line, and at scale it is most of the bill.
2. **Reinstate the embeddings tier** as the gate, with the lexicon demoted to a recall
   backstop. Cheap per document, and it is what keeps precision from collapsing as the
   universe diversifies.
3. **Turn the QC harness into blocking gates** with thresholds from this run. Without a
   human reading output, silent degradation is the default failure mode — and every bug in
   this build was silent.

Not on the list: a bigger model, a bigger corpus, or more prompt engineering. None of those
are the constraint.

---

# Divergence hypotheses — expanded

The deck carries these in short form. Full reasoning and the evidence behind each.

## TEAM · gap +1.20 sd · text ≫ market

| | |
|---|---|
| semantic z | +0.15 |
| beta | −0.18 (t = −1.84, VIF 1.70) |
| substantive units | 138 |
| trajectory | salience 0.131 → 0.221 over six quarters, monotonic |

Measurement triage clears both sides: the beta is identified and not collinear, and 138
substantive units is well above the thinness threshold. So this is **channel or mispricing.**

Evidence, Atlassian's own filings:
- *"Rovo is Atlassian's AI offering, designed for teamwork rather than individual productivity alone."* (+2, business)
- *"AI at the Center — we have embedded AI capabilities across every surface of our platform."* (+2, business)
- *"We also face competition from AI-native companies and emerging startups that leverage generative AI and large language models."* (−2, risk factors)

**Hypothesis — missing channel.** The AI factor's long leg is AVGO, AMD, TSM, ARM, MRVL,
ANET, VRT, SMCI, ASML, PLTR: semiconductors and infrastructure. A software company whose AI
exposure runs through seat monetisation and competitive displacement has no mechanical
reason to load on it. The factor is mis-specified for this name, and the negative beta is
measuring "not a semiconductor" rather than "not AI-exposed."

**Test:** rebuild the factor from a software-AI basket (MSFT, NOW, CRM, PLTR) and re-estimate.
If the loading appears, it is the factor. If the beta stays negative against both, the
market is pricing Atlassian as an AI *loser* despite rising positive disclosure — and that
is the tradeable case.

## NVDA · gap −0.91 sd · market ≫ text

| | |
|---|---|
| semantic z | +0.69 |
| beta | +0.46 (t = 6.30) — the strongest in the universe |
| salience | 0.31, highest |
| valence | +0.35, mid-table |

**Hypothesis — the semantic measure is bounded and the beta is not.** Salience is a share
with a ceiling of 1; NVIDIA is at 0.31 and cannot move much further, while the market can
price arbitrary earnings leverage. Compounding it, valence is dragged down because NVIDIA's
risk factors are dominated by *"regulatory restrictions"* and export controls — disclosure
it is legally obliged to foreground regardless of how the business is performing.

This is **a limitation of my metric, not a market inefficiency**, and I would not trade it.

**Test:** regress beta on salience alone, dropping valence. If the gap closes, valence is
the culprit and the composite needs a different functional form for high-salience names.

## DELL · gap +0.82 sd · text ≫ market

| | |
|---|---|
| semantic z | +2.44, the highest |
| beta | +0.46 (t = 5.21) |
| valence | +1.15, highest of the nine |

**Hypothesis — the text prices AI revenue, the market prices AI margin.** Dell's filings are
the most uniformly positive in the corpus, and the reason is visible: AI-optimised servers
are its growth narrative. But the same filings carry the counter-evidence the classifier
also found:

- *"ISG includes the Company's Artificial Intelligence-optimized servers offerings."* (+2)
- *"The decreases in gross margin percentage … were driven by a competitive pricing environment."* (−2, MD&A)
- *"The nature of the demand for AI solutions may have adverse effects on our operating performance."* (−1, risk factors)

The market appears to be weighting the second and third; my index weights all three by
count and materiality, and the positive language outnumbers it.

**Test:** restrict valence to MD&A only. If the gap narrows materially, the optimism is
concentrated in segment descriptions rather than in the discussion of economics — which
would mean the index should weight MD&A above Item 1 for valence, not treat them equally.

This is the most interesting of the three, because the disagreement is resolved by material
the pipeline already extracted. The information was there; the aggregation lost it.
