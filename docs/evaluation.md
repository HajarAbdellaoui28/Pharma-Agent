# Recorded evaluation results

Baseline full run: November 12, 2026, 15:38:04-15:48:35 UTC, as recorded in the existing project documentation.
Configuration: `gpt-5.6-terra`, `text-embedding-3-small`; 1,536 dimensions, 510 chunks, 21 source files.
Detailed run JSON and validation artifacts are not included in this V1 copy; the figures below
preserve the recorded baseline and are not new measurements of this checkout.

## Methodology

- Four public questions and six additional cases cover lookups, comparisons, thresholds and missing evidence.
- Dense/hybrid rankers share each real query embedding; scores precede agent decomposition and section reads.
- Document Recall@6 is macro-averaged over nine questions with gold sources; the absent-drug case is excluded.
- MRR@6 uses the first relevant source; facet coverage requires a matching source and relevant section.
- Exact quote/citation checks are deterministic. A separate call to the same model family grades semantics.

## Recorded results

| Check or metric | Result |
|---|---|
| Offline pytest | 48/48 passed; 2 dependency deprecation warnings |
| Ruff, compilation and dependency checks | Passed |
| Docker / browser checks | 12/12 / 8/8 passed |
| Preflight / targeted secret check | Passed / passed; 85 textual files scanned |
| Full-run answers / execution errors | 10/10 completed / 0 errors |
| Exact quote validity | 66/66 (100.0%) |
| Automated grounding / completeness | 10/10 / 10/10 |
| Flagged clinical ambiguity | 0/10 |
| Mean / median agent latency | 58.94 / 51.89 seconds |
| Mean model / tool calls | 8.40 / 8.60 |

| Dataset | Retriever | Recall@6 | MRR@6 | Facet coverage@6 |
|---|---|---:|---:|---:|
| Public | Dense | 87.5% | 0.875 | 62.5% |
| Public | Hybrid | 100.0% | 1.000 | 87.5% |
| All gold-source cases | Dense | 88.9% | 0.944 | 77.8% |
| All gold-source cases | Hybrid | 94.4% | 1.000 | 88.9% |

## Interpretation and reproduction

Hybrid retrieved both public comparison documents/facets where dense missed one document and
both facets. It still missed one interaction facet in the initial top six; later tools can recover it.
Document-level recall is easier than exact evidence recall, especially in this small corpus.
The dataset is development-informed, partially public and not held out; hidden questions were unavailable.
Quotes do not prove entailment. Drafting, criticism and grading can share errors; no human clinical
validation or general performance guarantee is claimed. Source snapshots may be incomplete or outdated.
Latency excludes startup embeddings, HTTP queue/transport and judge calls; SDK retries are not
counted as separate model calls. Repeated runs can differ or exhaust research/time budgets.
Use the [README evaluation commands](../README.md#testing-and-evaluation) to produce a new result file;
live checks incur provider usage. The recorded results should not be overwritten without rerunning evaluation.
