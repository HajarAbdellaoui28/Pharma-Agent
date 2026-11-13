# Architecture and rationale

FastAPI serves a local-corpus research API and a vanilla HTML/CSS/JavaScript frontend.
LangGraph controls research; an OpenAI-compatible SDK gateway calls the configured proxy.

## Module boundaries

| Module | Responsibility |
|---|---|
| `config.py` | Environment settings, proxy aliases and operational bounds |
| `documents.py` | UTF-8 loading, section-aware chunks and source provenance |
| `retrieval.py` | Validated embedding cache, FAISS, BM25, RRF and section lookup |
| `tools.py` | Function schemas and validated read-only tool execution |
| `llm.py` | Provider calls, structured-output validation, metrics and safe errors |
| `agent.py` | Explicit graph state, conditional research loops and answer rendering |
| `critic.py` | Deterministic claim/evidence checks after semantic review |
| `schemas.py` | Pydantic request, plan, claim, review and response contracts |
| `api.py` | Initialization, concurrency, endpoints and static assets |

## Agent workflow

The planner decomposes the question into research steps. The model receives both local tool
schemas with `tool_choice="auto"`, can call either/both tools, and decides when to stop research.
`search_documents` retrieves hybrid-ranked passages; `read_document_section` reads exact source
sections with pagination. Invalid arguments or missing sections return actionable tool errors.
Collected chunks are deduplicated before drafting atomic claims with evidence identifiers.
A separate critic reviews every claim, coverage, terminology, exceptions, thresholds and units.
Code requires complete claim review, valid evidence IDs, verbatim quotes and numeric provenance.
It assigns citation numbers and emits only accepted claims, with a disclaimer and cautious fallback.
Critic feedback can trigger another research round within the configured budget.

Defaults bound research to two rounds, four tool turns per round, six executions per turn and
60 retained passages. Structured output has two attempts; the SDK has one retry. A 240-second
request deadline and graph recursion limit prevent unbounded work; they do not establish truth.

## Retrieval and cache

The supplied 21 files produce 510 chunks, each retaining its filename, title, heading hierarchy,
source URL, character offsets and line span. Explicit file ordering is consistent across Windows
and Linux. Sections split at whitespace into at most 2,400 characters with about 240 overlap;
IDs hash source, section, offset, text and chunker version.

Embeddings include title, filename, section and passage. Normalized float32 vectors make FAISS
`IndexFlatIP` exact cosine-equivalent search. BM25 preserves medication/numeric lexical signals;
RRF combines positions using `1/(60+rank)`, avoiding incompatible raw score scales. Each ranker
provides at least 24 candidates; searches return six hits by default. Dense-only search is retained.

Corpus vectors load once per process from a valid NPZ cache or are generated at startup. The
cache checks model/provider/corpus/chunker identity, ordered IDs, dimensions and checksum; it
uses no pickle and writes atomically. Query dimensions must match. Searches embed only queries;
section reads need no embeddings. A model change hidden behind an unchanged alias is not fully detectable.

## Design rationale and operating boundaries

- LangGraph makes conditional tool/critic loops inspectable; Pydantic validates external data.
  The SDK gateway avoids another chat-wrapper layer or proxy-specific output-format assumption.
- Exact FAISS search needs no database at this corpus size; NumPy would also be adequate.
  Hybrid retrieval combines semantic and lexical signals; the dense baseline makes it measurable.
- Semantic review assesses entailment; code enforces provenance. Neither proves clinical validity.
- Same-origin static assets avoid a frontend build/service and external CDN dependencies.

One gateway/index is shared per API process; graph state and usage counters are request-local.
Initialization is locked; `/health` stays live if research initialization fails. Provider errors
are sanitized. Questions/evidence reach the configured model provider but are not persisted by
the application. Tools cannot open arbitrary model-supplied paths; the frontend uses text nodes
and restrictive CSP. Docker runs non-root and excludes `.env`. Public deployment needs access
controls, shared rate/budget limits and source governance; multiple workers duplicate the index.
