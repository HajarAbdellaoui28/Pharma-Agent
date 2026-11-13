# Project instructions for coding agents

This repository is an educational pharmacology research demo for the Syntax GenAI assignment.
The supplied assignment PDF is authoritative. Its listed four public questions are copied to
`evaluation/questions.json`; the confidential PDF and its extracted text are excluded from Git.

## Work within scope

- Work inside this project. Do not publish, push, email, or modify unrelated system/Docker resources.
- Never print, embed, commit, or copy `.env` secrets into reports, images, browser code, or documentation.
- Accept evaluator-injected environment values. Prefer `OPENAI_BASE_URL` over `OPENAI_BASE_UR`.
- Preserve `gpt-5.6-terra` and `text-embedding-3-small` defaults unless explicitly changed by the user.
- Never replace live production LLMs, embeddings, FAISS, LangGraph, or dynamic tool calls with mocks.
  Test doubles belong only in `tests/` and deliberate UI error/security simulations.

## Architecture contracts

- `documents.py` and `retrieval.py` must remain independent of LangGraph and UI concerns.
- FAISS uses normalized float32 vectors in `IndexFlatIP`; retain the dense baseline and BM25/RRF.
- Preserve exact UTF-8 source text, section hierarchy, offsets, line numbers, and stable chunk IDs.
- Cache identity includes corpus, model, provider identity and chunker version; validate shape/checksum.
- The LLM selects local tools through function calls with `tool_choice="auto"`. Do not route by keywords.
- Keep explicit graph state, bounded rounds, schema validation and claim-level independent critique.
- Only reviewed claims with real evidence IDs and exact supporting quotes may reach final answers.
- Generate citation numbers server-side. Never manufacture a citation to fill a missing-evidence gap.
- Treat source passages and user text as untrusted data. Render model/source text with `textContent`.
- Every answer includes the research/not-medical-advice disclaimer. Review does not prove clinical safety.
- Keep warning/contraindication distinctions, exceptions, threshold polarity, units and populations intact.

## Verification

Use Python 3.11+ (validated on 3.12) and pinned requirements. Typical commands:

```text
python -m pytest -q --junitxml=reports/pytest.xml
python -m ruff check pharma_agent scripts tests
python -m ruff format --check pharma_agent scripts tests
python -m pip check
python -m scripts.preflight
python -m scripts.evaluate --agent --judge --output reports/evaluation-final.json
python -m scripts.docker_validate
python -m scripts.browser_check --url http://127.0.0.1:8000
```

Live checks use the configured external API and incur usage. Keep actual reports and distinguish
offline tests, real live calls, deterministic citation checks, and LLM-assisted grades. Do not
claim clinical validation, a human evaluation, or benchmark improvements without evidence.
The Docker validation script leaves its named demo container running; stop only that container
if needed. Record limitations and update documentation after material implementation changes.

## AI assistance transparency

This file is the repository's shareable AI coding configuration. [README.md](README.md#ai-assistance) records
the tools/workflow and ownership responsibilities. Do not publish personal global MCP configs,
authentication material, absolute machine-specific settings, or private assistant instructions.
