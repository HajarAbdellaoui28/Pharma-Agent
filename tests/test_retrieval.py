import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from pharma_agent.documents import chunk_document, corpus_version, load_corpus
from pharma_agent.retrieval import RetrievalEngine, normalize, reciprocal_rank_fusion, tokenize
from pharma_agent.tools import LocalTools


def test_loading_utf8_text_nested_and_metadata(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Drug\nSource: https://example.test\n## Safety\nUse with caution.\n", encoding="utf-8-sig"
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("Plain text evidence", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("not corpus")
    chunks = load_corpus(tmp_path)
    assert {c.source for c in chunks} == {"a.md", "nested/b.txt"}
    assert chunks[0].title == "Drug" and chunks[0].source_url == "https://example.test"
    assert chunks[0].section == "Safety"


def test_chunking_bounded_stable_exact_offsets_and_coverage():
    text = (
        "# Drug\n## Safety\n"
        + ("A long medical sentence with numbers 25 and 30. " * 150)
        + "\n\n## Dose\nDose information.\n"
    )
    chunks = chunk_document("drug.md", text, max_chars=300, overlap=30)
    assert chunks == chunk_document("drug.md", text, max_chars=300, overlap=30)
    assert len(chunks) < 40  # Regression: no repeated tiny suffix chunks at section boundaries.
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text == text[c.char_start : c.char_end] and len(c.text) <= 300 for c in chunks)
    safety = [c for c in chunks if c.section == "Safety"]
    assert all(a.char_end >= b.char_start for a, b in zip(safety, safety[1:], strict=False))
    assert chunks[-1].section == "Dose"
    assert chunks[-1].line_start == 5


def test_empty_documents_and_invalid_parameters(tmp_path):
    assert chunk_document("empty.md", "# Heading\n\n") == []
    with pytest.raises(ValueError):
        load_corpus(tmp_path)
    with pytest.raises(ValueError):
        chunk_document("a", "text", 100, 80)


def test_section_hierarchy():
    chunks = chunk_document(
        "a.md",
        "# Drug\n## Warnings\nGeneral warning.\n### Bleeding\nSpecific bleeding risk.\n## Dose\nDose text.\n",
    )
    assert [c.section for c in chunks] == ["Warnings", "Warnings > Bleeding", "Dose"]


def test_real_corpus_offsets_and_inventory():
    root = Path(__file__).resolve().parents[1] / "corpus"
    chunks = load_corpus(root)
    assert len({c.source for c in chunks}) == 21
    assert 400 <= len(chunks) <= 800
    assert all(
        (root / c.source).read_text(encoding="utf-8-sig")[c.char_start : c.char_end] == c.text
        for c in chunks
    )


def test_corpus_order_is_explicitly_case_insensitive_on_all_platforms(tmp_path):
    for name in ["z-last.md", "GLOSSARY.md", "d-first.md"]:
        (tmp_path / name).write_text(
            "# Title\n## Evidence\nA relevant source passage.\n", encoding="utf-8"
        )
    assert [c.source for c in load_corpus(tmp_path)] == ["d-first.md", "GLOSSARY.md", "z-last.md"]


def test_normalization_and_bad_vectors():
    assert np.allclose(normalize(np.array([[3, 4]])), [[0.6, 0.8]])
    for bad in (np.array([[0, 0]]), np.array([[np.nan, 1]]), np.array([1, 2]), np.empty((1, 0))):
        with pytest.raises(ValueError):
            normalize(bad)


async def test_faiss_exact_inner_product(engine):
    assert isinstance(engine.index, faiss.IndexFlatIP)
    query = engine.vectors[2]
    hits = engine.dense(query, 4)
    expected = np.argsort(-(engine.vectors @ query))
    assert hits[0][0] == expected[0]
    assert hits[0][1] == pytest.approx(float((engine.vectors @ query).max()))
    with pytest.raises(ValueError, match="dimensions changed"):
        engine.dense(np.ones(13))


async def test_bm25_exact_medication_numeric_and_no_match(engine):
    assert engine.chunks[engine.lexical("metformin eGFR 30")[0][0]].source == "metformin.md"
    assert engine.lexical("xyzzy-nonexistent") == []
    assert "1.73" in tokenize("eGFR 30 mL/min/1.73 m2")


def test_rrf_rank_and_duplicate_handling():
    result = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert result[0][0] == 1 and result[1][0] == 2
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([[1, 1]]) == [(1, 1 / 61)]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], 0)


async def test_hybrid_and_dense_modes(engine):
    for mode in ("dense", "hybrid"):
        hits = await engine.search("metformin renal 30", 2, mode)
        assert len(hits) == 2 and hits[0][0].source == "metformin.md"
    with pytest.raises(ValueError):
        await engine.search("valid query", 0)
    with pytest.raises(ValueError):
        engine.rank("query", np.ones(6), mode="unknown")


async def test_cache_identity_corruption_and_corpus_invalidation(chunks, embedder, tmp_path):
    first = await RetrievalEngine.build(chunks, embedder, tmp_path, "model-a", "provider-a")
    assert not first.cache_hit
    calls = embedder.calls
    second = await RetrievalEngine.build(chunks, embedder, tmp_path, "model-a", "provider-a")
    assert second.cache_hit and embedder.calls == calls
    for path in tmp_path.glob("*.npz"):
        path.write_bytes(b"broken cache")
    assert not (
        await RetrievalEngine.build(chunks, embedder, tmp_path, "model-a", "provider-a")
    ).cache_hit
    assert not (
        await RetrievalEngine.build(chunks, embedder, tmp_path, "model-b", "provider-a")
    ).cache_hit
    assert not (
        await RetrievalEngine.build(chunks, embedder, tmp_path, "model-a", "provider-b")
    ).cache_hit
    assert corpus_version(chunks[:-1]) != corpus_version(chunks)
    assert not (
        await RetrievalEngine.build(chunks[:-1], embedder, tmp_path, "model-a", "provider-a")
    ).cache_hit


async def test_cache_dimension_and_checksum_validation(chunks, embedder, tmp_path):
    await RetrievalEngine.build(chunks, embedder, tmp_path, "model")
    path = next(tmp_path.glob("*.npz"))
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        vectors = cache["vectors"].copy()
    metadata["dimension"] += 1
    np.savez_compressed(path, vectors=vectors, metadata=json.dumps(metadata))
    assert not (await RetrievalEngine.build(chunks, embedder, tmp_path, "model")).cache_hit


async def test_local_tools_validation_source_lookup_and_search(engine):
    tools = LocalTools(engine)
    result, evidence = await tools.execute(
        "search_documents", '{"query":"warfarin pregnancy","top_k":2}'
    )
    assert len(evidence) == 2 and result["evidence"][0]["id"] == evidence[0].id
    result, evidence = await tools.execute(
        "read_document_section", '{"source":"warfarin.md","section":"Contraindications"}'
    )
    assert len(evidence) == 1 and "pregnancy" in evidence[0].text
    for name, args in [
        ("unknown", "{}"),
        ("search_documents", "bad-json"),
        ("search_documents", '{"query":"abc","extra":1}'),
        ("read_document_section", '{"source":"../.env","section":"secret"}'),
    ]:
        result, evidence = await tools.execute(name, args)
        assert "error" in result and not evidence


async def test_missing_ambiguous_sections_and_pagination(engine):
    result = engine.read_section("warfarin.md", "missing")
    assert "error" in result and "Contraindications" in result["sections"]
    result = engine.read_section("warfarin.md", "Contraindications", offset=10)
    assert result["evidence"] == [] and result["next_offset"] is None
    with pytest.raises(ValueError):
        engine.read_section("warfarin.md", "Contraindications", offset=-1)
