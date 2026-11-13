import numpy as np
import pytest

from pharma_agent.documents import chunk_document
from pharma_agent.retrieval import RetrievalEngine


class KeywordEmbedder:
    """Deterministic test double, never used by the application or live evaluation."""

    calls = 0

    async def embed(self, texts):
        self.calls += 1
        words = ["warfarin", "apixaban", "metformin", "bleeding", "renal", "pregnancy"]
        return np.array(
            [[1 + text.lower().count(w) for w in words] for text in texts], dtype=np.float32
        )


@pytest.fixture
def chunks():
    return (
        chunk_document(
            "warfarin.md",
            "# Warfarin\nSource: https://example.test/warfarin\n\n## Contraindications\nWarfarin is contraindicated in pregnancy with exceptions.\n\n## Interactions\nWarfarin increases bleeding risk with sertraline.\n",
        )
        + chunk_document(
            "apixaban.md",
            "# Apixaban\n\n## Contraindications\nApixaban is contraindicated in active pathological bleeding.\n",
        )
        + chunk_document(
            "metformin.md",
            "# Metformin\n\n## Renal function\nMetformin is contraindicated below an eGFR of 30 mL/min/1.73 m2.\n",
        )
    )


@pytest.fixture
def embedder():
    return KeywordEmbedder()


@pytest.fixture
async def engine(chunks, embedder):
    return RetrievalEngine(
        chunks, await embedder.embed([c.embedding_text for c in chunks]), embedder
    )
