import json

from pydantic import Field, ValidationError

from pharma_agent.llm import ProviderError
from pharma_agent.retrieval import RetrievalEngine
from pharma_agent.schemas import StrictModel


class SearchArgs(StrictModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=12)


class SectionArgs(StrictModel):
    source: str = Field(min_length=1, max_length=200)
    section: str = Field(min_length=1, max_length=400)
    offset: int = Field(default=0, ge=0, le=1000)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the local pharmacology corpus using hybrid FAISS/BM25/RRF retrieval. Use focused queries for each drug/aspect; returns exact source names, section paths and chunk IDs.",
            "parameters": SearchArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document_section",
            "description": "Read detailed evidence from an exact source and section returned by search. Use for complete contraindications, thresholds or interactions. If ambiguous, returns section names. Paginate with next_offset.",
            "parameters": SectionArgs.model_json_schema(),
        },
    },
]


class LocalTools:
    def __init__(self, engine: RetrievalEngine):
        self.engine = engine

    async def execute(self, name: str, arguments: str) -> tuple[dict, list]:
        if len(arguments) > 10000:
            return {"error": "Tool arguments exceed size limit"}, []
        try:
            if name == "search_documents":
                args = SearchArgs.model_validate_json(arguments)
                hits = await self.engine.search(args.query, args.top_k)
                chunks = [chunk for chunk, _ in hits]
                result = {
                    "evidence": [dict(chunk.as_evidence(), score=score) for chunk, score in hits]
                }
                return result, chunks
            if name == "read_document_section":
                args = SectionArgs.model_validate_json(arguments)
                result = self.engine.read_section(args.source, args.section, args.offset)
                chunks = result.get("evidence", [])
                return dict(result, evidence=[c.as_evidence() for c in chunks]), chunks
            return {"error": "Unknown tool. Select an advertised local tool."}, []
        except (ValidationError, json.JSONDecodeError):
            return {"error": "Malformed arguments. Follow the tool JSON schema."}, []
        except ProviderError as exc:
            return {"error": str(exc)}, []
        except ValueError as exc:
            return {"error": str(exc)}, []
