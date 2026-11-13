from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DISCLAIMER = "This is a research summary of the supplied corpus, not medical advice. Consult a qualified healthcare professional for clinical decisions."


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=3, max_length=2000)

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3 or not any(char.isalnum() for char in value):
            raise ValueError("Enter a question containing at least three characters")
        return value


class Plan(StrictModel):
    steps: list[str] = Field(min_length=1, max_length=6)


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=1600)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)


class Draft(StrictModel):
    claims: list[Claim] = Field(max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=5)


class Support(StrictModel):
    evidence_id: str
    quote: str = Field(min_length=8, max_length=2200)


class ClaimReview(StrictModel):
    claim_index: int = Field(ge=0)
    verdict: Literal["supported", "unsupported", "ambiguous"]
    reason: str = Field(max_length=1200)
    supports: list[Support] = Field(default_factory=list, max_length=6)


class Critique(StrictModel):
    reviews: list[ClaimReview] = Field(max_length=12)
    needs_more_evidence: bool
    feedback: list[str] = Field(default_factory=list, max_length=8)


class Citation(StrictModel):
    id: str
    chunk_id: str
    source: str
    title: str
    section: str
    line_start: int
    line_end: int
    source_url: str | None
    quote: str


class Event(StrictModel):
    stage: str
    summary: str


class Metrics(StrictModel):
    latency_ms: float = 0
    llm_calls: int = 0
    embedding_calls: int = 0
    tool_calls: int = 0
    research_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class AskResponse(StrictModel):
    answer: str
    status: Literal["verified", "limited"]
    citations: list[Citation]
    sources: list[str]
    workflow: list[Event]
    plan: list[str]
    limitations: list[str]
    metrics: Metrics
    disclaimer: str = DISCLAIMER
