from types import SimpleNamespace

import pytest

from pharma_agent.agent import ResearchAgent
from pharma_agent.config import Settings
from pharma_agent.critic import validate_reviews
from pharma_agent.llm import ProviderError
from pharma_agent.schemas import Claim, ClaimReview, Critique, Draft, Plan, Support
from pharma_agent.tools import LocalTools


def supported(chunk, text=None):
    draft = Draft(
        claims=[
            Claim(
                text=text or "Warfarin is contraindicated in pregnancy with exceptions.",
                evidence_ids=[chunk.id],
            )
        ]
    )
    critique = Critique(
        reviews=[
            ClaimReview(
                claim_index=0,
                verdict="supported",
                reason="Supported by exact text",
                supports=[Support(evidence_id=chunk.id, quote=chunk.text)],
            )
        ],
        needs_more_evidence=False,
    )
    return draft, critique


def test_critic_accepts_valid_and_rejects_fabricated_quote(chunks):
    draft, critique = supported(chunks[0])
    assert validate_reviews(draft, critique, {chunks[0].id: chunks[0]}, "question")[0]
    critique.reviews[0].supports[0].quote = "An invented sentence not present in the source."
    accepted, problems = validate_reviews(draft, critique, {chunks[0].id: chunks[0]}, "question")
    assert not accepted and problems


def test_critic_requires_explicit_boxed_warning_evidence(chunks):
    draft, critique = supported(chunks[0], "Warfarin has a boxed warning.")
    accepted, problems = validate_reviews(draft, critique, {chunks[0].id: chunks[0]}, "question")
    assert not accepted and "Boxed-warning" in problems[0]


def test_critic_can_remove_redundant_draft_citation(chunks):
    draft, critique = supported(chunks[0])
    draft.claims[0].evidence_ids.append(chunks[1].id)
    accepted, problems = validate_reviews(draft, critique, {c.id: c for c in chunks}, "question")
    assert accepted and not problems
    assert {s.evidence_id for s in accepted[0]} == {chunks[0].id}


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_id",
        "missing_review",
        "duplicate_review",
        "unsupported",
        "ambiguous",
        "number",
        "citation",
        "missing_support",
    ],
)
def test_critic_deterministic_guards(chunks, mutation):
    draft, critique = supported(chunks[0])
    if mutation == "unknown_id":
        draft.claims[0].evidence_ids = ["fabricated"]
    elif mutation == "missing_review":
        critique.reviews = []
    elif mutation == "duplicate_review":
        critique.reviews *= 2
    elif mutation in ("unsupported", "ambiguous"):
        critique.reviews[0].verdict = mutation
    elif mutation == "number":
        draft.claims[0].text += " Bleeding risk is 99 percent."
    elif mutation == "citation":
        draft.claims[0].text += " [1]"
    else:
        critique.reviews[0].supports = []
    accepted, problems = validate_reviews(draft, critique, {chunks[0].id: chunks[0]}, "question")
    assert not accepted and problems


class FakeGateway:
    def __init__(self, chunk, first_tool="read_document_section", reject=False, fail_critic=False):
        self.chunk = chunk
        self.first_tool = first_tool
        self.reject = reject
        self.fail_critic = fail_critic
        self.chat_calls = 0
        self.tool_options = []

    async def structured(self, system, payload, schema):
        if schema is Plan:
            return Plan(steps=["Read the drug's contraindications"])
        draft, critique = supported(self.chunk)
        if schema is Draft:
            return draft
        if self.fail_critic:
            raise ProviderError("Malformed review")
        if self.reject:
            critique.reviews[0].verdict = "unsupported"
            critique.needs_more_evidence = True
            critique.feedback = ["Retrieve more evidence"]
        return critique

    async def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.tool_options.append(kwargs)
        if self.chat_calls % 2 == 0:
            return SimpleNamespace(
                tool_calls=[],
                model_dump=lambda **kw: {"role": "assistant", "content": "Evidence collected."},
            )
        arguments = (
            '{"source":"warfarin.md","section":"Contraindications"}'
            if self.first_tool == "read_document_section"
            else '{"query":"warfarin pregnancy","top_k":2}'
        )
        call = {
            "id": f"call_{self.chat_calls}",
            "type": "function",
            "function": {"name": self.first_tool, "arguments": arguments},
        }
        tool_call = SimpleNamespace(model_dump=lambda: call)
        return SimpleNamespace(
            tool_calls=[tool_call],
            model_dump=lambda **kw: {"role": "assistant", "tool_calls": [call]},
        )


@pytest.mark.parametrize("first_tool", ["search_documents", "read_document_section"])
async def test_genuine_graph_dynamic_tool_selection_and_citations(engine, chunks, first_tool):
    gateway = FakeGateway(chunks[0], first_tool)
    agent = ResearchAgent(gateway, LocalTools(engine), Settings(_env_file=None))
    response = await agent.run("Warfarin contraindications?")
    assert response.status == "verified" and response.citations
    assert response.citations[0].quote in engine.by_id[response.citations[0].chunk_id].text
    assert "not medical advice" in response.answer
    stages = [event.stage for event in response.workflow]
    for stage in [
        "planner",
        "tool_agent",
        "execute_tools",
        "collect_evidence",
        "draft_answer",
        "critic",
        "final_answer",
    ]:
        assert stage in stages
    assert gateway.tool_options[0]["tool_choice"] == "auto"
    assert len(gateway.tool_options[0]["tools"]) == 2
    assert response.metrics.tool_calls == 1


async def test_critic_retries_are_bounded_and_rejected_claims_omitted(engine, chunks):
    gateway = FakeGateway(chunks[0], reject=True)
    settings = Settings(_env_file=None, max_research_rounds=2, max_tool_rounds=2)
    response = await ResearchAgent(gateway, LocalTools(engine), settings).run("Warfarin safety?")
    assert response.status == "limited" and not response.citations
    assert response.metrics.research_rounds == 2
    assert gateway.chat_calls <= 4
    assert "contraindicated in pregnancy" not in response.answer
    assert [event.stage for event in response.workflow].count("retry") == 1


async def test_critic_failure_never_emits_unreviewed_claim(engine, chunks):
    gateway = FakeGateway(chunks[0], fail_critic=True)
    response = await ResearchAgent(
        gateway, LocalTools(engine), Settings(_env_file=None, max_research_rounds=1)
    ).run("Warfarin contraindications?")
    assert response.status == "limited" and response.citations == []


async def test_tool_errors_and_missing_evidence_recover_without_fabrication(engine, chunks):
    gateway = FakeGateway(chunks[0], first_tool="nonexistent")
    response = await ResearchAgent(gateway, LocalTools(engine), Settings(_env_file=None)).run(
        "Unknown compound?"
    )
    assert response.status == "limited" and not response.citations
