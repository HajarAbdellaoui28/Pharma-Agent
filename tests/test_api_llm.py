import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, AuthenticationError, RateLimitError

from pharma_agent.api import create_app
from pharma_agent.config import Settings
from pharma_agent.llm import Gateway, ProviderError, safe_provider_error
from pharma_agent.schemas import AskResponse, Metrics, Plan


def result():
    return AskResponse(
        answer="Insufficient evidence. This is not medical advice.",
        status="limited",
        citations=[],
        sources=[],
        workflow=[],
        plan=[],
        limitations=["Missing evidence"],
        metrics=Metrics(),
    )


class RuntimeStub:
    def __init__(self):
        self.agent = SimpleNamespace(run=AsyncMock(return_value=result()))
        self.gateway = None
        self.engine = None
        self.error = None
        self.slots = asyncio.Semaphore(2)

    async def initialize(self):
        pass


def test_health_frontend_ask_and_sources():
    runtime = RuntimeStub()
    with TestClient(create_app(Settings(_env_file=None), runtime)) as client:
        assert client.get("/health").status_code == 200
        home = client.get("/")
        assert home.status_code == 200 and "Clarity, grounded" in home.text
        assert "default-src 'self'" in home.headers["content-security-policy"]
        assert client.get("/static/app.js").status_code == 200
        response = client.post("/ask", json={"question": "Warfarin contraindications?"})
        assert response.status_code == 200 and response.json()["status"] == "limited"
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/sources/fake").status_code == 503


async def test_source_lookup_exact_and_missing(engine):
    runtime = RuntimeStub()
    runtime.engine = engine
    with TestClient(create_app(Settings(_env_file=None), runtime)) as client:
        chunk = engine.chunks[0]
        response = client.get("/sources/" + chunk.id)
        assert response.status_code == 200 and response.json()["text"] == chunk.text
        assert response.json()["char_start"] == chunk.char_start
        assert client.get("/sources/unknown").status_code == 404
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").json()["paths"]["/ask"]


def test_degraded_liveness_survives_missing_configuration():
    runtime = RuntimeStub()
    runtime.agent = None
    runtime.initialize = AsyncMock(side_effect=ProviderError("Missing configuration"))
    with TestClient(create_app(Settings(_env_file=None), runtime)) as client:
        health = client.get("/health")
        assert health.status_code == 200 and not health.json()["ready"]
        assert client.post("/ask", json={"question": "Valid question"}).status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": "  "},
        {"question": "xx"},
        {"question": "??!!"},
        {"question": "x" * 2001},
        {"question": 123},
        {"question": "valid question", "extra": 1},
    ],
)
def test_invalid_requests(body):
    with TestClient(create_app(Settings(_env_file=None), RuntimeStub())) as client:
        assert client.post("/ask", json=body).status_code == 422


def test_bad_json_and_oversized_request():
    with TestClient(create_app(Settings(_env_file=None), RuntimeStub())) as client:
        assert (
            client.post(
                "/ask", content="{bad", headers={"Content-Type": "application/json"}
            ).status_code
            == 422
        )
        assert client.post("/ask", content="x" * 17000).status_code == 413


def test_provider_error_and_internal_error_are_safe():
    runtime = RuntimeStub()
    with TestClient(create_app(Settings(_env_file=None), runtime)) as client:
        runtime.agent.run.side_effect = ProviderError("Provider quota limit reached.")
        assert client.post("/ask", json={"question": "valid question"}).status_code == 503
        runtime.agent.run.side_effect = RuntimeError("secret upstream message")
        response = client.post("/ask", json={"question": "valid question"})
        assert response.status_code == 500 and "secret" not in response.text


def test_timeout():
    runtime = RuntimeStub()

    async def slow(question):
        await asyncio.sleep(0.1)

    runtime.agent.run.side_effect = slow
    settings = Settings(_env_file=None).model_copy(update={"request_timeout_seconds": 0.01})
    with TestClient(create_app(settings, runtime)) as client:
        assert client.post("/ask", json={"question": "valid question"}).status_code == 504


def test_base_url_alias_priority_and_secret_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_UR", "https://legacy.test/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://preferred.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-do-not-output")
    assert Settings(_env_file=None).base_url == "https://preferred.test/v1"
    assert "test-secret-do-not-output" not in repr(Settings(_env_file=None))
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert Settings(_env_file=None).base_url == "https://legacy.test/v1"


def test_docker_env_file_literal_quotes_are_normalized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "'fake-key-for-test'")
    monkeypatch.setenv("OPENAI_BASE_URL", "'https://proxy.example/v1'")
    monkeypatch.setenv("OPENAI_MODEL", '"gpt-5.6-terra"')
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "'text-embedding-3-small'")
    settings = Settings(_env_file=None)
    assert settings.api_key.get_secret_value() == "fake-key-for-test"
    assert settings.base_url == "https://proxy.example/v1"
    assert settings.model == "gpt-5.6-terra"
    assert settings.embedding_model == "text-embedding-3-small"


async def test_structured_output_repair_and_bounded_failure():
    gateway = Gateway(Settings(_env_file=None, api_key="fake-test-key"))
    try:
        gateway.chat = AsyncMock(
            side_effect=[
                SimpleNamespace(content="not json"),
                SimpleNamespace(content='{"steps":["Research"]}'),
            ]
        )
        assert (await gateway.structured("system", {}, Plan)).steps == ["Research"]
        assert gateway.chat.await_count == 2
        gateway.chat = AsyncMock(return_value=SimpleNamespace(content="invalid"))
        with pytest.raises(ProviderError, match="two attempts"):
            await gateway.structured("system", {}, Plan)
    finally:
        await gateway.close()


def test_provider_failure_redaction():
    response = httpx.Response(401, request=httpx.Request("POST", "https://secret.test"))
    auth = AuthenticationError(
        "secret credential payload", response=response, body={"secret": "value"}
    )
    assert "secret" not in safe_provider_error(auth)
    rate = RateLimitError(
        "secret", response=httpx.Response(429, request=response.request), body=None
    )
    assert "quota" in safe_provider_error(rate)
    connection = APIConnectionError(request=response.request)
    assert "connectivity" in safe_provider_error(connection)
