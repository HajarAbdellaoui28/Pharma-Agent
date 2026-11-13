"""Small OpenAI-compatible gateway; no provider-specific structured-output assumptions."""

import contextvars
import json
from typing import TypeVar

import numpy as np
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from pharma_agent.config import Settings
from pharma_agent.schemas import Metrics

usage: contextvars.ContextVar[Metrics | None] = contextvars.ContextVar("usage", default=None)
T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Safe to return to users: never includes a key, proxy URL, or raw provider body."""


def safe_provider_error(exc: Exception) -> str:
    if isinstance(exc, APITimeoutError):
        return "The model provider timed out. Please retry."
    if isinstance(exc, APIConnectionError):
        return "The model provider could not be reached. Check connectivity and configuration."
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "The model provider rejected the credentials or model permissions."
    if status == 429:
        return "The model provider rate or quota limit was reached. Please retry later."
    if status == 404:
        return "The configured model or API route is unavailable. Check model compatibility."
    return "The model provider rejected the request or returned an invalid response."


class Gateway:
    def __init__(self, settings: Settings):
        if not settings.api_key.get_secret_value():
            raise ProviderError("Set OPENAI_API_KEY to enable research.")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.api_timeout_seconds,
            max_retries=1,
        )

    async def close(self) -> None:
        await self.client.close()

    async def chat(self, messages: list[dict], **kwargs):
        stats = usage.get()
        if stats:
            stats.llm_calls += 1
        try:
            result = await self.client.chat.completions.create(
                model=self.settings.model, messages=messages, **kwargs
            )
            if not result.choices:
                raise ProviderError("The model provider returned no completion.")
            if stats and result.usage:
                stats.input_tokens += result.usage.prompt_tokens
                stats.output_tokens += result.usage.completion_tokens
            return result.choices[0].message
        except APIError as exc:
            raise ProviderError(safe_provider_error(exc)) from None

    async def structured(self, system: str, payload: dict, schema: type[T]) -> T:
        messages = [
            {
                "role": "system",
                "content": system
                + "\nReturn ONLY JSON matching this schema: "
                + json.dumps(schema.model_json_schema()),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for _ in range(2):
            message = await self.chat(messages)
            content = (message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError):
                messages.extend(
                    [
                        {"role": "assistant", "content": (message.content or "")[:12000]},
                        {
                            "role": "user",
                            "content": "Invalid JSON/schema. Correct the format; return only the requested JSON object.",
                        },
                    ]
                )
        raise ProviderError("The model returned malformed structured output after two attempts.")

    async def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), 64):
            stats = usage.get()
            if stats:
                stats.embedding_calls += 1
            batch = texts[start : start + 64]
            try:
                response = await self.client.embeddings.create(
                    model=self.settings.embedding_model, input=batch
                )
            except APIError as exc:
                raise ProviderError(safe_provider_error(exc)) from None
            data = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in data] != list(range(len(batch))):
                raise ProviderError("The embedding provider returned incomplete vectors.")
            rows.extend(item.embedding for item in data)
        try:
            vectors = np.asarray(rows, dtype=np.float32)
        except (TypeError, ValueError):
            raise ProviderError("The embedding provider returned invalid vectors.") from None
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ProviderError("The embedding provider returned invalid vectors.")
        return vectors
