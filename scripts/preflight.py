"""Live capability checks. Prints only safe status data, never credentials/provider bodies."""

import asyncio
import json
from pathlib import Path

from pharma_agent.config import Settings
from pharma_agent.llm import Gateway, ProviderError, safe_provider_error


async def main():
    settings = Settings()
    report = {"model": settings.model, "embedding_model": settings.embedding_model, "checks": {}}
    gateway = Gateway(settings)
    try:
        try:
            msg = await gateway.chat([{"role": "user", "content": "Reply with exactly OK."}])
            report["checks"]["chat"] = {"passed": bool(msg.content)}
        except ProviderError as exc:
            report["checks"]["chat"] = {"passed": False, "error": str(exc)}
        try:
            vectors = await gateway.embed(["warfarin", "metformin renal function"])
            report["checks"]["embeddings"] = {
                "passed": len(vectors) == 2,
                "dimensions": vectors.shape[1],
            }
        except ProviderError as exc:
            report["checks"]["embeddings"] = {"passed": False, "error": str(exc)}
        try:
            msg = await gateway.chat(
                [{"role": "user", "content": "Call capability_probe with value OK."}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "capability_probe",
                            "description": "Test tool calling",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                tool_choice="auto",
            )
            calls = msg.tool_calls or []
            passed = any(
                c.function.name == "capability_probe"
                and json.loads(c.function.arguments).get("value") == "OK"
                for c in calls
            )
            report["checks"]["tool_calling"] = {"passed": passed}
        except (ProviderError, ValueError) as exc:
            report["checks"]["tool_calling"] = {
                "passed": False,
                "error": str(exc) if isinstance(exc, ProviderError) else "Malformed tool JSON",
            }
        # Deliberately invalid route/model tests real provider failure handling without revealing response.
        try:
            await gateway.client.chat.completions.create(
                model="pharma-agent-nonexistent-capability-probe",
                messages=[{"role": "user", "content": "OK"}],
            )
            report["checks"]["error_handling"] = {
                "passed": False,
                "note": "Provider accepted unknown model",
            }
        except Exception as exc:
            report["checks"]["error_handling"] = {
                "passed": True,
                "safe_error": safe_provider_error(exc),
            }
    finally:
        await gateway.close()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
