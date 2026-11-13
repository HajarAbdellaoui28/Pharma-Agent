"""FastAPI service and same-origin, dependency-free research interface."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pharma_agent.agent import ResearchAgent
from pharma_agent.config import Settings
from pharma_agent.documents import load_corpus
from pharma_agent.llm import Gateway, ProviderError
from pharma_agent.retrieval import RetrievalEngine
from pharma_agent.schemas import AskRequest, AskResponse
from pharma_agent.tools import LocalTools

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gateway = None
        self.engine = None
        self.agent = None
        self.error = None
        self.lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(settings.max_concurrent_requests)

    async def initialize(self):
        async with self.lock:
            if self.agent:
                return
            try:
                chunks = await asyncio.to_thread(load_corpus, self.settings.corpus_dir)
                if self.gateway is None:
                    self.gateway = Gateway(self.settings)
                self.engine = await RetrievalEngine.build(
                    chunks,
                    self.gateway,
                    self.settings.cache_dir,
                    self.settings.embedding_model,
                    self.settings.base_url or "https://api.openai.com/v1",
                )
                self.agent = ResearchAgent(self.gateway, LocalTools(self.engine), self.settings)
                self.error = None
            except ProviderError as exc:
                self.error = str(exc)
                raise
            except (ValueError, OSError):
                self.error = "The local corpus or embedding index could not be initialized."
                raise ProviderError(self.error) from None


def create_app(settings: Settings | None = None, runtime=None):
    settings = settings or Settings()
    runtime = runtime or Runtime(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await asyncio.wait_for(runtime.initialize(), timeout=120)
        except (ProviderError, TimeoutError):
            # Liveness stays available when upstream initialization is unavailable.
            runtime.error = (
                runtime.error
                or "Index initialization timed out. Retry research when the provider is available."
            )
            logger.warning("Research initialization unavailable; health endpoint remains active.")
        yield
        if runtime.gateway:
            await runtime.gateway.close()

    app = FastAPI(
        title="Pharma Research",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.runtime = runtime

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if (
            request.headers.get("content-length", "").isdigit()
            and int(request.headers["content-length"]) > 16384
        ):
            return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.url.path.startswith(("/ask", "/sources")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    async def health():
        engine = runtime.engine
        return {
            "status": "ok",
            "ready": runtime.agent is not None,
            "research_status": "ready" if runtime.agent else "unavailable",
            "detail": runtime.error,
            "documents": len({c.source for c in engine.chunks}) if engine else 0,
            "chunks": len(engine.chunks) if engine else 0,
            "corpus_version": engine.version if engine else None,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest):
        # Queue acquisition is bounded too; request timeout covers lazy index rebuild + research.
        try:
            async with asyncio.timeout(settings.request_timeout_seconds):
                async with runtime.slots:
                    await runtime.initialize()
                    return await runtime.agent.run(body.question)
        except TimeoutError:
            raise HTTPException(
                504, "Research exceeded the time limit. Try a more focused question."
            ) from None
        except ProviderError as exc:
            raise HTTPException(503, str(exc)) from None
        except Exception:
            # Do not log prompts, evidence, credentials or raw provider exception bodies.
            logger.error("Research failed unexpectedly; request returned a safe error.")
            raise HTTPException(500, "Research could not be completed. Please retry.") from None

    @app.get("/sources/{chunk_id}")
    async def source(chunk_id: str):
        if not runtime.engine:
            raise HTTPException(503, "The corpus index is unavailable.")
        chunk = runtime.engine.by_id.get(chunk_id)
        if chunk is None:
            raise HTTPException(404, "Source passage not found.")
        return chunk.as_evidence()

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/docs", include_in_schema=False)
    async def api_reference():
        return FileResponse(STATIC_DIR / "api-docs.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
