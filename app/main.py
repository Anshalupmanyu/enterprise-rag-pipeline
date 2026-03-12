"""
Enterprise RAG Pipeline — FastAPI Application Entry Point

Creates the FastAPI app with CORS middleware, mounts all API routers,
and loads the persisted FAISS vector store on startup.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.api import auth, query, documents
from app.services.vector_store import VectorStoreManager

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load the persisted FAISS index if available."""
    logger.info("🚀  Enterprise RAG Pipeline starting up …")
    settings = get_settings()
    logger.info("   LLM model  : %s", settings.LLM_MODEL)
    logger.info("   Ollama URL : %s", settings.OLLAMA_BASE_URL)
    logger.info("   Embedding  : %s", settings.EMBEDDING_MODEL)

    try:
        vs_manager = VectorStoreManager()
        store = vs_manager.load_store()
        if store:
            logger.info("   Vector store loaded (%d vectors)", store.index.ntotal)
        else:
            logger.info(
                "   No persisted vector store found — "
                "call POST /api/v1/documents/ingest to build one"
            )
    except Exception as exc:
        logger.warning("   Failed to load vector store: %s", exc)

    yield  # ── application is running ──

    logger.info("👋  Enterprise RAG Pipeline shutting down")


# ── App Factory ──────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.API_TITLE,
        description=(
            "Production-ready Retrieval-Augmented Generation pipeline "
            "powered by Mistral (Ollama), FAISS, LangChain, and "
            "HuggingFace embeddings. JWT-authenticated REST API."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    # ── Root redirect to Swagger UI ──────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


# ── Application instance (imported by uvicorn) ──────────────────────────────
app = create_app()
