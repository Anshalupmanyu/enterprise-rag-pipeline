"""
Enterprise RAG Pipeline — Query API Routes

JWT-protected endpoints for RAG queries and health checks.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.services.rag_chain import RAGChainService

router = APIRouter(prefix="/query", tags=["RAG Query"])


class QueryRequest(BaseModel):
    """Incoming query payload."""

    question: str = Field(
        ..., min_length=3, max_length=2000, description="The question to ask"
    )


class SourceInfo(BaseModel):
    """Metadata about a source document used in the answer."""

    filename: str
    chunk_index: int | None = None
    excerpt: str


class QueryResponse(BaseModel):
    """RAG query response with answer, sources, and latency."""

    answer: str
    sources: list[SourceInfo]
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    """Service health status."""

    llm: dict
    vector_store: dict


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question (RAG)",
)
async def query_rag(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Submit a question to the RAG pipeline.

    The system will:
    1. Retrieve the most relevant document chunks from the FAISS vector store
    2. Pass them as context to the Mistral LLM via Ollama
    3. Return a grounded answer with source citations and latency metrics

    **Requires a valid JWT token.**
    """
    try:
        rag_service = RAGChainService()
        result = await rag_service.query(request.question)
        return QueryResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {exc}",
        ) from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check RAG pipeline health",
)
async def health_check():
    """
    Returns the current status of the LLM connection and vector store.
    This endpoint is **not** protected — useful for load balancers / monitoring.
    """
    rag_service = RAGChainService()
    return rag_service.health_check()
