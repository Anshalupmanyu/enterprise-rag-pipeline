"""
Enterprise RAG Pipeline — Document Management API Routes

JWT-protected endpoints for triggering document ingestion and
reviewing vector store statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.security import get_current_user
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStoreManager

router = APIRouter(prefix="/documents", tags=["Documents"])


class IngestResponse(BaseModel):
    """Result of a document ingestion operation."""

    status: str
    message: str
    documents_loaded: int
    chunks_created: int
    vector_store_path: str | None = None


class StatsResponse(BaseModel):
    """Vector store statistics."""

    status: str
    total_vectors: int
    dimensions: int | None = None
    store_path: str | None = None


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest and index documents",
)
async def ingest_documents(
    current_user: dict = Depends(get_current_user),
):
    """
    Trigger the document ingestion pipeline:

    1. Load all supported documents from `data/documents/`
    2. Split into chunks using RecursiveCharacterTextSplitter
    3. Generate embeddings with HuggingFace all-MiniLM-L6-v2
    4. Build and persist the FAISS vector store

    **Requires a valid JWT token.**
    """
    try:
        service = IngestionService()
        result = await service.ingest()
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get vector store statistics",
)
async def get_stats(
    current_user: dict = Depends(get_current_user),
):
    """
    Return current vector store statistics including total vectors,
    dimensionality, and storage path.

    **Requires a valid JWT token.**
    """
    vs_manager = VectorStoreManager()
    stats = vs_manager.get_stats()
    return StatsResponse(**stats)
