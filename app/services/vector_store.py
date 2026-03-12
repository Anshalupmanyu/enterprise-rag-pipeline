"""
Enterprise RAG Pipeline — FAISS Vector Store Manager

Manages the FAISS index lifecycle: creation, loading from disk,
similarity search, and persistence. Uses HuggingFace embeddings.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Module-Level Singleton ───────────────────────────────────────────────────
_vector_store: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily initialise and cache the HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
    return _embeddings


class VectorStoreManager:
    """Manages the FAISS vector store for document retrieval."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = _get_embeddings()

    # ── Build ────────────────────────────────────────────────────────────
    def build_store(self, documents: list[Document]) -> FAISS:
        """Create a new FAISS index from a list of LangChain Documents."""
        global _vector_store
        logger.info("Building FAISS index from %d document chunks …", len(documents))
        _vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        logger.info("FAISS index built with %d vectors", _vector_store.index.ntotal)
        return _vector_store

    # ── Load ─────────────────────────────────────────────────────────────
    def load_store(self) -> Optional[FAISS]:
        """Load a persisted FAISS index from disk, if it exists."""
        global _vector_store
        store_path = Path(self.settings.VECTOR_STORE_PATH)
        if not store_path.exists():
            logger.warning("No persisted vector store found at %s", store_path)
            return None

        logger.info("Loading FAISS index from %s …", store_path)
        _vector_store = FAISS.load_local(
            str(store_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(
            "FAISS index loaded — %d vectors", _vector_store.index.ntotal
        )
        return _vector_store

    # ── Save ─────────────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist the current FAISS index to disk."""
        global _vector_store
        if _vector_store is None:
            logger.warning("No vector store to save")
            return

        store_path = Path(self.settings.VECTOR_STORE_PATH)
        store_path.mkdir(parents=True, exist_ok=True)
        _vector_store.save_local(str(store_path))
        logger.info("FAISS index saved to %s", store_path)

    # ── Search ───────────────────────────────────────────────────────────
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[Document]:
        """Run a similarity search against the loaded index."""
        global _vector_store
        if _vector_store is None:
            raise RuntimeError(
                "Vector store is not initialised. "
                "Please ingest documents first via POST /api/v1/documents/ingest"
            )
        k = k or self.settings.SIMILARITY_TOP_K
        return _vector_store.similarity_search(query, k=k)

    # ── Retriever ────────────────────────────────────────────────────────
    def as_retriever(self, k: Optional[int] = None):
        """Return a LangChain retriever backed by this FAISS store."""
        global _vector_store
        if _vector_store is None:
            raise RuntimeError("Vector store is not initialised.")
        k = k or self.settings.SIMILARITY_TOP_K
        return _vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    # ── Stats ────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Return basic statistics about the vector store."""
        global _vector_store
        if _vector_store is None:
            return {"status": "not_initialised", "total_vectors": 0}
        return {
            "status": "ready",
            "total_vectors": _vector_store.index.ntotal,
            "dimensions": _vector_store.index.d,
            "store_path": self.settings.VECTOR_STORE_PATH,
        }
