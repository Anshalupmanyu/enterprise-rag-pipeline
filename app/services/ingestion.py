"""
Enterprise RAG Pipeline — Document Ingestion Service

Loads documents from the configured directory, splits them into chunks
using LangChain's RecursiveCharacterTextSplitter, and builds/updates
the FAISS vector store.
"""

import os
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Map file extensions to their LangChain loader classes
LOADER_MAPPING: dict[str, type] = {
    ".txt": TextLoader,
    ".md": TextLoader,
}


class IngestionService:
    """Handles loading, chunking, and indexing documents into FAISS."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _load_documents(self) -> list[Document]:
        """Load all supported documents from the documents directory."""
        docs_path = Path(self.settings.DOCUMENTS_PATH)
        if not docs_path.exists():
            logger.warning("Documents directory does not exist: %s", docs_path)
            return []

        all_documents: list[Document] = []
        for ext, loader_cls in LOADER_MAPPING.items():
            glob_pattern = f"**/*{ext}"
            try:
                loader = DirectoryLoader(
                    str(docs_path),
                    glob=glob_pattern,
                    loader_cls=loader_cls,
                    show_progress=True,
                    use_multithreading=True,
                )
                documents = loader.load()
                logger.info(
                    "Loaded %d documents with extension %s", len(documents), ext
                )
                all_documents.extend(documents)
            except Exception as exc:
                logger.error("Error loading %s files: %s", ext, exc)

        return all_documents

    def _enrich_metadata(self, documents: list[Document]) -> list[Document]:
        """Add source filename and chunk metadata to each document."""
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            doc.metadata["filename"] = os.path.basename(source)
        return documents

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks with overlap."""
        chunks = self.text_splitter.split_documents(documents)
        # Add chunk index metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
        logger.info(
            "Split %d documents into %d chunks (size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            self.settings.CHUNK_SIZE,
            self.settings.CHUNK_OVERLAP,
        )
        return chunks

    async def ingest(self) -> dict:
        """
        Full ingestion pipeline:
        1. Load documents from disk
        2. Enrich metadata
        3. Split into chunks
        4. Build / replace the FAISS vector store
        5. Persist the index to disk

        Returns a summary dict with stats.
        """
        logger.info("Starting document ingestion pipeline …")

        # Step 1 — Load
        documents = self._load_documents()
        if not documents:
            return {
                "status": "warning",
                "message": "No documents found to ingest",
                "documents_loaded": 0,
                "chunks_created": 0,
            }

        # Step 2 — Enrich
        documents = self._enrich_metadata(documents)

        # Step 3 — Chunk
        chunks = self._chunk_documents(documents)

        # Step 4 — Build vector store
        vs_manager = VectorStoreManager()
        vs_manager.build_store(chunks)

        # Step 5 — Persist
        vs_manager.save()

        result = {
            "status": "success",
            "message": "Documents ingested and indexed successfully",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vector_store_path": self.settings.VECTOR_STORE_PATH,
        }
        logger.info("Ingestion complete: %s", result)
        return result
