"""
Enterprise RAG Pipeline — RAG Chain Service

Connects the FAISS retriever to the Ollama LLM (Mistral) through
a LangChain RetrievalQA chain with a custom enterprise prompt template.
"""

import logging
import time
from typing import Optional

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.core.config import get_settings
from app.services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# ── Enterprise RAG Prompt ────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an enterprise AI assistant. Use the following retrieved context to answer the user's question accurately and concisely.

RULES:
1. Answer ONLY based on the provided context. If the context does not contain enough information, say so clearly.
2. Cite the source document(s) used in your answer.
3. Be professional, precise, and structured in your response.
4. If multiple sources are relevant, synthesise the information coherently.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ── Module-Level LLM Singleton ──────────────────────────────────────────────
_llm: Optional[Ollama] = None


def _get_llm() -> Ollama:
    """Lazily initialise the Ollama LLM client."""
    global _llm
    if _llm is None:
        settings = get_settings()
        logger.info(
            "Initialising Ollama LLM — model=%s, base_url=%s",
            settings.LLM_MODEL,
            settings.OLLAMA_BASE_URL,
        )
        _llm = Ollama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        logger.info("Ollama LLM initialised")
    return _llm


class RAGChainService:
    """Orchestrates retrieval-augmented generation queries."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vs_manager = VectorStoreManager()
        self.llm = _get_llm()

    def _build_chain(self) -> RetrievalQA:
        """Build a LangChain RetrievalQA chain."""
        retriever = self.vs_manager.as_retriever()
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

    async def query(self, question: str) -> dict:
        """
        Run an end-to-end RAG query:
        1. Retrieve relevant chunks from FAISS
        2. Feed context + question to the LLM
        3. Return the answer with sources and latency

        Returns:
            dict with keys: answer, sources, latency_ms, model
        """
        start = time.perf_counter()

        chain = self._build_chain()
        result = chain.invoke({"query": question})

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        # Extract source metadata
        source_docs = result.get("source_documents", [])
        sources = []
        seen = set()
        for doc in source_docs:
            src = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(
                    {
                        "filename": src,
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "excerpt": doc.page_content[:200] + "…"
                        if len(doc.page_content) > 200
                        else doc.page_content,
                    }
                )

        response = {
            "answer": result.get("result", ""),
            "sources": sources,
            "latency_ms": elapsed_ms,
            "model": self.settings.LLM_MODEL,
        }
        logger.info("RAG query completed in %.1fms", elapsed_ms)
        return response

    def health_check(self) -> dict:
        """Check the health of the LLM and vector store."""
        vs_stats = self.vs_manager.get_stats()

        # Try pinging Ollama
        llm_status = "unknown"
        try:
            import httpx

            resp = httpx.get(
                f"{self.settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                llm_status = (
                    "ready"
                    if any(self.settings.LLM_MODEL in m for m in models)
                    else f"model_not_found (available: {models})"
                )
            else:
                llm_status = f"error (HTTP {resp.status_code})"
        except Exception as exc:
            llm_status = f"unreachable ({exc})"

        return {
            "llm": {
                "status": llm_status,
                "model": self.settings.LLM_MODEL,
                "base_url": self.settings.OLLAMA_BASE_URL,
            },
            "vector_store": vs_stats,
        }
