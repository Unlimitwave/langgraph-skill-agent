from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import PrivateAttr
from llama_index.core import (  # type: ignore
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding  # type: ignore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever  # type: ignore
from llama_index.core.schema import NodeWithScore  # type: ignore
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent

_RAG_RETRIEVER: Optional[BaseRetriever] = None

logger = logging.getLogger(__name__)


def _trace(msg: str) -> None:
    if os.environ.get("RAG_TRACE", "").strip() in {"1", "true", "yes", "on"}:
        logger.info("[RAG_TRACE] %s", msg)


class RemoteOpenAICompatibleEmbedding(BaseEmbedding):
    """Call an OpenAI-compatible /v1/embeddings endpoint for embeddings."""

    _base_url: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _model: str = PrivateAttr()
    _timeout_s: float = PrivateAttr()

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60.0):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    @classmethod
    def class_name(cls) -> str:
        return "RemoteOpenAICompatibleEmbedding"

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self._base_url}/embeddings"  # base_url like http://host:port/v1
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model, "input": inputs}
        t0 = time.perf_counter()
        r = requests.post(url, headers=headers, json=payload, timeout=self._timeout_s)
        r.raise_for_status()
        data = r.json()["data"]
        _trace(
            f"embeddings POST took {time.perf_counter() - t0:.3f}s "
            f"(n_inputs={len(inputs)} model={self._model!r})"
        )
        return [item["embedding"] for item in data]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self._embed(texts)

    # LlamaIndex BaseEmbedding (newer versions) also expects query embedding methods.
    # For OpenAI-compatible embeddings, query/document embeddings are typically identical,
    # so we reuse the same endpoint.

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        # Keep it simple: run sync request in async context.
        # If you need true async, switch to httpx.AsyncClient.
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self._get_text_embedding_batch(texts, **kwargs)


def _build_pdf_file_extractor() -> dict:
    """Use an explicit PDF reader so binary/raw PDF streams are not indexed as text.

    PyMuPDF typically extracts CJK text more reliably than defaults; pypdf is the fallback.
    """
    try:
        from llama_index.readers.file import PyMuPDFReader  # type: ignore

        return {".pdf": PyMuPDFReader()}
    except ImportError:
        pass
    try:
        from llama_index.readers.file import PDFReader  # type: ignore

        return {".pdf": PDFReader()}
    except ImportError as e:
        raise ImportError(
            "请安装 PDF 解析依赖：pip install llama-index-readers-file pymupdf"
        ) from e


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _build_remote_openai_compatible_embedding() -> BaseEmbedding:
    base_url = os.environ.get("EMBED_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise ValueError("请在 .env 设置 EMBED_BASE_URL，例如 http://host:port/v1")

    api_key = os.environ.get("EMBED_API_KEY", "dummy")
    model = os.environ.get("EMBED_MODEL", "bge-m3")
    return RemoteOpenAICompatibleEmbedding(base_url=base_url, api_key=api_key, model=model)


def _build_or_load_index(embed_model: BaseEmbedding) -> VectorStoreIndex:
    data_dir = Path(os.environ.get("RAG_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
    storage_dir = Path(os.environ.get("RAG_STORAGE_DIR", str(PROJECT_ROOT / "storage"))).expanduser()

    data_dir = data_dir if data_dir.is_absolute() else (PROJECT_ROOT / data_dir)
    storage_dir = storage_dir if storage_dir.is_absolute() else (PROJECT_ROOT / storage_dir)

    force_rebuild = _env_truthy("RAG_FORCE_REBUILD")
    if force_rebuild and storage_dir.exists():
        shutil.rmtree(storage_dir)
        _trace(f"removed storage_dir for rebuild (RAG_FORCE_REBUILD): {storage_dir}")

    if storage_dir.exists():
        t0 = time.perf_counter()
        sc = StorageContext.from_defaults(persist_dir=str(storage_dir))
        index = load_index_from_storage(sc, embed_model=embed_model)
        _trace(f"load_index_from_storage took {time.perf_counter() - t0:.3f}s (dir={storage_dir})")
        return index

    if not data_dir.exists():
        raise ValueError(
            f"知识库目录不存在：{data_dir}（请创建并放入文档，或设置 RAG_DATA_DIR）"
        )

    pdf_extractor = _build_pdf_file_extractor()
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(str(data_dir), file_extractor=pdf_extractor).load_data()
    _trace(f"SimpleDirectoryReader.load_data took {time.perf_counter() - t0:.3f}s (dir={data_dir})")

    t1 = time.perf_counter()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    _trace(f"VectorStoreIndex.from_documents took {time.perf_counter() - t1:.3f}s (n_docs={len(docs)})")

    t2 = time.perf_counter()
    index.storage_context.persist(persist_dir=str(storage_dir))
    _trace(f"index.persist took {time.perf_counter() - t2:.3f}s (dir={storage_dir})")
    return index


class HybridRRFRetriever(BaseRetriever):
    """Hybrid retriever that fuses results with Reciprocal Rank Fusion (RRF).

    This avoids any dependency on LlamaIndex Settings.llm / OpenAI keys.
    """

    def __init__(
        self,
        *,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        similarity_top_k: int = 8,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._similarity_top_k = similarity_top_k
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight

    def _node_key(self, nws: NodeWithScore) -> str:
        node = getattr(nws, "node", None)
        node_id = getattr(node, "node_id", None)
        if isinstance(node_id, str) and node_id:
            return node_id
        # Fallback: use text hash-ish key
        text = getattr(node, "text", None) or ""
        return f"text:{hash(text)}"

    def _rrf_fuse(self, ranked_lists: list[tuple[list[NodeWithScore], float]]) -> list[NodeWithScore]:
        t0 = time.perf_counter()
        # key -> (node, score)
        fused: dict[str, tuple[NodeWithScore, float]] = {}
        for results, weight in ranked_lists:
            for rank, nws in enumerate(results, start=1):
                key = self._node_key(nws)
                add = weight * (1.0 / (self._rrf_k + rank))
                if key in fused:
                    prev_nws, prev_score = fused[key]
                    fused[key] = (prev_nws, prev_score + add)
                else:
                    fused[key] = (nws, add)

        fused_list: list[NodeWithScore] = []
        for nws, score in fused.values():
            fused_list.append(NodeWithScore(node=nws.node, score=score))
        fused_list.sort(key=lambda x: (x.score or 0.0), reverse=True)
        out = fused_list[: self._similarity_top_k]
        _trace(
            f"rrf_fuse took {time.perf_counter() - t0:.4f}s "
            f"(candidates={len(fused_list)} top_k={self._similarity_top_k})"
        )
        return out

    def _retrieve(self, query: str) -> list[NodeWithScore]:
        t0 = time.perf_counter()
        vec = self._vector_retriever.retrieve(query)
        t1 = time.perf_counter()
        bm25 = self._bm25_retriever.retrieve(query)
        t2 = time.perf_counter()
        out = self._rrf_fuse(
            [
                (vec, self._vector_weight),
                (bm25, self._bm25_weight),
            ]
        )
        _trace(
            "retrieve breakdown: "
            f"vector={t1 - t0:.3f}s bm25={t2 - t1:.3f}s total={time.perf_counter() - t0:.3f}s "
            f"(vec_n={len(vec)} bm25_n={len(bm25)} out_n={len(out)})"
        )
        return out

    async def _aretrieve(self, query: str) -> list[NodeWithScore]:
        # Simple async: run sync retrievers in async path.
        return self._retrieve(query)


def _build_hybrid_retriever(index: VectorStoreIndex) -> BaseRetriever:
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=8)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=8,
    )
    fusion = HybridRRFRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        similarity_top_k=8,
        rrf_k=60,
        vector_weight=1.0,
        bm25_weight=1.0,
    )
    return fusion


def _get_rag_retriever() -> BaseRetriever:
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is not None:
        return _RAG_RETRIEVER

    t0 = time.perf_counter()
    embed_model = _build_remote_openai_compatible_embedding()
    index = _build_or_load_index(embed_model=embed_model)
    _RAG_RETRIEVER = _build_hybrid_retriever(index)
    _trace(f"_get_rag_retriever init took {time.perf_counter() - t0:.3f}s")
    return _RAG_RETRIEVER

