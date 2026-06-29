from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import requests
from llama_index.core import (  # type: ignore
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding  # type: ignore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever  # type: ignore
from llama_index.core.vector_stores.types import (  # type: ignore
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)
from pydantic import PrivateAttr

try:
    from llama_index.vector_stores.milvus import MilvusVectorStore  # type: ignore
    from pymilvus import DataType  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "请安装 Milvus 集成：pip install llama-index-vector-stores-milvus pymilvus"
    ) from e

from langgraph_skill_agent.utility.paths import (
    RAG_DATA_DIR,
    RAG_STORAGE_DIR,
    resolve_rag_data_dir,
    resolve_rag_storage_dir,
)
from langgraph_skill_agent.utility.tenant import normalize_tenant_id, normalize_user_id

_RAG_RETRIEVERS: dict[tuple[str, str], BaseRetriever] = {}
_RAG_TENANT_KEYS = ("user_id", "tenant_id")

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

    def _embed(self, inputs: list[str]) -> list[list[float]]:
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

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embedding_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self._embed(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embedding_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self._get_text_embedding_batch(texts, **kwargs)


def _build_pdf_file_extractor() -> dict:
    """Use an explicit PDF reader so binary/raw PDF streams are not indexed as text."""
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


def _resolve_rag_paths(user_id: str) -> tuple[Path, Path]:
    """Per-user RAG dirs; legacy var/data + var/storage for default user when workspace empty."""
    data_dir = resolve_rag_data_dir(user_id)
    storage_dir = resolve_rag_storage_dir(user_id)
    if user_id == "default":
        if (not data_dir.exists() or not any(data_dir.iterdir())) and RAG_DATA_DIR.exists():
            data_dir = RAG_DATA_DIR
        if not _storage_has_index(storage_dir) and _storage_has_index(RAG_STORAGE_DIR):
            storage_dir = RAG_STORAGE_DIR
    return data_dir, storage_dir


def _tenant_metadata_filters(user_id: str, tenant_id: str) -> MetadataFilters:
    return MetadataFilters(
        filters=[
            MetadataFilter(key="user_id", value=user_id, operator=FilterOperator.EQ),
            MetadataFilter(key="tenant_id", value=tenant_id, operator=FilterOperator.EQ),
        ]
    )


def _stamp_docs_with_tenant(docs: list, *, user_id: str, tenant_id: str) -> None:
    for doc in docs:
        doc.metadata["user_id"] = user_id
        doc.metadata["tenant_id"] = tenant_id


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _build_remote_openai_compatible_embedding() -> BaseEmbedding:
    base_url = os.environ.get("EMBED_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise ValueError("请在 .env 设置 EMBED_BASE_URL，例如 http://host:port/v1")

    api_key = os.environ.get("EMBED_API_KEY", "dummy")
    model = os.environ.get("EMBED_MODEL", "bge-m3")
    return RemoteOpenAICompatibleEmbedding(base_url=base_url, api_key=api_key, model=model)


def _milvus_vector_store(*, overwrite: bool) -> MilvusVectorStore:
    """Milvus 2.4+：稠密向量 + 内置 BM25 稀疏字段；hybrid 查询时 Milvus 侧 RRFRanker 融合。"""
    uri = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530").strip()
    token = os.environ.get("MILVUS_TOKEN", "").strip()
    collection_name = os.environ.get("MILVUS_COLLECTION", "rag_llamaindex").strip()
    dim_raw = os.environ.get("EMBED_DIM", "").strip()
    if not dim_raw:
        raise ValueError(
            "使用 Milvus 时请在 .env 设置 EMBED_DIM（与 EMBED_MODEL 输出维度一致，例如 bge-m3 常为 1024）"
        )
    dim = int(dim_raw)
    metric = os.environ.get("MILVUS_METRIC", "IP").strip().upper()
    rrf_k = int(os.environ.get("MILVUS_RRF_K", "60"))
    # 同步检索路径下关闭 async client，避免部分环境下异步客户端问题
    return MilvusVectorStore(
        uri=uri,
        token=token,
        collection_name=collection_name,
        dim=dim,
        overwrite=overwrite,
        enable_sparse=True,
        similarity_metric=metric,
        hybrid_ranker="RRFRanker",
        hybrid_ranker_params={"k": rrf_k},
        use_async_client=False,
        scalar_field_names=list(_RAG_TENANT_KEYS),
        scalar_field_types=[DataType.VARCHAR, DataType.VARCHAR],
    )


def _storage_has_index(persist_dir: Path) -> bool:
    """仅当存在完整本地元数据时才走 load；目录非空但缺文件会误判并触发 FileNotFoundError。"""
    if not persist_dir.is_dir():
        return False
    # 与 llama_index StorageContext.from_defaults(persist_dir=...) 加载路径一致
    docstore = persist_dir / "docstore.json"
    index_store = persist_dir / "index_store.json"
    return docstore.is_file() and index_store.is_file()


def _dir_has_indexable_files(data_dir: Path) -> bool:
    if not data_dir.is_dir():
        return False
    for path in data_dir.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            return True
    return False


def _build_or_load_index(
    embed_model: BaseEmbedding,
    *,
    user_id: str,
    tenant_id: str,
) -> VectorStoreIndex | None:
    data_dir, storage_dir = _resolve_rag_paths(user_id)

    force_rebuild = _env_truthy("RAG_FORCE_REBUILD")
    if force_rebuild and storage_dir.exists():
        shutil.rmtree(storage_dir)
        _trace(
            f"removed storage_dir for rebuild (RAG_FORCE_REBUILD): {storage_dir} "
            f"(user_id={user_id!r} tenant_id={tenant_id!r})"
        )

    # 有本地持久化元数据则从 Milvus 恢复索引图；稠密+BM25 数据在 Milvus collection 内
    if _storage_has_index(storage_dir):
        vector_store = _milvus_vector_store(overwrite=False)
        t0 = time.perf_counter()
        sc = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(storage_dir),
        )
        index = load_index_from_storage(sc, embed_model=embed_model)
        _trace(
            f"load_index_from_storage+milvus took {time.perf_counter() - t0:.3f}s "
            f"(storage_dir={storage_dir} user_id={user_id!r} tenant_id={tenant_id!r})"
        )
        return index

    if not _dir_has_indexable_files(data_dir):
        _trace(
            f"skip index build: no documents in {data_dir} "
            f"(user_id={user_id!r} tenant_id={tenant_id!r})"
        )
        return None

    pdf_extractor = _build_pdf_file_extractor()
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(str(data_dir), file_extractor=pdf_extractor).load_data()
    _stamp_docs_with_tenant(docs, user_id=user_id, tenant_id=tenant_id)
    _trace(
        f"SimpleDirectoryReader.load_data took {time.perf_counter() - t0:.3f}s "
        f"(dir={data_dir} user_id={user_id!r} tenant_id={tenant_id!r})"
    )

    storage_dir.mkdir(parents=True, exist_ok=True)
    # RAG_FORCE_REBUILD=1 时已删本地 storage，需同步 drop Milvus 旧 collection（避免旧 schema 无稀疏）
    vector_store = _milvus_vector_store(overwrite=force_rebuild)
    # 新建索引时勿传 persist_dir：否则会按「已有持久化」去读 docstore.json，目录为空即 FileNotFoundError
    sc = StorageContext.from_defaults(vector_store=vector_store)

    t1 = time.perf_counter()
    index = VectorStoreIndex.from_documents(docs, storage_context=sc, embed_model=embed_model)
    _trace(
        f"VectorStoreIndex.from_documents took {time.perf_counter() - t1:.3f}s "
        f"(n_docs={len(docs)} user_id={user_id!r} tenant_id={tenant_id!r})"
    )

    t2 = time.perf_counter()
    index.storage_context.persist(persist_dir=str(storage_dir))
    _trace(f"index.persist took {time.perf_counter() - t2:.3f}s (dir={storage_dir})")
    return index


def _build_hybrid_retriever(
    index: VectorStoreIndex,
    *,
    user_id: str,
    tenant_id: str,
) -> BaseRetriever:
    """稠密 + Milvus 内置 BM25，RRF 在 Milvus hybrid_search（RRFRanker）中完成。"""
    top_k = int(os.environ.get("RAG_TOP_K", "8"))
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        filters=_tenant_metadata_filters(user_id, tenant_id),
    )


def _get_rag_retriever(user_id: str, tenant_id: str = "default") -> BaseRetriever | None:
    uid = normalize_user_id(user_id)
    tid = normalize_tenant_id(tenant_id)
    cache_key = (uid, tid)
    cached = _RAG_RETRIEVERS.get(cache_key)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    embed_model = _build_remote_openai_compatible_embedding()
    index = _build_or_load_index(embed_model=embed_model, user_id=uid, tenant_id=tid)
    if index is None:
        return None

    retriever = _build_hybrid_retriever(index, user_id=uid, tenant_id=tid)
    _RAG_RETRIEVERS[cache_key] = retriever
    _trace(
        f"_get_rag_retriever init took {time.perf_counter() - t0:.3f}s "
        f"(user_id={uid!r} tenant_id={tid!r})"
    )
    return retriever
