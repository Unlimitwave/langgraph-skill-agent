"""Milvus connectivity smoke test (requires a running Milvus instance)."""

import os

import pytest
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

pytestmark = pytest.mark.integration


def _milvus_host_port() -> tuple[str, str]:
    uri = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530").strip()
    if uri.startswith("http://"):
        host_port = uri.removeprefix("http://")
    elif uri.startswith("https://"):
        host_port = uri.removeprefix("https://")
    else:
        host_port = uri
    if ":" in host_port:
        host, port = host_port.rsplit(":", 1)
        return host, port
    return host_port, "19530"


@pytest.fixture(scope="module")
def milvus_connection():
    host, port = _milvus_host_port()
    connections.connect(alias="default", host=host, port=port)
    yield
    connections.disconnect("default")


def test_milvus_create_search_drop_collection(milvus_connection) -> None:
    collection_name = "test_hello_milvus_ci"

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8),
    ]
    schema = CollectionSchema(fields, description="CI smoke test collection")
    collection = Collection(name=collection_name, schema=schema)

    import random

    texts = [
        "Milvus is an open-source vector database",
        "Designed for similarity search",
        "Supports large-scale vector retrieval",
    ]
    embeddings = [[random.random() for _ in range(8)] for _ in texts]
    collection.insert([texts, embeddings])
    collection.flush()

    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 8}},
    )
    collection.load()

    query_vec = [[random.random() for _ in range(8)]]
    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=2,
        output_fields=["text"],
    )
    assert len(results[0]) >= 1

    utility.drop_collection(collection_name)
