"""Unit tests for per-tenant RAG isolation helpers."""

from llama_index.core.vector_stores.types import FilterOperator

from langgraph_skill_agent.rag.retriever import _tenant_metadata_filters


def test_tenant_metadata_filters_include_user_and_tenant() -> None:
    filters = _tenant_metadata_filters("alice", "acme")
    assert len(filters.filters) == 2
    assert filters.filters[0].key == "user_id"
    assert filters.filters[0].value == "alice"
    assert filters.filters[0].operator == FilterOperator.EQ
    assert filters.filters[1].key == "tenant_id"
    assert filters.filters[1].value == "acme"
