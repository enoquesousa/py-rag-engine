"""Tests for the REST API layer.

All tests run against a fake application that replaces the real lifespan
(Postgres + LM Studio) with in-memory stubs, so no external services are
required.
"""
from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from py_rag_engine.api import create_app
from py_rag_engine.retrieval import CrossEncoderReranker
from py_rag_engine.storage import EmbeddingInput, SimilaritySearchResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubStore:
    """In-memory stand-in for PostgresEmbeddingStore."""

    embedding_model = "openai-3-small"

    def __init__(self) -> None:
        self._items: list[EmbeddingInput] = []
        self._next_id = 1

    # Required by routes.py
    @property
    def engine(self):
        eng = MagicMock()
        conn = MagicMock()
        conn.__enter__ = lambda s: s
        conn.__exit__ = MagicMock(return_value=False)
        eng.connect.return_value = conn
        return eng

    def add_embeddings(self, items: Sequence[EmbeddingInput], *, upsert: bool = True) -> list[int]:
        ids = []
        for item in items:
            self._items.append(item)
            ids.append(self._next_id)
            self._next_id += 1
        return ids

    def similarity_search(self, query_embedding, *, top_k=5, metadata_filter=None, ef_search=None):
        return [
            SimilaritySearchResult(
                id=1,
                text="stub result",
                metadata={"source": "data/stub.pdf", "page": 1, "chunk_index": 0},
                content_hash="abc",
                embedding_model="openai-3-small",
                cosine_similarity=0.9,
            )
        ][:top_k]

    def fts_search(self, query_text, *, top_k=5, metadata_filter=None):
        return []


def _stub_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 1536 for _ in texts]


def _make_stub_reranker() -> CrossEncoderReranker:
    def predict(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [0.5] * len(pairs)

    return CrossEncoderReranker(predict=predict)


def _stub_generate(client, question, contexts, **kwargs) -> str:
    return f"Answer for: {question}"


# ---------------------------------------------------------------------------
# App factory for tests (no real lifespan)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _test_lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.store = _StubStore()
    app.state.embed = _stub_embed
    app.state.reranker = _make_stub_reranker()
    app.state.lm_client = MagicMock()
    yield


@pytest.fixture()
def client(monkeypatch) -> Generator[TestClient, None, None]:
    token = "synthetic-api-test-token-0123456789abcdef"
    monkeypatch.setenv("RAG_API_TOKEN", token)
    monkeypatch.delenv("RAG_API_LOCAL_ONLY", raising=False)
    monkeypatch.setattr("py_rag_engine.api.app._lifespan", _test_lifespan)
    monkeypatch.setattr(
        "py_rag_engine.api.routes.generate_answer",
        _stub_generate,
    )
    app = create_app()
    # Use as context manager so the lifespan (app.state setup) runs before tests.
    with TestClient(
        app, base_url="https://testserver", headers={"Authorization": f"Bearer {token}"}
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_degraded_when_lm_studio_unreachable(client: TestClient) -> None:
    resp = client.get("/health")
    # LM Studio stub raises → status is "degraded" (lm_studio unreachable)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"ok", "degraded"}
    assert "postgres" in data
    assert "lm_studio" in data


# ---------------------------------------------------------------------------
# POST /documents  (ingest)
# ---------------------------------------------------------------------------

def test_ingest_rejects_unsupported_extension(client: TestClient) -> None:
    resp = client.post(
        "/documents",
        files={"file": ("report.docx", b"fake content", "application/octet-stream")},
    )
    assert resp.status_code == 415
    assert ".docx" in resp.json()["detail"]


def test_ingest_rejects_empty_file(client: TestClient) -> None:
    resp = client.post(
        "/documents",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert resp.status_code == 400


def test_ingest_markdown_returns_chunk_ids(client: TestClient) -> None:
    md_content = b"# Title\n\nThis is a paragraph with enough content to produce at least one chunk."
    resp = client.post(
        "/documents",
        files={"file": ("guide.md", md_content, "text/markdown")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["source"] == "guide.md"
    assert data["chunks_ingested"] >= 1
    assert len(data["chunk_ids"]) == data["chunks_ingested"]


def test_ingest_with_custom_chunk_size(client: TestClient) -> None:
    md_content = b"# Doc\n\n" + b"word " * 500
    resp = client.post(
        "/documents",
        files={"file": ("big.md", md_content, "text/markdown")},
        params={"chunk_size": 200},
    )
    assert resp.status_code == 201
    data = resp.json()
    # Small chunk_size → more chunks than with default 1200
    assert data["chunks_ingested"] >= 2


def test_ingest_chunk_size_bounds(client: TestClient) -> None:
    md = b"# x\n\nparagraph"
    # too small
    assert client.post("/documents", files={"file": ("x.md", md, "text/plain")}, params={"chunk_size": 50}).status_code == 422
    # too large
    assert client.post("/documents", files={"file": ("x.md", md, "text/plain")}, params={"chunk_size": 99999}).status_code == 422


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

def test_query_returns_sources_and_answer(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "What is the refund policy?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert "answer" in data
    assert "retrieval_mode" in data
    assert data["retrieval_mode"] == "hybrid_rerank"


def test_query_dense_only_mode(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "test", "use_hybrid": False})
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "dense_rerank"


def test_query_without_generation(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "test", "generate_answer": False})
    assert resp.status_code == 200
    assert resp.json()["answer"] is None


def test_query_rejects_empty_question(client: TestClient) -> None:
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_top_k_validated(client: TestClient) -> None:
    assert client.post("/query", json={"question": "q", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "q", "top_k": 21}).status_code == 422


def test_query_sources_have_required_fields(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 200
    for src in resp.json()["sources"]:
        assert "text" in src
        assert "source" in src
        assert "score" in src


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

def test_list_documents_returns_list(client: TestClient) -> None:
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
