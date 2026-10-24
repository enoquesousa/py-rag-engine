"""End-to-end smoke test of search.py + reranker.py against real services.

Requires:
  - pgvector configured through EVAL_POSTGRES_URL or POSTGRES_* parts.
    POSTGRES_PASSWORD is required when the full URL is unset. Set
    POSTGRES_PORT=5434 for the local Docker example in the README.
  - LM Studio listening on http://localhost:1234 with bge-m3 embeddings.
  - flashrank installed (pip install 'py-rag-engine[rerank]').

Run:
  python scripts/demo_async_search_rerank.py
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time

if sys.platform == "win32":
    # psycopg's async driver does not support Windows' default ProactorEventLoop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig, PostgresConfig
from py_rag_engine.reranker import FlashrankReranker, rerank_documents
from py_rag_engine.search import AsyncHybridSearcher
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore

CORPUS = [
    "Reciprocal Rank Fusion (RRF) merges multiple ranked lists into one. Each document scores 1/(k+rank) per list, with k usually 60. Documents appearing in several lists rank higher than those in only one.",
    "pgvector is a PostgreSQL extension that stores embedding vectors and supports cosine distance via the <=> operator. HNSW indexes give approximate nearest-neighbour search in milliseconds.",
    "PostgreSQL full-text search uses tsvector and tsquery types, with ts_rank_cd implementing cover-density ranking. websearch_to_tsquery accepts Google-style queries with quoted phrases and OR.",
    "Hybrid retrieval combines dense vector search with sparse keyword search. Vector search captures semantic similarity; keyword search anchors exact terms. Fusing both improves recall in production RAG.",
    "FlashRank is a tiny cross-encoder reranking library based on ONNX runtime. Its ms-marco-MiniLM-L-6-v2 model fits in 23 MB and reranks 100 passages in under a second on CPU.",
    "Cross-encoders re-score query-document pairs with a transformer that sees both texts jointly. They are slower than bi-encoders but produce sharper relevance scores, ideal for the final reranking stage.",
    "SQLAlchemy 2.0 exposes an async engine via create_async_engine. Combined with asyncio.gather, it can dispatch independent SQL queries concurrently against PostgreSQL.",
    "Retrieval Augmented Generation feeds retrieved passages into a language model prompt to ground responses. Quality of retrieval (recall, precision) directly drives answer faithfulness.",
    "The HNSW index in pgvector is built greedily layer by layer. Increasing hnsw.ef_search raises recall at query time at the cost of slightly higher latency.",
    "Asynchronous database calls release the event loop while waiting for the network, so a single Python process can fan out parallel queries without threads.",
    "Lemmatisation is not what PostgreSQL FTS does by default. The english config applies snowball stemming, mapping 'running' to 'run' but not 'better' to 'good'.",
    "Embeddings produced by BGE-M3 are 1024-dimensional and L2-normalised. Cosine similarity over normalised vectors equals their dot product.",
]


def _conn_url() -> str:
    """Use the shared credential validation and URL escaping rules."""
    return PostgresConfig.from_env().url


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest(sync_url: str, embedder: LMStudioClient, *, model: str, table: str) -> None:
    engine = create_engine(sync_url, future=True)
    try:
        with engine.begin() as conn:
            conn.execute(sql_text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        store = PostgresEmbeddingStore(
            engine, embedding_model=model, table_name=table,
        )
        store.create_schema()
        print(f"[ingest] schema created for table '{table}' (model={model})")

        vectors = embedder.embed(CORPUS, model=model)
        items = [
            EmbeddingInput(
                text=text,
                embedding=vec,
                content_hash=_sha256(text),
                metadata={"source": "demo", "chunk_index": idx},
                embedding_model=model,
            )
            for idx, (text, vec) in enumerate(zip(CORPUS, vectors, strict=True))
        ]
        ids = store.add_embeddings(items)
        print(f"[ingest] inserted {len(ids)} rows  ids={ids[:3]}…")
    finally:
        engine.dispose()


async def run_hybrid_search(
    async_url: str,
    embedder: LMStudioClient,
    query: str,
    *,
    model: str,
    table: str,
) -> list:
    [query_vec] = embedder.embed([query], model=model)
    async_engine = create_async_engine(async_url, future=True)
    try:
        searcher = AsyncHybridSearcher(
            async_engine, table_name=table, embedding_model=model,
        )
        start = time.perf_counter()
        results = await searcher.search(query, query_vec, top_k=20)
        elapsed = time.perf_counter() - start
        print(f"[hybrid] returned {len(results)} docs in {elapsed*1000:.1f} ms")
        return results
    finally:
        await async_engine.dispose()


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    sync_url = _conn_url()
    async_url = sync_url  # psycopg3 driver works in both sync and async modes
    embed_model = os.environ.get("LM_STUDIO_EMBED_MODEL", "text-embedding-bge-m3")
    table = "demo_async_hybrid"

    banner("End-to-end demo: AsyncHybridSearcher + FlashrankReranker")
    print("Postgres   : configured")
    print(f"LM Studio  : http://localhost:1234   model={embed_model}")

    embedder = LMStudioClient(LMStudioConfig(embed_model=embed_model))

    try:
        models = [m["id"] for m in embedder.models()]
        if embed_model not in models:
            print(f"!! embed model '{embed_model}' not loaded; LM Studio has: {models}")
            return 2
    except Exception as exc:
        print(f"!! LM Studio not reachable: {exc}")
        return 2

    storage_model_alias = "bge-m3"  # PostgresEmbeddingStore mapping (1024 dims)
    ingest(sync_url, embedder, model=storage_model_alias, table=table)

    # Query chosen so both branches contribute:
    #   * vector branch matches the broader topic by embedding similarity
    #   * FTS branch finds docs that contain *all* unquoted stems (websearch
    #     semantics are AND), so we pick three clearly-anchored keywords.
    query = "pgvector cosine HNSW"
    banner(f"Query: {query}")
    fused = asyncio.run(
        run_hybrid_search(async_url, embedder, query, model=storage_model_alias, table=table)
    )

    print()
    print("--- Top 5 hybrid (RRF k=60) ---")
    for rank, doc in enumerate(fused[:5], start=1):
        print(
            f"{rank}. id={doc.id}  rrf={doc.rrf_score:.5f}  "
            f"vec_rank={doc.vector_rank}  fts_rank={doc.fts_rank}"
        )
        print(f"     {doc.text[:90]}…")

    banner("Re-ranking with flashrank ms-marco-MiniLM-L-6-v2")
    rerank_cache = os.environ.get(
        "FLASHRANK_CACHE_DIR",
        r"C:\Users\sousa\projects\github-enoquesousa-offline\py-rag-engine\.cache",
    )
    reranker = FlashrankReranker(
        model_name="ms-marco-MiniLM-L-6-v2",
        cache_dir=rerank_cache,
    )
    start = time.perf_counter()
    top_5 = rerank_documents(query, fused, reranker=reranker, top_k=5)
    elapsed = time.perf_counter() - start
    print(f"[rerank] returned {len(top_5)} docs in {elapsed*1000:.1f} ms")

    print()
    print("--- Final top 5 (re-ranked) ---")
    for rank, doc in enumerate(top_5, start=1):
        print(
            f"{rank}. id={doc.id}  confidence={doc.confidence_score:.5f}  "
            f"rerank={doc.rerank_score:.5f}  rrf={doc.rrf_score:.5f}"
        )
        print(f"     {doc.text[:90]}…")

    banner("Checks")
    fts_matches = sum(1 for d in fused if d.fts_rank is not None)
    checks = {
        "RRF returned <= 20 docs": len(fused) <= 20,
        "Re-rank returned exactly 5": len(top_5) == 5,
        "RRF used k=60 (top doc score <= 2/(60+1))": fused[0].rrf_score <= 2.0 / 61.0 + 1e-9,
        "confidence_score == rerank_score": all(
            abs(d.confidence_score - d.rerank_score) < 1e-9 for d in top_5
        ),
        "Hybrid mix: at least one vector hit": any(d.vector_rank is not None for d in fused),
        "Hybrid mix: at least one FTS hit": fts_matches > 0,
    }
    for label, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
