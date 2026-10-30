r"""End-to-end demo of the RAG pipeline with re-ranking.

Steps:
    1. Ingest a document into chunks via `py_rag_engine.ingestion`.
    2. Generate embeddings via LM Studio (`LMStudioClient`).
    3. Persist chunks + embeddings in PostgreSQL with pgvector.
    4. Run dense ANN search (top candidate_k) and re-rank with a Cross-Encoder
       to keep the top_k most relevant chunks for the query.

Configuration is taken entirely from the environment — no credentials live in
this file. Set either `DEMO_POSTGRES_URL` (full DSN) or the `POSTGRES_*`
parts (`POSTGRES_PASSWORD` plus optional `POSTGRES_USER` / `POSTGRES_HOST` /
`POSTGRES_PORT` / `POSTGRES_DB`).

Run with PowerShell:

    $env:PYTHONPATH               = "src"
    $env:LM_STUDIO_BASE_URL       = "http://localhost:1234"
    $env:LM_STUDIO_EMBED_MODEL    = "text-embedding-bge-m3"
    $env:DEMO_POSTGRES_URL        = "postgresql+psycopg://<user>:<password>@<host>:<port>/<db>"
    python scripts\demo_rerank.py data\gdp_document_0.pdf `
        --query "What is the geotechnical investigation methodology?"
"""
from __future__ import annotations

import argparse
import time

from sqlalchemy import create_engine, text

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig, PostgresConfig
from py_rag_engine.ingestion import ingest_file
from py_rag_engine.retrieval import CrossEncoderReranker, retrieve_with_rerank
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore


def reset_table(engine, table_name: str = "embeddings") -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))


def _print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to a .pdf, .md, or .markdown file.")
    parser.add_argument(
        "--query",
        default="What is the geotechnical investigation methodology used in this report?",
        help="User question to retrieve and re-rank against.",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--reset-table", action="store_true",
        help="Drop the embeddings table before inserting.",
    )
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument(
        "--storage-model", default="bge-m3",
        help="Storage discriminator (must be in EMBEDDING_MODEL_DIMENSIONS).",
    )
    args = parser.parse_args()

    lm_cfg = LMStudioConfig.from_env()
    client = LMStudioClient(lm_cfg)
    postgres_url = PostgresConfig.from_env(var="DEMO_POSTGRES_URL").url

    # ── 1. Ingestion ────────────────────────────────────────────────────────
    _print_section("1) INGESTION")
    chunks = ingest_file(args.input)
    print(f"file              = {args.input}")
    print(f"chunks produced   = {len(chunks)}")
    sample = chunks[0]
    print(
        f"sample chunk[0]   = page={sample.metadata.page} "
        f"chars={len(sample.text)} hash={sample.content_hash[:12]}..."
    )

    # ── 2. Embedding ────────────────────────────────────────────────────────
    _print_section("2) EMBEDDING (LM Studio)")
    print(f"lm_studio_url     = {lm_cfg.base_url}")
    print(f"lm_studio_model   = {lm_cfg.embed_model}")
    t0 = time.perf_counter()
    vectors = client.embed([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - t0
    print(f"chunks_embedded   = {len(vectors)} in {elapsed:.2f}s")
    print(f"vector_dim        = {len(vectors[0])}")

    # ── 3. Persist ──────────────────────────────────────────────────────────
    _print_section("3) PERSIST (PostgreSQL + pgvector)")
    print("postgres          = configured")
    print(f"storage_model     = {args.storage_model}")
    engine = create_engine(postgres_url)
    if args.reset_table:
        reset_table(engine)
        print("table reset      = embeddings dropped")
    store = PostgresEmbeddingStore(engine, embedding_model=args.storage_model)
    store.create_schema()
    inserted_ids = store.add_embeddings([
        EmbeddingInput(
            text=chunk.text,
            embedding=vector,
            content_hash=chunk.content_hash,
            metadata={
                "source": chunk.metadata.source,
                "page": chunk.metadata.page,
                "chunk_index": chunk.metadata.chunk_index,
            },
            embedding_model=args.storage_model,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ])
    print(f"rows_upserted     = {len(inserted_ids)}")

    # ── 4. Dense recall ─────────────────────────────────────────────────────
    _print_section(f"4) DENSE RECALL (pgvector, top {args.candidate_k})")
    query_vector = client.embed([args.query])[0]
    dense_candidates = store.similarity_search(
        query_vector, top_k=args.candidate_k, ef_search=80,
    )
    print(f"query             = {args.query!r}")
    print(f"candidates_pulled = {len(dense_candidates)}")
    print()
    print(f"{'rank':>4} {'cos_sim':>8} {'page':>5} {'idx':>4}  preview")
    print("-" * 78)
    for rank, candidate in enumerate(dense_candidates, start=1):
        preview = candidate.text.replace("\n", " ")[:55]
        page = candidate.metadata.get("page")
        idx = candidate.metadata.get("chunk_index")
        print(f"{rank:>4} {candidate.cosine_similarity:>8.4f} "
              f"{page!s:>5} {idx!s:>4}  {preview}")

    # ── 5. Re-rank ──────────────────────────────────────────────────────────
    _print_section(f"5) RE-RANK (Cross-Encoder, top {args.top_k})")
    print(f"reranker_model    = {args.reranker_model}")
    print("loading cross-encoder (first run downloads ~80MB)...")
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    t0 = time.perf_counter()
    final = retrieve_with_rerank(
        args.query,
        query_vector,
        store,
        reranker,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        ef_search=80,
    )
    elapsed = time.perf_counter() - t0
    print(f"reranked_in       = {elapsed:.2f}s")
    print()
    print(f"{'rank':>4} {'rerank':>8} {'cos_sim':>8} {'page':>5} {'idx':>4}  preview")
    print("-" * 78)
    for rank, item in enumerate(final, start=1):
        preview = item.text.replace("\n", " ")[:55]
        page = item.metadata.get("page")
        idx = item.metadata.get("chunk_index")
        print(
            f"{rank:>4} {item.rerank_score:>8.3f} {item.cosine_similarity:>8.4f} "
            f"{page!s:>5} {idx!s:>4}  {preview}"
        )

    # ── 6. Comparison ───────────────────────────────────────────────────────
    _print_section("6) DENSE vs RERANK COMPARISON")
    dense_ids = [c.id for c in dense_candidates[: args.top_k]]
    rerank_ids = [item.id for item in final]
    moved_up = [
        rerank_ids.index(cid) - dense_ids.index(cid) if cid in dense_ids else None
        for cid in rerank_ids
    ]
    print(f"dense_top_{args.top_k:<3}      = {dense_ids}")
    print(f"rerank_top_{args.top_k:<3}     = {rerank_ids}")
    print(f"position_shift    = {moved_up}  (None = new to top {args.top_k})")


if __name__ == "__main__":
    main()
