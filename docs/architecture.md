# Architecture

The project follows a `src` layout with strict per-responsibility modules.
Library code lives under `src/py_rag_engine/`, CLI entry points under
`scripts/`, gold-standard data under `data/`, and JSON reports under
`reports/`.

## Module Map

| Module | Responsibility |
|---|---|
| `py_rag_engine.config` | `LMStudioConfig` / `PostgresConfig` / `EvalConfig` — env-driven, single source of truth |
| `py_rag_engine.domain` | Shared entities (`DocumentChunk`, `ChunkMetadata`) |
| `py_rag_engine.vector_math` | `cosine_similarity` helpers |
| `py_rag_engine.clients` | HTTP adapters (today: `LMStudioClient`, `detect_chat_model`) |
| `py_rag_engine.ingestion` | File loaders (`load_pdf`, `load_markdown`) + ingestion pipeline |
| `py_rag_engine.chunking` | Recursive and semantic chunking strategies |
| `py_rag_engine.chunker` | Public `SemanticChunker` + `calibrate_distance_threshold` helper |
| `py_rag_engine.embeddings` | Hashing + embedder factories (LM Studio, SentenceTransformer) |
| `py_rag_engine.embedder` | Public `VectorClient` (OpenAI-compatible / SentenceTransformer) with async batching and rate-limit backoff |
| `py_rag_engine.storage` | PostgreSQL + pgvector persistence and ANN search |
| `py_rag_engine.retrieval` | Dense ranking + cross-encoder re-ranking |
| `py_rag_engine.generation` | LLM-grounded answer generation (LM Studio chat) |
| `py_rag_engine.evaluation` | RAGAS-style metrics and the `EvalRunner` orchestrator |
| `py_rag_engine.api.app` | Application factory, startup resources, and shutdown disposal |
| `py_rag_engine.api.access` | Startup access configuration and header/connection authorization |
| `py_rag_engine.api.middleware` | ASGI authentication before body parsing, total body limit, no-store responses |

Each module is a *single responsibility*: clients perform HTTP, embedders
generate vectors, the runner orchestrates. Scripts are thin CLIs (~150 LOC
each) that wire the library together — they do not contain business logic.

## End-to-End Pipeline

```
PDF / Markdown
      │
      ▼
  Ingestion (ingestion.loaders + ingestion.pipeline)
  ├─ Loader picks `.pdf` or `.md`
  ├─ Optional semantic paragraph chunking (chunking.semantic)
  ├─ Recursive character splitting (chunking.recursive)
  └─ SHA-256 dedupe (embeddings.hashing)
      │
      ▼
  Embedding (embeddings.lm_studio_embedder | embeddings.sentence_transformer)
      │  bge-m3 (1024d) via LM Studio, or all-MiniLM-L6-v2 (384d) locally
      ▼
  Storage (storage.postgres.PostgresEmbeddingStore)
      │  pgvector HNSW cosine index, JSONB metadata, idempotent upsert
      ▼
  Dense Recall (storage.similarity_search, top-N candidates)
      │
      ▼
  Re-ranking (retrieval.rerank.CrossEncoderReranker)
      │  ms-marco-MiniLM-L-6-v2, top-K final
      ▼
  Generation (generation.lm_studio_chat)
      │  Grounded answer constrained to retrieved context
      ▼
  Evaluation (evaluation.metrics + evaluation.runner)
      │  Faithfulness / Answer Relevancy / Context Precision per question
      ▼
  JSON Report (reports/eval_report_<UTC>.json)
```

## Ingestion Flow

1. Resolve and validate the input path.
2. Select a loader from the file suffix (`.pdf`, `.md`, `.markdown`).
3. Extract text as page-like units (`LoadedPage`).
4. Optionally split each page into semantic paragraph groups using embeddings.
5. Apply recursive character chunking with dynamic overlap (10–15% of chunk size).
6. Hash normalized chunk text with SHA-256.
7. Emit `DocumentChunk` objects with text, source metadata, and content hash.

## Persistence

`storage.postgres.PostgresEmbeddingStore` defines the embeddings table:

- `embedding`: pgvector `vector(n)`, with `n` from `EMBEDDING_MODEL_DIMENSIONS`
  (1024 for `bge-m3`, 1536 for `openai-3-small`, 384 for `all-MiniLM-L6-v2`).
- `text`: original chunk text.
- `metadata`: PostgreSQL `JSONB` for source, page, chunk_index, and extras.
- `content_hash` + `embedding_model`: idempotent-upsert key.

Indexes:

```sql
CREATE INDEX ix_<t>_embedding_hnsw_cosine
ON <t> USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX ix_<t>_metadata_gin
ON <t> USING gin (metadata);
```

The store supports `table_name=` so the eval pipeline can isolate each
configuration into its own table (`eval_<model>_<chunk>`) without polluting the
default `embeddings` table.

## Evaluation Pipeline

The `EvalRunner` (`evaluation.runner`) runs one configuration in five stages:

1. **Ingest** — `ingest_file(path, chunk_size)` with internal caching so the
   same document is only chunked once per chunk size.
2. **Embed** — calls the supplied `EmbedFn` (LM Studio or SentenceTransformer).
3. **Persist** — drops the per-config table, recreates it, bulk-inserts.
4. **Retrieve + Generate** — for each gold-standard question, runs dense ANN
   search (`top_k=5`, `ef_search=80`) then calls `generate_answer`.
5. **Score** — three RAGAS metrics via `evaluate_samples`, with optional
   fallthrough to the official `ragas` library when
   `EVAL_USE_OFFICIAL_RAGAS=1`.

The top-level driver in `scripts/eval_ragas.py` materialises a list of
(model, chunk_size) configurations, invokes the runner for each, and writes
a single timestamped JSON report aggregating all results.

See [evaluation.md](evaluation.md) for the metric definitions and report
schema.

## Public Chunker / Embedder API

`py_rag_engine.chunker` and `py_rag_engine.embedder` expose a small, async-first
public surface that is independent of the ingestion pipeline. They are useful
when you want to embed and chunk arbitrary text from your own application code
(notebook experiments, batch scripts, alternative pipelines):

```python
import asyncio
from openai import AsyncOpenAI
from py_rag_engine import SemanticChunker, VectorClient, calibrate_distance_threshold

async def main() -> None:
    # OpenAI-compatible client pointing at LM Studio (or OpenAI itself).
    openai_client = AsyncOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    embedder = VectorClient(
        provider="openai",
        model="text-embedding-bge-m3",
        client=openai_client,
        batch_size=32,
        max_retries=5,        # exponential backoff on HTTP 429 / RateLimitError
        initial_backoff=1.0,
    )

    # Auto-tune the cosine-distance threshold from a representative sample.
    chunker = await SemanticChunker.from_sample(
        sample_texts=open("data/eval_document.md", encoding="utf-8").read(),
        embedder=embedder,
        percentile=0.85,
        margin=0.05,
    )

    chunks = await chunker.chunk(text, page=1, source="manual.md")
    for c in chunks:
        print(c.metadata.chunk_index, c.content_hash[:10], c.text[:80])

asyncio.run(main())
```

### `SemanticChunker`

- Async paragraph-based splitter. Splits the input into paragraphs, embeds them,
  and groups adjacent paragraphs whose cosine distance stays below
  `distance_threshold`. A new chunk also starts when `max_paragraphs_per_chunk`
  is hit, providing a hard upper bound on chunk length.
- Each emitted `DocumentChunk` carries `text`, `ChunkMetadata` (`source`, `page`,
  `chunk_index`), and a SHA-256 `content_hash` of the chunk text — enabling
  idempotent upserts in `PostgresEmbeddingStore`.
- Single-paragraph input short-circuits the embedding call, so callers can pass
  small fragments without paying a round-trip.
- `chunk_index_start` allows chunks for a multi-page document to keep monotonic
  indices across page boundaries.

### `calibrate_distance_threshold` and `from_sample`

`calibrate_distance_threshold(sample_texts, *, percentile=0.85, margin=0.05, ...)`
estimates a good threshold from adjacent-paragraph distances in your sample
corpus and returns `percentile(distances) + margin`, clamped to cosine bounds
`[0.0, 2.0]`. `SemanticChunker.from_sample(...)` is a convenience constructor
that runs the calibration and builds the chunker in one call.

Tuning notes:

- The sample must be **representative**: include many same-topic paragraphs so
  the chosen percentile lands on intra-topic distances (i.e. *below* the typical
  topic-shift jump). Tiny samples with only a couple of transitions tend to
  produce thresholds above every distance and collapse the whole document into
  a single chunk.
- Empirical reference for `bge-m3` (LM Studio): intra-topic adjacent distances
  cluster around `~0.45`, inter-topic shifts around `~0.65`. The package default
  `DEFAULT_DISTANCE_THRESHOLD = 0.55` is tuned for that gap.
- If the sample has fewer than two paragraphs (no adjacent pairs to measure),
  the helper returns `fallback_threshold` without calling the embedder.

### `VectorClient`

- Async embedder with two providers: `provider="openai"` (works with any
  OpenAI-compatible endpoint, including LM Studio) and
  `provider="sentence-transformers"` for fully-local inference. The model
  default is provider-aware: `text-embedding-3-small` and `all-MiniLM-L6-v2`
  respectively.
- `get_embeddings(texts)` batches the inputs (`batch_size`) and returns one
  `list[float]` per input. Empty input short-circuits to `[]` without contacting
  the provider.
- Rate-limit handling for the OpenAI path: requests are retried with
  exponential backoff plus optional jitter on `RateLimitError`, `status_code ==
  429`, or any error whose `.response.status_code == 429`. After `max_retries`
  the original exception is re-raised so callers can react. Non-rate-limit
  errors propagate immediately.
- Tests inject a fake OpenAI client and a fake `sleep` callable to assert the
  backoff schedule without sleeping; see `tests/test_embedder_module.py`.

## Configuration

Frozen dataclasses concentrate the environment-driven settings:

| Class | Source | Used by |
|---|---|---|
| `LMStudioConfig` | `LM_STUDIO_*` env vars | `LMStudioClient`, scripts |
| `PostgresConfig` | Named DSN (`EVAL_POSTGRES_URL` by default), otherwise `POSTGRES_*` fields | API lifespan and scripts |
| `EvalConfig` | `EVAL_*` env vars | `scripts/eval_ragas.py` |
| `api.access.AccessConfig` | `RAG_API_TOKEN` or explicit `RAG_API_LOCAL_ONLY=1` | API factory and middleware |

Construct from environment via `LMStudioConfig.from_env()`. Override
fields in tests by passing keyword args directly to the dataclass.

`PostgresConfig.url` remains a connection string. A full DSN is accepted as
provided, including socket or external authentication options. With separate
fields, the password is required, the host and port are validated, and
SQLAlchemy 2.1+ escapes the user, password, and database name. This dependency
requires Python 3.11 or later. The whole DSN is
excluded from the dataclass representation because query parameters can also
contain secrets. Scripts report only that PostgreSQL is configured. Avoid
logging `.url` or passing secrets in command-line arguments.

## HTTP access and request flow

The factory reads and validates `AccessConfig` before the lifespan creates a
database engine or model clients. It refuses to start without an explicit
access mode. This requirement applies to the API factory, not to importing
the package or using its CLI/library interfaces.

```mermaid
flowchart TD
    Client["HTTP client"] --> Gate["Connection and header policy"]
    Gate -->|"denied"| Error["Fixed error response"]
    Gate -->|"authorized"| Body["10 MiB body gate and spool"]
    Body -->|"too large"| Error
    Body -->|"accepted"| Route["FastAPI route and parsing"]
    Route --> Store["PostgreSQL store"]
    Route --> Model["LM Studio and reranker"]
    Store --> Result["Document, query, or health response"]
    Model --> Result
```

The policy is applied to every HTTP route, including health, docs, and OpenAPI.
Authentication occurs before receiving and parsing a multipart upload. Bodies
up to 1 MiB are held in memory; larger bodies spill to a temporary file. The
10 MiB cap includes multipart framing. The spool is closed after the request,
including rejection and disconnect paths. Responses through the middleware
carry `Cache-Control: no-store`. WebSocket connections are closed; this API has
no WebSocket authentication protocol.

In token mode, one `Authorization: Bearer ...` header is checked against a
SHA-256 digest with constant-length `hmac.compare_digest`. The configured
token must be 32–4096 ASCII Bearer-token characters. Missing, empty, malformed,
duplicate, and incorrect credentials are distinguished only with fixed errors;
their values are never included in those errors. Credential query parameters
are rejected. The token is not forwarded to PostgreSQL or LM Studio.

Token-bearing traffic needs HTTPS or an actual loopback connection. Run Uvicorn
with `--no-proxy-headers`. The app rejects `Forwarded` and all `X-Forwarded-*`
headers even when an outer ASGI proxy middleware has already rewritten the
scope. For a same-host TLS proxy, bind the backend to loopback and strip these
headers. See the [README](../README.md#bearer-access-and-https) for commands.

Without a token, `RAG_API_LOCAL_ONLY=1` explicitly allows same-machine clients.
The peer must be a literal loopback address; `Host` must identify loopback or
`localhost`; an `Origin`, if present, must match the authority exactly after
scheme/host/default-port normalization. Cross-site fetch metadata is refused.
No DNS lookup decides whether a peer or authority is local. Authorization
headers are refused in this mode to avoid implying that an unconfigured token
was validated.
Do not publish local-only mode through a proxy: a same-host proxy counts as a
loopback peer. Remote access through a proxy must use token mode.

The single token grants access to the entire corpus. `metadata_filter` and
`embedding_model` select retrieval results; they are not tenant permissions.
There are no sessions, JWTs, refresh tokens, user identities, or document ACLs.
Changing the token requires restarting the application, which replaces the
digest snapshot. The plaintext environment value remains accessible to the
process and its administrators. See [SECURITY.md](../SECURITY.md) for the
separate database/model trust boundaries and resource limits.

## Data Layout

- `data/eval_document.md` — source document used by the offline evaluation.
- `data/eval_questions.json` — 10 gold-standard Q&A pairs (schema v1).
- `data/gdp_document_0.pdf` — demo PDF (gitignored binary; metadata is versioned).
- `reports/` — timestamped JSON evaluation reports (gitignored).
- `examples/`, `results/` — local working artifacts (gitignored when generated by scripts).
