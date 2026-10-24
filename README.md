<div align="center">
<h1>py-rag-engine</h1>

<p>End-to-end Python RAG (Retrieval-Augmented Generation) engine: PDF / Markdown <strong>ingestion</strong>, recursive and <strong>semantic chunking</strong>, embeddings via <strong>LM Studio</strong> (or any OpenAI-compatible endpoint), <strong>PostgreSQL + pgvector</strong> with HNSW cosine index, <strong>hybrid search</strong> (dense ANN + Postgres FTS) fused with <strong>Reciprocal Rank Fusion</strong>, <strong>Cross-Encoder re-ranking</strong>, a <strong>FastAPI</strong> REST API, offline <strong>RAGAS</strong> evaluation, and structured tests on Python 3.11 / 3.12.</p>

  <img src="assets/python.png" alt="py-rag-engine banner" width="600px">

  <br>

[![CI](https://github.com/enoquesousa/py-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/enoquesousa/py-rag-engine/actions/workflows/ci.yml)
[![CodeQL](https://github.com/enoquesousa/py-rag-engine/actions/workflows/codeql.yml/badge.svg)](https://github.com/enoquesousa/py-rag-engine/actions/workflows/codeql.yml)
[![Dependency review](https://img.shields.io/badge/dependency%20review-in%20CI-0085CA?style=flat&logo=githubactions&logoColor=white)](https://github.com/enoquesousa/py-rag-engine/actions/workflows/ci.yml)
[![Publish](https://img.shields.io/badge/Publish-release%20%7C%20manual-blue?style=flat&logo=githubactions&logoColor=white)](https://github.com/enoquesousa/py-rag-engine/actions/workflows/publish.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python&logoColor=white)](https://github.com/enoquesousa/py-rag-engine/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/enoquesousa/py-rag-engine?style=flat)](https://github.com/enoquesousa/py-rag-engine/commits)
[![Ruff](https://img.shields.io/badge/Ruff-linting-261230?style=flat&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![pytest](https://img.shields.io/badge/tests-pytest-blue?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![CodeFactor](https://www.codefactor.io/repository/github/enoquesousa/py-rag-engine/badge)](https://www.codefactor.io/repository/github/enoquesousa/py-rag-engine)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=flat&logo=pre-commit&logoColor=white)](https://github.com/enoquesousa/py-rag-engine/blob/main/.pre-commit-config.yaml)

</div>

---

**py-rag-engine** is a small, production-minded **RAG engine**: **ingest** PDF and Markdown files, **chunk** them recursively (with optional embedding-based semantic splitting), **embed** through LM Studio, **store** vectors in PostgreSQL with `pgvector` (HNSW cosine index) alongside a `tsvector` GIN index for full-text search, **retrieve** with three composable stages (dense ANN + Postgres FTS → Reciprocal Rank Fusion → Cross-Encoder re-rank), and optionally **generate** grounded answers via an LM Studio chat model. A **FastAPI** app exposes the whole pipeline behind `/health`, `/documents`, and `/query`. An offline **RAGAS** runner benchmarks chunk size and embedding-model combinations. Canonical repository: `github.com/enoquesousa/py-rag-engine`.

## Pipeline overview

```
PDF / Markdown
      │
      ▼
  Ingestion ──► Recursive / Semantic Chunking ──► SHA-256 deduplication
      │
      ▼
  Embedding (LM Studio / any OpenAI-compatible endpoint)
      │
      ▼
  PostgreSQL + pgvector  ◄──  HNSW cosine index (bge-m3 · 1024 dims)
                         ◄──  GIN index on tsvector (Postgres FTS)
      │
      ├──► Dense recall  (top-20 via pgvector cosine ANN)
      │
      └──► FTS recall    (top-20 via ts_rank_cd, websearch_to_tsquery)
                │
                ▼
       Reciprocal Rank Fusion (RRF, k=60)
                │
                ▼
  Cross-Encoder re-rank  (ms-marco-MiniLM-L-6-v2, top-5 final)
                │
                ▼
  RerankedResult list ordered by relevance score
```

## Demo (quick smoke test)

Use Python 3.11 or 3.12 for the constrained CI environment below. Create a virtual environment, install the package, and run a fast wiring check against a small sample document — no Postgres or LM Studio required for the unit suite.

**Linux / macOS (bash)**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/api-ci.txt -e ".[api]" pytest httpx

pytest -q
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -c constraints/api-ci.txt -e ".[api]" pytest httpx

python -m pytest -q
```

To exercise the **full pipeline** end-to-end (ingest → embed → store → hybrid search → rerank), install the `embeddings` extra, spin up Postgres + LM Studio (see Quick Start below), and run `python scripts\demo_rerank.py data\gdp_document_0.pdf`. The REST API requires an explicit access mode before it starts; follow the local or Bearer setup below. See [docs/architecture.md](docs/architecture.md) and [docs/evaluation.md](docs/evaluation.md) for the full module map and metric definitions.

## Features

| Area | What you get |
| ---- | -------------- |
| Ingestion | **PDF** page extraction via `pypdf` and **Markdown** (`.md` / `.markdown`) loaders with source-aware metadata. |
| Chunking | **Recursive** splitter with dynamic overlap, plus optional **semantic chunking** using cosine similarity between paragraph embeddings. A standalone async `SemanticChunker` exposes `from_sample(...)` to **auto-calibrate** the cosine-distance threshold from a representative sample (`calibrate_distance_threshold`). |
| Dedup | **SHA-256** content hash per chunk; upserts on `(embedding_model, content_hash)` keep the table idempotent across re-runs. |
| Storage | **PostgreSQL + pgvector** with HNSW cosine index, JSONB metadata GIN index, and a `tsvector` GIN index for full-text search. |
| Retrieval | **Three-stage**: dense pgvector ANN + Postgres FTS (`ts_rank_cd`, `websearch_to_tsquery`) fused via **Reciprocal Rank Fusion (RRF, k=60)**, then re-ranked by a **Cross-Encoder**. |
| Re-ranking | `CrossEncoderReranker` (default `ms-marco-MiniLM-L-6-v2`) with lazy model loading and an injectable `predict` for tests. |
| Generation | Optional **grounded answer** via an LM Studio chat model (`generate_answer` assembles the prompt + citation contexts). |
| API | **FastAPI** with `/health`, `/documents` (upload + list), and `/query`; shared Bearer authentication or explicit restricted local access, with a 10 MiB total request-body limit. |
| Evaluation | Offline **RAGAS** runner with faithfulness / answer relevancy / context precision; outputs a timestamped JSON report and a config ranking. |
| Tests | Unit and authentication regression tests on Python 3.11 and 3.12 in CI; the external integration test is gated on `TEST_POSTGRES_URL` + `LM_STUDIO_BASE_URL`. |

## Tech stack

| Component | Role |
| --------- | ---- |
| Python 3.11+ | Language and runtime (CI runs 3.11 and 3.12) |
| PostgreSQL 16 + pgvector | Vector storage, HNSW cosine ANN, FTS via `tsvector` |
| SQLAlchemy 2.1+ with asyncio support + psycopg 3 | DB engine, connection pool, and escaping of connection fields |
| LM Studio (OpenAI-compatible) | Embeddings (`bge-m3`) and chat (`qwen2.5-7b-instruct`) |
| sentence-transformers | Cross-Encoder re-ranking and local semantic embeddings |
| langchain-text-splitters | Recursive character chunking |
| pypdf | PDF page extraction |
| FastAPI + uvicorn | REST API |
| RAGAS (optional) | Faithfulness / answer relevancy / context precision metrics |
| pytest / pytest-cov | Tests and coverage |
| Ruff | Lint + format |

## Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.11 / 3.12 in CI | `>=3.11` per `pyproject.toml`; required by SQLAlchemy 2.1 |
| PostgreSQL + pgvector | pg16 + pgvector 0.8.2 | via Docker (see below) |
| LM Studio | any | OpenAI-compatible server on `localhost:1234` |
| Embedding model | `gpustack/bge-m3-GGUF` (bge-m3-Q8_0) | 1024d, multilingual |
| Chat model (for eval) | `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M | ~4.7 GB, recommended for 12 GB GPUs |
| sentence-transformers | 5.4.x | Cross-Encoder re-ranking + local embeddings |
| Git LFS | any | to clone the Cross-Encoder weights |

No external job/database server beyond Postgres is required; the API and CLI scripts are stateless.

## Installation and usage

### From source (recommended)

```bash
git clone https://github.com/enoquesousa/py-rag-engine.git
cd py-rag-engine
pip install -e ".[api,embeddings]"
# add ".[eval]" too if you plan to run the RAGAS evaluation
```

### Development install (editable)

```bash
pip install -e ".[api,embeddings,eval]"
pip install pytest
```

### PyPI

There is **no PyPI publish workflow yet** — install from source as shown above. A `publish.yml` workflow would build wheels/sdists on release and upload them via PyPI trusted publishing; the badge link above is wired so that any future publish workflow renders without further edits.

The current `ci.yml` runs the Python test matrix, `pip check`, an audit of installed API/test dependencies, and a package build. `constraints/api-ci.txt` fixes that test environment without installing model weights or the optional embeddings/evaluation stacks. Those optional stacks need their own deployment-specific dependency review. CodeQL runs in its separate workflow.

## Quick Start

### 1. Postgres + pgvector via Docker

```powershell
$password = Read-Host "Postgres password" -AsSecureString
$env:POSTGRES_PASSWORD = [System.Net.NetworkCredential]::new("", $password).Password
$env:POSTGRES_DB       = "rag"
$env:POSTGRES_PORT     = "5434"

docker run -d `
  --name rag-pgvector `
  --env POSTGRES_PASSWORD `
  --env POSTGRES_DB `
  -p "127.0.0.1:$($env:POSTGRES_PORT):5432" `
  pgvector/pgvector:pg16
```

### 2. LM Studio — embedding + chat model

Install [LM Studio](https://lmstudio.ai), download `gpustack/bge-m3-GGUF` (embeddings) and `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M (chat, for grounded answers / evaluation), load **both** in **Developer → Local Server**, and click **Start Server**. Verify:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models | Select-Object -ExpandProperty data | Format-Table id
```

### 3. Run the REST API locally

This opt-in allows clients on the same machine to read and ingest the entire
corpus without a token. The app checks the actual peer, `Host`, `Origin`, and
browser fetch metadata. Keep both Postgres and LM Studio on loopback. A missing
access mode or an empty configured token prevents the API factory from starting.
Do not publish this token-free mode through a proxy; use Bearer mode for remote
access, including when a proxy connects to the app over loopback.

```powershell
Remove-Item Env:\EVAL_POSTGRES_URL -ErrorAction SilentlyContinue
$env:POSTGRES_USER        = "postgres"
$env:POSTGRES_HOST        = "127.0.0.1"
# POSTGRES_PASSWORD, POSTGRES_DB, and POSTGRES_PORT come from step 1.
$env:RAG_API_LOCAL_ONLY   = "1"
$env:LM_STUDIO_BASE_URL    = "http://localhost:1234"
$env:LM_STUDIO_EMBED_MODEL = "text-embedding-bge-m3"
$env:LM_STUDIO_CHAT_MODEL  = "qwen2.5-7b-instruct"

.venv\Scripts\python.exe -m uvicorn "py_rag_engine.api:create_app" --factory --host 127.0.0.1 --port 8001 --no-proxy-headers
```

If `RAG_API_TOKEN` is already set, it takes precedence over local mode and every
request must authenticate. Explicitly remove that variable only when you intend
to use the local, token-free mode. The separate database fields are escaped by
SQLAlchemy; avoid constructing a DSN by interpolating a password into a URL.

### 4. Drive it with curl

```bash
# Health
curl -s http://127.0.0.1:8001/health
# {"status":"ok","postgres":"ok","lm_studio":"ok"}

# Ingest
curl -X POST "http://127.0.0.1:8001/documents?chunk_size=1024" \
     -F "file=@data/gdp_document_0.pdf"

# Hybrid + Cross-Encoder rerank + grounded answer
curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" \
  -d '{"question":"How does HNSW work in pgvector?","top_k":3,
       "use_hybrid":true,"use_rerank":true,"generate_answer":true}'
```

### Bearer access and HTTPS

For authenticated access, set `RAG_API_TOKEN` to a randomly generated secret of
32–4096 ASCII Bearer-token characters. The app uses the token for every HTTP
route, including `/health`, `/docs`, and `/openapi.json`. It is a shared service
credential: its holder can read and ingest all documents. There are no user
accounts, tenant boundaries, JWTs, refresh tokens, or per-document permissions.

For example, in bash, after configuring the database and LM Studio:

```bash
export RAG_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
unset RAG_API_LOCAL_ONLY
python -m uvicorn "py_rag_engine.api:create_app" --factory \
  --host 0.0.0.0 --port 8001 --no-proxy-headers \
  --ssl-certfile /run/secrets/rag-server.crt \
  --ssl-keyfile /run/secrets/rag-server.key
```

Provision a certificate valid for your service name and trusted by clients.
Restrict access to the key files. For a local client using that certificate:

```bash
printf 'Authorization: Bearer %s\n' "$RAG_API_TOKEN" |
  curl --fail --header @- --cacert /path/to/trusted-ca.crt \
    https://localhost:8001/health
```

Use the `Authorization: Bearer ...` header on uploads and queries as well. Never
put the token in a URL: credential query parameters are rejected, including
encoded names and empty values. Duplicate or malformed authorization headers
are rejected. Ordinary browser navigation does not add a Bearer header, so use
a header-capable API client to fetch protected documentation in this mode.

Bearer requests require HTTPS unless the actual connection peer is loopback.
For TLS termination, use a trusted proxy on the same host, bind Uvicorn to
loopback, keep `--no-proxy-headers`, and strip `Forwarded` and all
`X-Forwarded-*` headers before forwarding. The app rejects these headers rather
than trusting client-supplied connection metadata. An external proxy hop over
plaintext is not supported.

Authentication runs before FastAPI parses a request body. The total body limit
is **10 MiB**, including multipart framing, so an upload file must be smaller.
Bodies are spooled to a temporary file after 1 MiB and closed on completion or
failure. This is not a global concurrency limit or a limit on decompressed PDF
content and model work. Configure those resource controls at deployment time.
See [SECURITY.md](SECURITY.md) for credential lifetimes and trust boundaries.

### Retrieval modes (single endpoint)

| `use_hybrid` | `use_rerank` | Mode | Score returned |
|---|---|---|---|
| `false` | `false` | `dense` (pgvector ANN only) | cosine similarity |
| `true`  | `false` | `hybrid` (dense + FTS via RRF) | RRF score |
| `false` | `true`  | `dense_rerank` (cosine + Cross-Encoder) | rerank score |
| `true`  | `true`  | `hybrid_rerank` (RRF + Cross-Encoder) | rerank score |

### End-to-end demo CLI

```powershell
python scripts\demo_rerank.py data\gdp_document_0.pdf `
  --query "What is the geotechnical investigation methodology used in this report?" `
  --reranker-model "$PWD\.cache\ms-marco-MiniLM-L-6-v2"
```

### Programmatic chunker + embedder API

For ad-hoc scripts that just need to embed text or split it into semantic chunks (without the full ingestion → Postgres pipeline), the top-level `SemanticChunker` and `VectorClient` cover the entire flow with async + type-checked APIs:

```python
import asyncio
from openai import AsyncOpenAI
from py_rag_engine import SemanticChunker, VectorClient, calibrate_distance_threshold

async def main() -> None:
    # Any OpenAI-compatible endpoint. LM Studio uses a dummy api_key.
    openai_client = AsyncOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    embedder = VectorClient(
        provider="openai",                # or "sentence-transformers" for local
        model="text-embedding-bge-m3",
        client=openai_client,
        batch_size=32,
        max_retries=5,                    # exponential backoff on HTTP 429
        initial_backoff=1.0,
    )

    # Calibrate the cosine-distance threshold from a representative sample.
    # For tiny / non-representative samples, prefer a fixed threshold.
    threshold = await calibrate_distance_threshold(
        sample_texts=open("data/eval_document.md", encoding="utf-8").read(),
        embedder=embedder,
        percentile=0.85,
        margin=0.05,
    )
    chunker = SemanticChunker(embedder=embedder, distance_threshold=threshold)

    # Equivalent one-liner:
    # chunker = await SemanticChunker.from_sample(sample_text, embedder=embedder)

    text = open("examples/article.md", encoding="utf-8").read()
    chunks = await chunker.chunk(text, page=1, source="article.md")

    for c in chunks:
        # idempotency: c.content_hash is sha256(c.text); safe to upsert
        print(c.metadata.chunk_index, c.content_hash[:10], c.text[:80])

asyncio.run(main())
```

**Key points**

- `SemanticChunker` groups adjacent paragraphs whose cosine distance is below `distance_threshold` and emits `DocumentChunk(text, metadata, content_hash)` — the SHA-256 `content_hash` is what `PostgresEmbeddingStore` uses for idempotent upserts.
- `VectorClient.get_embeddings(texts)` batches the input and retries `RateLimitError` / HTTP 429 with exponential backoff (`initial_backoff * 2**attempt`, capped at `max_backoff`, plus optional `jitter`). Non-rate-limit errors propagate immediately.
- For `bge-m3` empirically same-topic paragraph distances cluster around `~0.45` and topic shifts around `~0.65`; `DEFAULT_DISTANCE_THRESHOLD = 0.55` is tuned for that gap. Run `scripts/probe_semantic_distances.py` to inspect distances for your own corpus.

See [docs/architecture.md](docs/architecture.md#public-chunker--embedder-api) for the full API contract and calibration guidance.

## Resilience (LM Studio retries)

The `LMStudioClient` wraps every embedding / chat call with retries on `OSError`, `WinError 10054` (Windows socket reset under load), and malformed JSON, using exponential backoff tuned via `LMStudioConfig.retries` and `LMStudioConfig.backoff`. Failed calls are logged and surface to the API as `502` / `503` responses. Tune behaviour via env vars and the wiring in [`src/py_rag_engine/clients/lm_studio.py`](src/py_rag_engine/clients/lm_studio.py) and [`src/py_rag_engine/api/routes.py`](src/py_rag_engine/api/routes.py).

## Documentation

| Document | Contents |
| -------- | -------- |
| [LICENSE](LICENSE) | MIT License |
| [docs/architecture.md](docs/architecture.md) | Module map, pipeline diagram, data flow |
| [docs/evaluation.md](docs/evaluation.md) | RAGAS metrics, run modes, report schema |
| [SECURITY.md](SECURITY.md) | API access, credentials, transport, and deployment boundaries |
| [`pyproject.toml`](pyproject.toml) | Build config, optional extras (`api`, `embeddings`, `eval`), Ruff settings |

## Project layout

| Path | Role |
| ---- | ---- |
| `src/py_rag_engine/config.py` | `LMStudioConfig` / `PostgresConfig` / `EvalConfig` (env-driven) |
| `src/py_rag_engine/clients/lm_studio.py` | `LMStudioClient` HTTP wrapper + `detect_chat_model` |
| `src/py_rag_engine/domain.py` | `DocumentChunk`, `ChunkMetadata` |
| `src/py_rag_engine/vector_math.py` | `cosine_similarity` (numpy) |
| `src/py_rag_engine/ingestion/` | PDF + Markdown loaders, `ingest_file` / `ingest_path` orchestration |
| `src/py_rag_engine/chunking/` | Recursive splitter with dynamic overlap; embedding-based semantic splitting |
| `src/py_rag_engine/chunker.py` | Public `SemanticChunker` + `calibrate_distance_threshold` (async; threshold auto-tuning) |
| `src/py_rag_engine/embeddings/` | SHA-256 hashing, `make_lm_studio_embed`, `make_sentence_transformer_embed` |
| `src/py_rag_engine/embedder.py` | Public `VectorClient` (OpenAI-compatible or SentenceTransformer, async batching + rate-limit backoff) |
| `src/py_rag_engine/storage/postgres.py` | `PostgresEmbeddingStore` (pgvector HNSW + Postgres FTS) |
| `src/py_rag_engine/retrieval/hybrid.py` | `retrieve_hybrid`, `reciprocal_rank_fusion` |
| `src/py_rag_engine/retrieval/rerank.py` | `CrossEncoderReranker`, `rerank_candidates` |
| `src/py_rag_engine/retrieval/service.py` | `retrieve_with_rerank`, `retrieve_hybrid_with_rerank` |
| `src/py_rag_engine/api/` | FastAPI factory + lifespan, access policy, ASGI body gate, REST routes, schemas |
| `src/py_rag_engine/generation/lm_studio_chat.py` | `generate_answer` grounded in context |
| `src/py_rag_engine/evaluation/` | Gold-standard loader, metrics, official-RAGAS adapter, `EvalRunner` |
| `scripts/demo_rerank.py` | E2E demo CLI (dense + Cross-Encoder rerank) |
| `scripts/demo_hybrid.py` | Hybrid Search demo CLI (dense vs FTS vs RRF) |
| `scripts/eval_ragas.py` | Offline RAGAS CLI |
| `scripts/process_document.py` | Standalone chunking CLI |
| `data/` | Sample PDF metadata, eval document, gold-standard Q&A |
| `tests/` | `pytest` suite (chunking, ingestion, storage, retrieval, API, evaluation) |
| `.github/workflows/ci.yml` | CI: pytest matrix on Python 3.11 / 3.12 |

## Tests

```bash
python -m pip install -c constraints/api-ci.txt -e ".[api]" pytest httpx
env -u TEST_POSTGRES_URL -u LM_STUDIO_BASE_URL python -m pytest -q
```

The normal suite uses fake database/model clients and does not need model
downloads. On PowerShell, remove `Env:\TEST_POSTGRES_URL` and
`Env:\LM_STUDIO_BASE_URL` before running `python -m pytest -q`.
`tests/test_postgres_integration.py` is skipped unless both variables are set.

### Integration test (full round-trip)

Use a **disposable database**: this test drops and recreates the `embeddings`
table. Set the `POSTGRES_*` fields for that database, then generate the DSN
through `PostgresConfig` so special characters are escaped correctly. The
captured command below does not print the DSN to the terminal.

```powershell
Remove-Item Env:\EVAL_POSTGRES_URL -ErrorAction SilentlyContinue
$env:TEST_POSTGRES_URL = python -c "from py_rag_engine.config import PostgresConfig; print(PostgresConfig.from_env().url)"
$env:LM_STUDIO_BASE_URL        = "http://localhost:1234"
$env:LM_STUDIO_EMBEDDING_MODEL = "text-embedding-bge-m3"
$env:STORAGE_EMBEDDING_MODEL   = "bge-m3"
python -m pytest -q tests/test_postgres_integration.py
Remove-Item Env:\TEST_POSTGRES_URL
```

### Coverage

```bash
pip install -c constraints/api-ci.txt -e ".[api]" pytest httpx pytest-cov
pytest --cov=py_rag_engine --cov-report=term-missing
```

## Offline RAGAS evaluation

`scripts/eval_ragas.py` runs the full pipeline against a 10-question gold-standard set and writes a JSON report comparing chunk sizes and embedding models. Three metrics are computed per question — **faithfulness**, **answer relevancy**, **context precision** — and an `overall_ranking` block ranks configurations by mean score.

| Mode | Env var | Configs × Questions | Wall time¹ |
|---|---|---|---|
| Smoke (sanity check) | `EVAL_SMOKE=1` | 1 × 1 | ~30 s |
| Quick (single model, 3 Qs) | `EVAL_QUICK=1 EVAL_SKIP_MINILM=1` | 3 × 3 | ~3 min |
| Full single-model | `EVAL_SKIP_MINILM=1` | 3 × 10 | ~8 min |
| Full comparison | _(no flags)_ | 6 × 10 | ~15 min |

¹ Qwen2.5-7B-Instruct Q4_K_M on RTX 3060 12 GB. See [docs/evaluation.md](docs/evaluation.md) for the full metric definitions and report schema.

## Contributing

Issues and pull requests are welcome. Run `pytest -q` and `ruff check src tests` before opening a PR; keep modules single-purpose and prefer extending existing scripts over adding new top-level entry points.

## Changelog

Tracked via GitHub releases and the [commit history](https://github.com/enoquesousa/py-rag-engine/commits).

## License

[MIT](LICENSE).

<div align="center">

## Author

**Enoque Sousa**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/enoque-sousa-bb89aa168/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/enoquesousa)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=flat&logo=target&logoColor=white)](https://enoquesousa.vercel.app)

**[⬆ Back to Top](#py-rag-engine)**

Made with ❤️ by [Enoque Sousa](https://github.com/enoquesousa)

**Project status:** Study project

</div>
