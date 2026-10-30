# Offline RAGAS Evaluation

`scripts/eval_ragas.py` benchmarks the full RAG pipeline against a fixed
10-question gold-standard set across multiple (embedding model, chunk size)
configurations and writes a timestamped JSON report.

## Quick start

```powershell
# 1. Bring up infrastructure
docker start rag-pgvector

# 2. Make sure LM Studio is serving:
#      - text-embedding-bge-m3
#      - qwen2.5-7b-instruct
#    (enable JIT loading so both stay resident)

# 3. Install eval extras
pip install -e ".[eval]"

# 4. Smoke (30 s) → quick (3 min) → full (8-15 min)
$env:PYTHONPATH = "src"
$env:EVAL_SMOKE = "1"; python scripts\eval_ragas.py; Remove-Item Env:\EVAL_SMOKE
```

## Configurations matrix

| chunk_size \ embedder | bge-m3 (LM Studio, 1024d) | all-MiniLM-L6-v2 (ST, 384d) |
|---|---|---|
| 512 | ✔ | ✔ |
| 1024 | ✔ | ✔ |
| 2048 | ✔ | ✔ |

Six configurations total. Each runs against the same 10 gold-standard
questions, isolated in its own PostgreSQL table named
`eval_<model>_<chunk>` (e.g. `eval_bge_m3_512`).

## Metrics

All three follow the original RAGAS paper. Scores are in `[0, 1]`.

### Faithfulness
Fraction of atomic factual statements in the generated answer that can be
inferred from the retrieved context. `1.0` = every claim is supported,
`0.0` = the model fabricated everything. This is the most critical metric
for production RAG.

### Answer Relevancy
Cosine similarity between the user's original question and a **reverse-
generated** question derived from the answer by the LLM judge. High score
means the answer directly addresses the question; low score means it drifted
into an unrelated topic.

### Context Precision
Fraction of retrieved chunks the LLM judge considers useful for answering
the question. High = retriever found mostly relevant chunks; low = retrieval
brought in noise that could mislead the generator.

## Run modes

| Mode | Env vars | Configs × Questions | Wall time¹ |
|---|---|---|---|
| Smoke (sanity check) | `EVAL_SMOKE=1` | 1 × 1 | ~30 s |
| Quick (single model, 3 Qs) | `EVAL_QUICK=1 EVAL_SKIP_MINILM=1` | 3 × 3 | ~3 min |
| Full single-model | `EVAL_SKIP_MINILM=1` | 3 × 10 | ~8 min |
| Full comparison | _(no flags)_ | 6 × 10 | ~15 min |

¹ Qwen2.5-7B-Instruct Q4_K_M on RTX 3060 12 GB.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234` | LM Studio server |
| `LM_STUDIO_EMBED_MODEL` | `text-embedding-bge-m3` | Embedding model ID |
| `LM_STUDIO_CHAT_MODEL` | _(auto-detect)_ | Override the auto-pick |
| `LM_STUDIO_RETRIES` | `5` | HTTP retries per request |
| `LM_STUDIO_BACKOFF` | `2.0` | Exponential-backoff base |
| `LM_STUDIO_EMBED_BATCH` | `64` | Batch size for embeddings |
| `LM_STUDIO_TIMEOUT` | `180` | Per-request seconds |
| `EVAL_POSTGRES_URL` | _(required, or set parts below)_ | Full Postgres DSN |
| `POSTGRES_PASSWORD` | _(required if URL unset)_ | DB password — never put in code |
| `POSTGRES_USER` | `postgres` | DB user |
| `POSTGRES_HOST` | `localhost` | DB host |
| `POSTGRES_PORT` | `5432` | DB port |
| `POSTGRES_DB` | `rag` | DB name |
| `EVAL_DOCUMENT` | `data/eval_document.md` | Source document |
| `EVAL_QUESTIONS` | `data/eval_questions.json` | Gold-standard JSON |
| `EVAL_REPORTS_DIR` | `reports` | Output directory |
| `EVAL_SKIP_MINILM` | `0` | `1` = skip all-MiniLM-L6-v2 |
| `EVAL_QUICK` | `0` | `1` = first 3 questions per config |
| `EVAL_SMOKE` | `0` | `1` = 1 q × 1 chunk × 1 model |
| `EVAL_USE_OFFICIAL_RAGAS` | `0` | `1` = try official ragas with fallback |

The evaluation CLI uses the database and model connections directly; it does
not call the HTTP API and does not require `RAG_API_TOKEN`. A full DSN takes
precedence over the separate database fields. Prefer those fields when entering
a password so `PostgresConfig` can escape special characters correctly. The
configuration representation and script header omit the entire DSN, including
its query parameters. Keep credentials out of `LM_STUDIO_BASE_URL`: that URL is
part of diagnostic and evaluation reports.

Use a dedicated evaluation database. The runner drops and recreates its
per-configuration tables. Reports contain source questions, generated answers,
and retrieved content, so protect them with the same access controls as the
input documents. Tokenizer bootstrap downloads retain normal TLS certificate
verification; they do not disable certificate revocation checks.

## Chat model auto-detection

The script picks a chat model from LM Studio's `/v1/models` in this order:

1. Substring match against `PREFERRED_CHAT_PATTERNS` — 7-8B sweet spot:
   `qwen2.5-7b`, `qwen-2-5-7b`, `llama-3.1-8b`, `llama-3.2-8b`,
   `mistral-7b`, `gemma-2-9b`, `phi-3.5-mini`, `phi-3-mini`.
2. Any non-embedding model with 6-9B parameters, smallest first.
3. Smallest non-embedding model with `>= 4B` parameters.
4. First non-embedding model.
5. First model of any kind.

Override entirely with `LM_STUDIO_CHAT_MODEL=<id>`.

On a 12 GB GPU (e.g. RTX 3060), models in the 13B+ range tend to trigger
`WinError 10054` mid-eval because LM Studio can't keep them resident
alongside the embedding model and the cross-encoder reranker.

## Gold-standard schema

`data/eval_questions.json`:

```json
{
  "schema_version": 1,
  "description": "...",
  "source_document": "data/eval_document.md",
  "questions": [
    {"id": "rag-definition", "question": "What is RAG?", "ground_truth": "..."},
    ...
  ]
}
```

Constraints validated by `evaluation.dataset.load_gold_standard`:
- `schema_version` ∈ `{1}`.
- At least one question.
- Each question has `id`, `question`, `ground_truth`.
- All `id` values are unique.

## Report schema

```json
{
  "evaluation_date": "2026-05-17T15:59:08.123456+00:00",
  "document": "data/eval_document.md",
  "questions_path": "data/eval_questions.json",
  "num_questions": 10,
  "lm_studio_base_url": "http://localhost:1234",
  "lm_studio_embed_model": "text-embedding-bge-m3",
  "lm_studio_chat_model": "qwen2.5-7b-instruct",
  "configurations": [
    {
      "id": "bge-m3_chunk512",
      "embedding_model": "bge-m3",
      "embedding_label": "BGE-M3 via LM Studio (1024d, multilingual)",
      "chunk_size": 512,
      "num_chunks": 35,
      "embed_time_sec": 12.34,
      "metrics": {
        "faithfulness": 0.82,
        "answer_relevancy": 0.79,
        "context_precision": 0.85,
        "mean_retrieval_similarity": 0.61
      },
      "per_question": [{"question": "...", "answer": "...", ...}, ...]
    }
  ],
  "summary": {
    "best_faithfulness": {"config": "...", "score": 0.92},
    "best_answer_relevancy": {"config": "...", "score": 0.85},
    "best_context_precision": {"config": "...", "score": 0.88},
    "best_retrieval_similarity": {"config": "...", "score": 0.70},
    "overall_ranking": [
      {"rank": 1, "config": "...", "avg_score": 0.85},
      ...
    ]
  },
  "errors": []
}
```

## Manual vs official RAGAS

The default path implements the three metrics directly via HTTP calls to LM
Studio, following the paper's definitions. This avoids two known fragilities
of the `ragas` + `langchain-openai` stack against LM Studio:

- `langchain_openai.ChatOpenAI` sends `n>1` for answer relevancy by default;
  LM Studio rejects this. The official path sets `strictness=1` to compensate.
- Some metrics pass non-string tool inputs to the LLM, which several local
  GGUF models truncate or refuse.

Set `EVAL_USE_OFFICIAL_RAGAS=1` to try the official library first. The script
catches all exceptions and falls back automatically to the manual
implementation.
