r"""Teste funcional do Hybrid Search (Dense + FTS fundidos via RRF).

Demonstra na prática por que o Hybrid Search melhora a recuperação de termos
técnicos e códigos de erro: ingere um pequeno corpus com chunks artificiais
contendo códigos específicos e compara três modos de busca para a mesma query.

Pré-requisitos:
    - LM Studio com `text-embedding-bge-m3` carregado em http://localhost:1234
    - Postgres + pgvector acessível (porta 5434, db `rag`, user `postgres`)

Execução (PowerShell):
    $env:LM_STUDIO_BASE_URL    = "http://localhost:1234"
    $env:LM_STUDIO_EMBED_MODEL = "text-embedding-bge-m3"
    $env:POSTGRES_PASSWORD    = "<your-password>"
    $env:POSTGRES_PORT        = "5434"
    .venv\Scripts\python.exe scripts\demo_hybrid.py
"""
from __future__ import annotations

import hashlib
import http.client
import json
import time
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import create_engine, text

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig, PostgresConfig
from py_rag_engine.retrieval import retrieve_hybrid
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore

EMBED_MODEL_NAME = "bge-m3"  # storage discriminator (1024 dims)
TABLE_NAME = "embeddings_hybrid_demo"


CORPUS: list[tuple[str, str]] = [
    # (id_humano, texto)
    ("doc-01",
     "Para configurar autenticação OAuth2 em uma API REST, registre o cliente "
     "no provedor de identidade e defina o redirect URI. Use grant_type "
     "authorization_code para apps web e PKCE para mobile."),
    ("doc-02",
     "Logs estruturados em JSON facilitam consulta em ferramentas como "
     "Elasticsearch, Loki ou Datadog. Cada evento deve conter timestamp, "
     "level, service e trace_id para correlação distribuída."),
    ("doc-03",
     "O erro TS2304 do TypeScript indica que o compilador não encontrou um "
     "símbolo referenciado. Geralmente acontece por import faltando, "
     "definição @types ausente ou erro de digitação no nome da variável."),
    ("doc-04",
     "Padrões de cache em sistemas distribuídos incluem cache-aside, "
     "read-through, write-through e write-behind. Cada um tem trade-offs "
     "diferentes de consistência, latência e tolerância a falhas."),
    ("doc-05",
     "Quando uma aplicação Node.js falha com ECONNREFUSED, o socket TCP "
     "tentou abrir conexão mas o destino recusou. Verifique se o serviço "
     "está rodando, a porta está correta e firewall/Docker network estão ok."),
    ("doc-06",
     "Modelagem dimensional em data warehouse usa o esquema estrela ou "
     "snowflake. Tabelas fato registram métricas; dimensões adicionam contexto "
     "como tempo, produto e cliente."),
    ("doc-07",
     "Migrations de schema em Postgres devem ser idempotentes e reversíveis. "
     "Para adicionar coluna NOT NULL em tabela grande, faça em três passos: "
     "adicione nullable, backfill em batches, depois aplique a constraint."),
    ("doc-08",
     "O código HTTP 429 Too Many Requests indica que o cliente excedeu o "
     "rate limit. A resposta deve incluir o header Retry-After informando "
     "quantos segundos esperar antes de tentar novamente."),
    ("doc-09",
     "Em arquitetura hexagonal (ports & adapters) o domínio fica isolado "
     "da infraestrutura. Adapters traduzem chamadas externas (HTTP, fila, "
     "banco) para ports definidos pelo núcleo de negócio."),
    ("doc-10",
     "Garbage collection da JVM pode pausar a aplicação. G1GC é o coletor "
     "padrão moderno; ZGC e Shenandoah oferecem pausas sub-milissegundo "
     "trocando throughput por previsibilidade de latência."),
]


def _hash(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def _print_header(title: str) -> None:
    print()
    print("=" * 90)
    print(f" {title}")
    print("=" * 90)


def _print_dense(rows: list, label: str) -> None:
    print(f"\n[{label}]  candidatos={len(rows)}")
    print(f"  {'rank':>4} {'cos_sim':>8}  doc_id  preview")
    print("  " + "-" * 84)
    for rank, item in enumerate(rows, start=1):
        prev = item.text.replace("\n", " ")[:60]
        doc_id = item.metadata.get("doc_id", "?")
        print(f"  {rank:>4} {item.cosine_similarity:>8.4f}  {doc_id:<6}  {prev}")


def _print_fts(rows: list, label: str) -> None:
    print(f"\n[{label}]  matches={len(rows)}")
    print(f"  {'rank':>4} {'fts_score':>9}  doc_id  preview")
    print("  " + "-" * 84)
    for rank, item in enumerate(rows, start=1):
        prev = item.text.replace("\n", " ")[:60]
        doc_id = item.metadata.get("doc_id", "?")
        print(f"  {rank:>4} {item.fts_score:>9.4f}  {doc_id:<6}  {prev}")


def _print_hybrid(rows: list, label: str) -> None:
    print(f"\n[{label}]  resultados fundidos={len(rows)}")
    print(f"  {'rank':>4} {'rrf':>7} {'cos':>6} {'fts':>6}  doc_id  preview")
    print("  " + "-" * 84)
    for rank, item in enumerate(rows, start=1):
        prev = item.text.replace("\n", " ")[:60]
        doc_id = item.metadata.get("doc_id", "?")
        print(f"  {rank:>4} {item.rrf_score:>7.4f} "
              f"{item.cosine_similarity:>6.3f} {item.fts_score:>6.3f}  "
              f"{doc_id:<6}  {prev}")


def _http_json_via_httpclient(url: str, payload: dict | None = None, *, timeout: int | None = None) -> Any:
    """Transporte HTTP via http.client com retry (evita uma incompatibilidade
    do urllib.request com OpenSSL neste build do Python 3.14 no Windows; e
    cobre os WinError 10054 que o LM Studio gera em conexões longas)."""
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise ValueError(f"Esperado http://, got {parsed.scheme!r}")
    retries = 5
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout or 120)
        try:
            if payload is None:
                conn.request("GET", parsed.path)
            else:
                body = json.dumps(payload).encode("utf-8")
                conn.request(
                    "POST", parsed.path, body=body,
                    headers={"Content-Type": "application/json", "Connection": "close"},
                )
            resp = conn.getresponse()
            data = resp.read()
            return json.loads(data.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2.0 ** attempt
                print(f"   [retry {attempt}/{retries-1}] {type(exc).__name__}: {exc} — wait {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
        finally:
            conn.close()
    if last_exc is not None:
        raise last_exc


def main() -> None:
    base_cfg = LMStudioConfig.from_env()
    # Forçar batch=1 para evitar 10054 do LM Studio em payloads grandes
    lm_cfg = LMStudioConfig(
        base_url=base_cfg.base_url,
        embed_model=base_cfg.embed_model,
        chat_model=base_cfg.chat_model,
        request_timeout=180,
        retries=5,
        backoff=2.0,
        embed_batch_size=1,
    )
    client = LMStudioClient(lm_cfg, http_json=_http_json_via_httpclient)
    postgres_url = PostgresConfig.from_env(var="DEMO_POSTGRES_URL").url

    _print_header("1) Setup")
    print(f"  lm_studio_url    = {lm_cfg.base_url}")
    print(f"  embed_model      = {lm_cfg.embed_model}")
    print("  postgres         = configured")
    print(f"  storage_model    = {EMBED_MODEL_NAME}")
    print(f"  table            = {TABLE_NAME}")
    print(f"  corpus_size      = {len(CORPUS)} documentos")

    # Embed PRIMEIRO (urllib + ssl carregam OpenSSL antes da libpq).
    _print_header("2) Embedding via LM Studio")
    t0 = time.perf_counter()
    vectors = client.embed([doc_text for _, doc_text in CORPUS])
    print(f"  embedded         = {len(vectors)} chunks em {time.perf_counter()-t0:.2f}s")
    print(f"  vector_dim       = {len(vectors[0])}")

    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))

    store = PostgresEmbeddingStore(
        engine, embedding_model=EMBED_MODEL_NAME, table_name=TABLE_NAME,
    )
    store.create_schema()

    _print_header("3) Persistência")
    ids = store.add_embeddings([
        EmbeddingInput(
            text=doc_text,
            embedding=vector,
            content_hash=_hash(doc_text),
            metadata={"doc_id": doc_id, "source": "demo_hybrid"},
            embedding_model=EMBED_MODEL_NAME,
        )
        for (doc_id, doc_text), vector in zip(CORPUS, vectors, strict=True)
    ])
    print(f"  inserted         = {len(ids)} rows")

    # Validação: confirmar que o tsvector foi materializado
    with engine.begin() as conn:
        tsv_count = conn.execute(text(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE text_search_tsv IS NOT NULL"
        )).scalar()
    print(f"  text_search_tsv  = {tsv_count} rows com tsvector materializado")

    # ── Queries ─────────────────────────────────────────────────────────────
    # Notes:
    #   - websearch_to_tsquery une termos com AND, então queries longas em
    #     português com stopwords ("como", "resolver", "fazer") restringem
    #     demais o FTS. Mostramos os dois cenários abaixo para tornar o ganho
    #     do Hybrid evidente: query natural longa vs. query "técnica" curta.
    queries = [
        ("Como resolver erro TS2304?", "doc-03"),
        ("TS2304", "doc-03"),
        ("ECONNREFUSED em Node.js, o que fazer?", "doc-05"),
        ("ECONNREFUSED", "doc-05"),
        ("HTTP 429 retry-after", "doc-08"),
        ("backfill NOT NULL column batches", "doc-07"),
    ]

    for query, expected_doc in queries:
        _print_header(f"QUERY: {query!r}   (esperado: {expected_doc})")

        # Dense puro
        query_vec = client.embed([query])[0]
        dense = store.similarity_search(query_vec, top_k=5, ef_search=80)
        _print_dense(dense, "Vetorial puro (dense top-5)")

        # FTS puro
        fts = store.fts_search(query, top_k=5)
        _print_fts(fts, "FTS puro (top-5)")

        # Hybrid
        hybrid = retrieve_hybrid(
            query_vec, query, store,
            dense_k=10, fts_k=10, top_k=5, rrf_k=60, ef_search=80,
        )
        _print_hybrid(hybrid, "Hybrid (RRF: dense + FTS)")

        # Veredito
        def first_doc(rows, attr_to_get_doc_id):
            return rows[0].metadata.get("doc_id") if rows else None

        dense_top = first_doc(dense, "doc_id")
        fts_top = first_doc(fts, "doc_id")
        hybrid_top = first_doc(hybrid, "doc_id")

        def check(label, got):
            ok = "OK " if got == expected_doc else "MISS"
            print(f"  -> {label:<10} top1 = {got!s:<8}  [{ok}]")

        print()
        check("dense", dense_top)
        check("fts", fts_top)
        check("hybrid", hybrid_top)

    _print_header("FIM")


if __name__ == "__main__":
    main()
