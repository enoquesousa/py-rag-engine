"""Centralised, env-driven configuration for the RAG engine.

A single source of truth for connection strings, default model IDs, retry
parameters, and pipeline tunables. All scripts and library entry points read
through `LMStudioConfig` / `PostgresConfig` instead of calling `os.environ`
directly — this keeps env-var names, defaults, and validation in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from ipaddress import IPv6Address

from sqlalchemy.engine import URL


@dataclass(frozen=True, slots=True)
class LMStudioConfig:
    """LM Studio (OpenAI-compatible) server configuration.

    `chat_model` may be left empty; the caller can resolve it at runtime via
    `py_rag_engine.clients.lm_studio.detect_chat_model()`.
    """

    base_url: str = "http://localhost:1234"
    embed_model: str = "text-embedding-bge-m3"
    chat_model: str = ""
    request_timeout: int = 180
    retries: int = 5
    backoff: float = 2.0
    embed_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "LMStudioConfig":
        return cls(
            base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234").rstrip("/"),
            embed_model=os.environ.get("LM_STUDIO_EMBED_MODEL", "text-embedding-bge-m3"),
            chat_model=os.environ.get("LM_STUDIO_CHAT_MODEL", ""),
            request_timeout=int(os.environ.get("LM_STUDIO_TIMEOUT", "180")),
            retries=int(os.environ.get("LM_STUDIO_RETRIES", "5")),
            backoff=float(os.environ.get("LM_STUDIO_BACKOFF", "2.0")),
            embed_batch_size=int(os.environ.get("LM_STUDIO_EMBED_BATCH", "64")),
        )


@dataclass(frozen=True, slots=True)
class PostgresConfig:
    """PostgreSQL + pgvector connection settings.

    Credentials are NEVER hardcoded. Construct from env via `from_env`, which
    accepts either a full DSN (preferred) or the standard `POSTGRES_*` parts.
    """

    # Omit the whole DSN: query parameters can contain secrets too.
    url: str = field(repr=False)

    @classmethod
    def from_env(cls, *, var: str = "EVAL_POSTGRES_URL") -> "PostgresConfig":
        """Resolve a Postgres DSN from the environment.

        Resolution order:
          1. The named DSN env var (`var`, default `EVAL_POSTGRES_URL`).
          2. Standard `POSTGRES_*` parts:
               POSTGRES_USER       (default: postgres)
               POSTGRES_PASSWORD   (REQUIRED — fails fast if missing)
               POSTGRES_HOST       (default: localhost)
               POSTGRES_PORT       (default: 5432)
               POSTGRES_DB         (default: rag)

        Raises:
            RuntimeError when neither path supplies credentials or the separate
            host/port fields cannot safely describe a connection.

        A full DSN is returned unchanged and may use external authentication.
        Separate user, password and database fields are URL-encoded by SQLAlchemy.
        """
        url = os.environ.get(var)
        if url:
            return cls(url=url)

        password = os.environ.get("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError(
                f"Postgres credentials missing. Set {var}=<full-dsn> or "
                "POSTGRES_PASSWORD (plus optional POSTGRES_USER / POSTGRES_HOST / "
                "POSTGRES_PORT / POSTGRES_DB)."
            )
        user = os.environ.get("POSTGRES_USER", "postgres")
        host = _postgres_host(os.environ.get("POSTGRES_HOST", "localhost"))
        port = _postgres_port(os.environ.get("POSTGRES_PORT", "5432"))
        db = os.environ.get("POSTGRES_DB", "rag")
        connection_url = URL.create(
            "postgresql+psycopg",
            username=user,
            password=password,
            host=host,
            port=port,
            database=db,
        )
        return cls(url=connection_url.render_as_string(hide_password=False))


def _postgres_port(value: str) -> int:
    """Validate a separate TCP port without echoing environment contents."""
    message = "POSTGRES_PORT must be an integer from 1 to 65535."
    if not value.isascii() or not value.isdecimal():
        raise RuntimeError(message)
    try:
        port = int(value)
    except ValueError:
        raise RuntimeError(message) from None
    if not 1 <= port <= 65535:
        raise RuntimeError(message)
    return port


def _postgres_host(value: str) -> str:
    """Keep an authority host from introducing other URL components."""
    message = "POSTGRES_HOST must be a hostname or IP address without URL separators."
    bracketed = value.startswith("[") and value.endswith("]")
    if bracketed:
        value = value[1:-1]
    if bracketed or ":" in value:
        try:
            IPv6Address(value)
        except ValueError:
            raise RuntimeError(message) from None
    if any(
        char in "/?#@[]\\" or char.isspace() or ord(char) < 32 or ord(char) == 127
        for char in value
    ):
        raise RuntimeError(message)
    return value


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Offline evaluation tunables."""

    chunk_sizes: tuple[int, ...] = (512, 1024, 2048)
    top_k: int = 5
    ef_search: int = 80
    skip_minilm: bool = False
    quick: bool = False
    smoke: bool = False
    use_official_ragas: bool = False
    questions_path: str = "data/eval_questions.json"
    reports_dir: str = "reports"
    eval_document: str = "data/eval_document.md"

    # Preferred chat models, substring matched against LM Studio /v1/models IDs.
    # Sized for a 12 GB GPU running alongside an embedding model and the
    # cross-encoder reranker — 13B+ tends to trigger WinError 10054 mid-eval.
    preferred_chat_patterns: tuple[str, ...] = field(default_factory=lambda: (
        "qwen2.5-7b",
        "qwen-2-5-7b",
        "llama-3.1-8b",
        "llama-3.2-8b",
        "mistral-7b",
        "gemma-2-9b",
        "phi-3.5-mini",
        "phi-3-mini",
    ))

    @classmethod
    def from_env(cls) -> "EvalConfig":
        return cls(
            skip_minilm=os.environ.get("EVAL_SKIP_MINILM", "0") == "1",
            quick=os.environ.get("EVAL_QUICK", "0") == "1",
            smoke=os.environ.get("EVAL_SMOKE", "0") == "1",
            use_official_ragas=os.environ.get("EVAL_USE_OFFICIAL_RAGAS", "0") == "1",
            questions_path=os.environ.get("EVAL_QUESTIONS", "data/eval_questions.json"),
            reports_dir=os.environ.get("EVAL_REPORTS_DIR", "reports"),
            eval_document=os.environ.get("EVAL_DOCUMENT", "data/eval_document.md"),
        )
