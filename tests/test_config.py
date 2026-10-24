from __future__ import annotations

import traceback

import pytest
from sqlalchemy.engine import make_url

from py_rag_engine.config import EvalConfig, LMStudioConfig, PostgresConfig


def test_lm_studio_defaults():
    cfg = LMStudioConfig()
    assert cfg.base_url == "http://localhost:1234"
    assert cfg.embed_model == "text-embedding-bge-m3"
    assert cfg.chat_model == ""
    assert cfg.embed_batch_size == 64
    assert cfg.retries == 5


def test_lm_studio_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://example:9999/")
    monkeypatch.setenv("LM_STUDIO_CHAT_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("LM_STUDIO_EMBED_BATCH", "16")
    cfg = LMStudioConfig.from_env()
    assert cfg.base_url == "http://example:9999"           # trailing slash stripped
    assert cfg.chat_model == "qwen2.5-7b-instruct"
    assert cfg.embed_batch_size == 16


def test_postgres_from_full_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_POSTGRES_URL", "postgresql+psycopg://u:p@h:5/db")
    assert PostgresConfig.from_env().url == "postgresql+psycopg://u:p@h:5/db"


def test_postgres_assembles_from_parts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    monkeypatch.setenv("POSTGRES_USER", "rag")
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "ragdb")
    cfg = PostgresConfig.from_env()
    assert cfg.url == "postgresql+psycopg://rag:s3cret@db.local:6543/ragdb"


def test_postgres_uses_part_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    for var in ("POSTGRES_USER", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    cfg = PostgresConfig.from_env()
    assert cfg.url == "postgresql+psycopg://postgres:p@localhost:5432/rag"


def test_postgres_requires_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="credentials missing"):
        PostgresConfig.from_env()


def test_postgres_named_var_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("DEMO_POSTGRES_URL", "postgresql+psycopg://a:b@c:1/d")
    assert PostgresConfig.from_env(var="DEMO_POSTGRES_URL").url == "postgresql+psycopg://a:b@c:1/d"


def test_postgres_construct_with_explicit_url():
    """Direct construction with a URL still works (for tests / lib use)."""
    cfg = PostgresConfig(url="postgresql+psycopg://x:y@z:1/q")
    assert cfg.url == "postgresql+psycopg://x:y@z:1/q"


@pytest.mark.parametrize("password", ["p@:/?#%", "literal%40%2F", "senha 日本語 🗝️ + espaço"])
def test_postgres_password_round_trip(monkeypatch: pytest.MonkeyPatch, password: str):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    monkeypatch.setenv("POSTGRES_USER", "rag")
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "ragdb")

    cfg = PostgresConfig.from_env()
    parsed = make_url(cfg.url)
    assert isinstance(cfg.url, str)
    assert parsed.password == password
    assert parsed.username == "rag"
    assert parsed.host == "db.local"
    assert parsed.port == 5432
    assert parsed.database == "ragdb"
    assert not parsed.query


def test_postgres_user_and_database_round_trip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    username = "usuário@:/?#% + 日本語"
    database = "dados@:/?#% + 日本語?sslmode=disable&host=other.example"
    monkeypatch.setenv("POSTGRES_USER", username)
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-private-password")
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", database)

    parsed = make_url(PostgresConfig.from_env().url)
    assert parsed.username == username
    assert parsed.password == "synthetic-private-password"
    assert parsed.database == database
    assert parsed.host == "db.local"
    assert parsed.port == 5432
    assert not parsed.query


@pytest.mark.parametrize("host", ["2001:db8::1", "[2001:db8::1]", "::1", "[::1]"])
def test_postgres_ipv6_round_trip(monkeypatch: pytest.MonkeyPatch, host: str):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@:/?#%")
    monkeypatch.setenv("POSTGRES_HOST", host)
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    parsed = make_url(PostgresConfig.from_env().url)
    assert parsed.host == host.strip("[]")
    assert parsed.port == 6543
    assert parsed.password == "p@:/?#%"


@pytest.mark.parametrize("port", [
    "", "0", "65536", "-1", "+5432", "5432.0", " 5432", "٥٤٣٢", "secret-port-marker",
])
def test_postgres_rejects_invalid_port_without_echo(monkeypatch: pytest.MonkeyPatch, port: str):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-private-password")
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", port)
    with pytest.raises(RuntimeError, match="POSTGRES_PORT must be an integer") as failure:
        PostgresConfig.from_env()
    diagnostic = "".join(traceback.format_exception(failure.value))
    assert "synthetic-private-password" not in diagnostic
    assert "secret-port-marker" not in diagnostic


@pytest.mark.parametrize("host", [
    "db.local/other", "db.local?token=host-secret-marker", "db.local#fragment",
    "u@db.local", "db.local:5432", "[::1", "bad\nhost",
])
def test_postgres_rejects_host_url_injection(monkeypatch: pytest.MonkeyPatch, host: str):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-private-password")
    monkeypatch.setenv("POSTGRES_HOST", host)
    with pytest.raises(RuntimeError, match="POSTGRES_HOST must be") as failure:
        PostgresConfig.from_env()
    diagnostic = "".join(traceback.format_exception(failure.value))
    assert "synthetic-private-password" not in diagnostic
    assert "host-secret-marker" not in diagnostic


def test_postgres_full_url_is_authoritative_without_password(monkeypatch: pytest.MonkeyPatch):
    full = (
        "postgresql+psycopg://external-user@localhost/rag"
        "?sslmode=verify-full&passfile=/private/pass"
    )
    monkeypatch.setenv("EVAL_POSTGRES_URL", full)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.setenv("POSTGRES_PORT", "ignored-invalid-port")
    assert PostgresConfig.from_env().url == full


def test_postgres_custom_url_wins_over_default_and_parts(monkeypatch: pytest.MonkeyPatch):
    full = "postgresql+psycopg:///rag?host=%2Fvar%2Frun%2Fpostgresql&sslmode=disable"
    monkeypatch.setenv("DEMO_POSTGRES_URL", full)
    monkeypatch.setenv("EVAL_POSTGRES_URL", "postgresql+psycopg://other:secret@other/db")
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.setenv("POSTGRES_PORT", "ignored-invalid-port")
    assert PostgresConfig.from_env(var="DEMO_POSTGRES_URL").url == full


def test_postgres_custom_var_absent_uses_parts_not_default_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEMO_POSTGRES_URL", raising=False)
    monkeypatch.setenv("EVAL_POSTGRES_URL", "postgresql+psycopg://other:secret@other/db")
    monkeypatch.setenv("POSTGRES_PASSWORD", "parts-password")
    monkeypatch.setenv("POSTGRES_HOST", "parts.local")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    parsed = make_url(PostgresConfig.from_env(var="DEMO_POSTGRES_URL").url)
    assert parsed.password == "parts-password"
    assert parsed.host == "parts.local"


@pytest.mark.parametrize("full", [
    "postgresql+psycopg://private-user:private-password@private-host/private-db",
    "postgresql+psycopg://localhost/rag?password=query-secret&token=another-secret",
])
def test_postgres_repr_omits_entire_dsn(full: str):
    cfg = PostgresConfig(url=full)
    assert cfg.url == full
    assert repr(cfg) == "PostgresConfig()"
    assert str(cfg) == "PostgresConfig()"


def test_eval_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_SMOKE", "1")
    monkeypatch.setenv("EVAL_USE_OFFICIAL_RAGAS", "1")
    cfg = EvalConfig.from_env()
    assert cfg.smoke is True
    assert cfg.use_official_ragas is True
    assert cfg.preferred_chat_patterns[0] == "qwen2.5-7b"
