"""Exercise CLI diagnostics without contacting databases or model servers."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from py_rag_engine.config import EvalConfig, LMStudioConfig, PostgresConfig


_SECRET = "synthetic-database-secret-marker"
_TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


class _StopBeforeExternalIO(RuntimeError):
    pass


def _stop(*args, **kwargs):
    raise _StopBeforeExternalIO


@pytest.fixture
def load_script(tmp_path, monkeypatch):
    # The evaluation script's import bootstrap must not access the network.
    cache = tmp_path / "tokenizer-cache"
    cache.mkdir()
    name = hashlib.sha1(_TOKENIZER_URL.encode(), usedforsecurity=False).hexdigest()
    (cache / name).write_bytes(b"synthetic cache fixture")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    def load(name):
        path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"credential_probe_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def private_dsn(monkeypatch):
    # Cover secrets in both userinfo and arbitrary connection query options.
    dsn = f"postgresql+psycopg://user:{_SECRET}@localhost/rag?sslpassword={_SECRET}"
    monkeypatch.setenv("EVAL_POSTGRES_URL", dsn)
    monkeypatch.setenv("DEMO_POSTGRES_URL", dsn)
    return dsn


def test_evaluation_header_does_not_print_database_credentials(load_script, private_dsn, capsys):
    module = load_script("eval_ragas")
    module._print_header(
        LMStudioConfig(), PostgresConfig(url=private_dsn), EvalConfig(), Path("sample.md")
    )
    output = capsys.readouterr()
    assert "PostgreSQL" in output.out
    assert _SECRET not in output.out + output.err


def test_hybrid_demo_does_not_print_database_credentials(
    load_script, private_dsn, monkeypatch, capsys
):
    module = load_script("demo_hybrid")
    monkeypatch.setattr(module, "LMStudioClient", lambda *a, **kw: SimpleNamespace(embed=_stop))
    monkeypatch.setattr(module, "create_engine", _stop)
    with pytest.raises(_StopBeforeExternalIO):
        module.main()
    output = capsys.readouterr()
    assert "postgres" in output.out
    assert _SECRET not in output.out + output.err


def test_rerank_demo_does_not_print_database_credentials(
    load_script, private_dsn, monkeypatch, capsys
):
    module = load_script("demo_rerank")
    monkeypatch.setattr(sys, "argv", ["demo_rerank.py", "sample.md"])
    chunk = SimpleNamespace(
        text="sample", content_hash="a" * 64, metadata=SimpleNamespace(page=1)
    )
    monkeypatch.setattr(module, "ingest_file", lambda *a, **kw: [chunk])
    client = SimpleNamespace(embed=lambda texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(module, "LMStudioClient", lambda *a, **kw: client)
    monkeypatch.setattr(module, "create_engine", _stop)
    with pytest.raises(_StopBeforeExternalIO):
        module.main()
    output = capsys.readouterr()
    assert "postgres" in output.out
    assert _SECRET not in output.out + output.err


def test_async_demo_does_not_print_database_credentials(
    load_script, private_dsn, monkeypatch, capsys
):
    module = load_script("demo_async_search_rerank")
    monkeypatch.setattr(
        module, "LMStudioClient", lambda *a, **kw: SimpleNamespace(models=lambda: [])
    )
    assert module._conn_url() == private_dsn
    assert module.main() == 2  # Stops before ingestion because no model is loaded.
    output = capsys.readouterr()
    assert "Postgres" in output.out
    assert _SECRET not in output.out + output.err


def test_async_demo_requires_explicit_database_credentials(load_script, monkeypatch):
    module = load_script("demo_async_search_rerank")
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="Postgres credentials missing"):
        module._conn_url()


def test_tokenizer_bootstrap_keeps_tls_verification(load_script, monkeypatch, tmp_path):
    module = load_script("eval_ragas")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "new-cache"))
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    module._ensure_tiktoken_cache()
    assert len(calls) == 1
    args, options = calls[0]
    assert "-fsSL" in args and _TOKENIZER_URL in args
    assert not {"-k", "--insecure", "--ssl-no-revoke"}.intersection(args)
    assert options["check"] is True
