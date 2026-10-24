"""Access-boundary regressions with real ASGI middleware and no external services."""
from __future__ import annotations

import asyncio
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from py_rag_engine.api import create_app
from py_rag_engine.api import app as app_module
from py_rag_engine.api import middleware as middleware_module
from py_rag_engine.api.access import AccessConfig
from py_rag_engine.api.middleware import APIAccessMiddleware, MAX_BODY_BYTES

TOKEN = "synthetic-api-token-0123456789abcdef"
OTHER_TOKEN = "different-synthetic-token-0123456789"
AUTH = (b"authorization", b"Bearer " + TOKEN.encode())


@pytest.fixture(autouse=True)
def clean_api_environment(monkeypatch):
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    monkeypatch.delenv("RAG_API_LOCAL_ONLY", raising=False)


def _scope(*, headers=(), client=("198.51.100.10", 50000), scheme="https", path="/query",
           method="POST", query=b""):
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": scheme, "path": path, "raw_path": path.encode(),
        "query_string": query, "root_path": "", "headers": list(headers),
        "client": client, "server": ("127.0.0.1", 8001),
    }


def _invoke(app, scope, *, chunks=(b"",), forbid_receive=False, disconnect=False):
    messages = deque({"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
                     for index, chunk in enumerate(chunks))
    if disconnect:
        messages = deque([{"type": "http.disconnect"}])
    sent = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        if forbid_receive:
            raise AssertionError("The access boundary consumed a rejected request body")
        return messages.popleft() if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    starts = [message for message in sent if message["type"] == "http.response.start"]
    return SimpleNamespace(
        status=starts[0]["status"] if starts else None,
        headers=dict(starts[0].get("headers", [])) if starts else {},
        body=b"".join(message.get("body", b"") for message in sent), reads=reads, sent=sent,
    )


async def _ok(scope, receive, send):
    await PlainTextResponse("ok")(scope, receive, send)


def _guard(*, local=False, app=_ok):
    env = {"RAG_API_LOCAL_ONLY": "1"} if local else {"RAG_API_TOKEN": TOKEN}
    return APIAccessMiddleware(app, AccessConfig.from_env(env))


@pytest.mark.parametrize("env", [
    {}, {"RAG_API_LOCAL_ONLY": "0"}, {"RAG_API_LOCAL_ONLY": "true"},
    {"RAG_API_LOCAL_ONLY": ""}, {"RAG_API_TOKEN": ""},
    {"RAG_API_TOKEN": "", "RAG_API_LOCAL_ONLY": "1"},
    {"RAG_API_TOKEN": "short"}, {"RAG_API_TOKEN": "x" * 4097},
    {"RAG_API_TOKEN": "x" * 31 + "\n"}, {"RAG_API_TOKEN": "é" * 32},
    {"RAG_API_TOKEN": "=" * 32}, {"RAG_API_TOKEN": TOKEN, "RAG_API_LOCAL_ONLY": "yes"},
])
def test_invalid_access_configuration_fails_before_services(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    service = MagicMock(side_effect=AssertionError("External services must not initialize"))
    monkeypatch.setattr(app_module.LMStudioConfig, "from_env", service)
    monkeypatch.setattr(app_module.PostgresConfig, "from_env", service)
    monkeypatch.setattr(app_module, "create_engine", service)
    monkeypatch.setattr(app_module, "CrossEncoderReranker", service)
    with pytest.raises(RuntimeError):
        create_app()
    service.assert_not_called()


def test_config_does_not_reveal_token_and_token_wins_over_local_flag(monkeypatch):
    config = AccessConfig.from_env({"RAG_API_TOKEN": TOKEN, "RAG_API_LOCAL_ONLY": "1"})
    assert TOKEN not in repr(config)
    assert config.local_only is False
    monkeypatch.setenv("RAG_API_TOKEN", TOKEN)
    monkeypatch.setenv("RAG_API_LOCAL_ONLY", "1")
    result = _invoke(create_app(), _scope(client=("127.0.0.1", 1234)), forbid_receive=True)
    assert result.status == 401


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/health"), ("GET", "/documents"), ("POST", "/documents"),
    ("POST", "/query"), ("GET", "/docs"), ("GET", "/redoc"),
    ("GET", "/openapi.json"), ("GET", "/docs/oauth2-redirect"),
    ("GET", "/documents/"), ("OPTIONS", "/documents"),
    ("HEAD", "/health"), ("GET", "/unknown"),
])
def test_every_route_authenticates_before_body_or_dependencies(monkeypatch, method, path):
    monkeypatch.setenv("RAG_API_TOKEN", TOKEN)
    # No lifespan or app.state services: a route/dependency invocation would fail the test.
    result = _invoke(create_app(), _scope(method=method, path=path), forbid_receive=True)
    assert result.status == 401
    assert result.headers[b"www-authenticate"] == b"Bearer"
    assert result.headers[b"cache-control"] == b"no-store"
    assert result.reads == 0


@pytest.mark.parametrize("headers", [
    [], [(b"authorization", b"")], [(b"authorization", b"Basic " + TOKEN.encode())],
    [(b"authorization", b"Bearer")], [(b"authorization", b"Bearer  " + TOKEN.encode())],
    [(b"authorization", b"Bearer " + OTHER_TOKEN.encode())],
    [(b"authorization", b"Bearer " + TOKEN.encode() + b"\n")],
    [(b"cookie", b"access_token=" + TOKEN.encode())],
    [(b"x-api-key", TOKEN.encode())],
])
def test_only_valid_bearer_header_authenticates(headers):
    result = _invoke(_guard(), _scope(headers=headers), forbid_receive=True)
    assert result.status == 401
    assert TOKEN.encode() not in result.body
    assert OTHER_TOKEN.encode() not in result.body


@pytest.mark.parametrize("headers", [[AUTH, AUTH], [AUTH, (b"Authorization", b"Bearer wrong")]])
def test_duplicate_authorization_is_rejected_before_body(headers):
    result = _invoke(_guard(), _scope(headers=headers), forbid_receive=True)
    assert result.status == 400


@pytest.mark.parametrize("query", [
    b"token=", b"TOKEN=secret", b"t%6fken=secret", b"token=&token=secret",
    b"access_token=secret", b"%61uthorization=secret", b"api_key=secret",
    b"rag_api_token=secret",
])
@pytest.mark.parametrize("local", [False, True])
def test_url_credentials_are_rejected_even_when_empty_encoded_or_duplicated(query, local):
    headers = [(b"host", b"localhost:8001")] + ([] if local else [AUTH])
    result = _invoke(_guard(local=local), _scope(headers=headers, query=query,
                     client=("127.0.0.1", 50000), scheme="http"), forbid_receive=True)
    assert result.status == 400


@pytest.mark.parametrize(("scheme", "peer", "expected"), [
    ("https", ("198.51.100.10", 50000), 200),
    ("http", ("198.51.100.10", 50000), 403),
    ("http", ("127.0.0.1", 50000), 200),
    ("http", ("::1", 50000), 200),
    ("http", ("::ffff:127.0.0.1", 50000), 200),
    ("http", ("localhost", 50000), 403), ("http", None, 403),
])
def test_bearer_transport_requires_tls_or_literal_loopback(scheme, peer, expected):
    result = _invoke(_guard(), _scope(scheme=scheme, client=peer, headers=[AUTH]),
                     forbid_receive=expected != 200)
    assert result.status == expected
    assert result.headers[b"cache-control"] == b"no-store"


@pytest.mark.parametrize("name", [b"forwarded", b"x-forwarded-for", b"X-Forwarded-Proto",
                                 b"x-forwarded-host", b"x-forwarded-port"])
def test_forwarding_headers_never_raise_connection_trust(name):
    result = _invoke(_guard(), _scope(scheme="http", headers=[AUTH, (name, b"https")]),
                     forbid_receive=True)
    assert result.status == 400


@pytest.mark.parametrize("local", [False, True])
def test_uvicorn_proxy_header_rewriting_cannot_bypass_the_guard(local):
    guarded = ProxyHeadersMiddleware(_guard(local=local), trusted_hosts="*")
    headers = [(b"host", b"localhost"), (b"x-forwarded-for", b"127.0.0.1"),
               (b"x-forwarded-proto", b"https")] + ([] if local else [AUTH])
    result = _invoke(guarded, _scope(scheme="http", headers=headers), forbid_receive=True)
    assert result.status == 400


@pytest.mark.parametrize(("host", "origin", "site"), [
    (b"localhost:8001", None, None), (b"127.0.0.1:8001", None, b"none"),
    (b"[::1]:8001", b"http://[::1]:8001", b"same-origin"),
    (b"localhost", b"http://localhost:80", b"same-origin"),
])
def test_explicit_local_mode_accepts_only_local_request_authorities(host, origin, site):
    headers = [(b"host", host)]
    if origin is not None:
        headers.append((b"origin", origin))
    if site is not None:
        headers.append((b"sec-fetch-site", site))
    result = _invoke(_guard(local=True), _scope(scheme="http", client=("127.0.0.1", 1234),
                     headers=headers))
    assert result.status == 200


@pytest.mark.parametrize("headers", [
    [], [(b"host", b"evil.example")], [(b"host", b"localhost.evil.example")],
    [(b"host", b"0.0.0.0")], [(b"host", b"localhost:")],
    [(b"host", b"[::1]evil")], [(b"host", b"[::1%lo]")],
    [(b"host", b"localhost"), (b"host", b"localhost")],
    [(b"host", b"localhost"), (b"origin", b"null")],
    [(b"host", b"localhost"), (b"origin", b"https://localhost")],
    [(b"host", b"localhost"), (b"origin", b"http://localhost:8001")],
    [(b"host", b"localhost"), (b"origin", b"http://evil.example")],
    [(b"host", b"localhost"), (b"origin", b"http://localhost"),
     (b"origin", b"http://localhost")],
    [(b"host", b"localhost"), (b"sec-fetch-site", b"cross-site")],
    [(b"host", b"localhost"), (b"sec-fetch-site", b"same-site")],
    [(b"host", b"localhost"), (b"sec-fetch-site", b"unknown")],
    [(b"host", b"localhost"), (b"sec-fetch-site", b"none"),
     (b"sec-fetch-site", b"none")],
    [(b"host", b"localhost"), AUTH],
])
def test_local_mode_rejects_unsafe_or_ambiguous_host_origin_and_fetch_site(headers):
    result = _invoke(_guard(local=True), _scope(scheme="http", client=("127.0.0.1", 1234),
                     headers=headers), forbid_receive=True)
    assert result.status in {400, 403}


@pytest.mark.parametrize("peer", [("198.51.100.10", 1234), ("localhost", 1234), None,
                                 ("::1%lo", 1234)])
def test_local_mode_requires_the_real_literal_loopback_peer(peer):
    result = _invoke(_guard(local=True), _scope(client=peer, headers=[(b"host", b"localhost")]),
                     forbid_receive=True)
    assert result.status == 403


def test_declared_body_limit_is_enforced_before_multipart_or_receive(monkeypatch):
    monkeypatch.setenv("RAG_API_TOKEN", TOKEN)
    headers = [AUTH, (b"content-length", str(MAX_BODY_BYTES + 1).encode()),
               (b"content-type", b"multipart/form-data; boundary=example")]
    result = _invoke(create_app(), _scope(path="/documents", headers=headers), forbid_receive=True)
    assert result.status == 413
    assert result.headers[b"cache-control"] == b"no-store"


@pytest.mark.parametrize("headers", [[(b"content-length", b"-1")],
                                    [(b"content-length", b"not-an-integer")],
                                    [(b"content-length", b"0"), (b"content-length", b"0")]])
def test_ambiguous_or_invalid_content_length_is_rejected(headers):
    result = _invoke(_guard(), _scope(headers=[AUTH, *headers]), forbid_receive=True)
    assert result.status == 400


def test_authentication_precedes_even_the_declared_body_limit():
    scope = _scope(headers=[(b"content-length", str(MAX_BODY_BYTES + 1).encode())])
    result = _invoke(_guard(), scope, forbid_receive=True)
    assert result.status == 401


def _track_spools(monkeypatch):
    opened = []
    original = tempfile.SpooledTemporaryFile

    def create(**kwargs):
        body = original(**kwargs)
        opened.append(body)
        return body

    monkeypatch.setattr(middleware_module.tempfile, "SpooledTemporaryFile", create)
    return opened


def test_chunked_body_limit_stops_before_parser_and_closes_buffer(monkeypatch):
    monkeypatch.setattr(middleware_module, "MAX_BODY_BYTES", 16)
    opened = _track_spools(monkeypatch)

    async def forbidden_handler(scope, receive, send):
        raise AssertionError("Oversized multipart must not reach a handler or parser")

    result = _invoke(_guard(app=forbidden_handler), _scope(headers=[AUTH]),
                     chunks=(b"a" * 8, b"b" * 8, b"c", b"must not be read"))
    assert result.status == 413
    assert result.reads == 3
    assert len(opened) == 1 and opened[0].closed


def test_body_replay_preserves_bytes_and_closes_buffer(monkeypatch):
    monkeypatch.setattr(middleware_module, "_REPLAY_CHUNK_BYTES", 4)
    opened = _track_spools(monkeypatch)

    async def echo(scope, receive, send):
        data = bytearray()
        while True:
            message = await receive()
            data.extend(message["body"])
            if not message.get("more_body", False):
                break
        await PlainTextResponse(bytes(data))(scope, receive, send)

    result = _invoke(_guard(app=echo), _scope(headers=[AUTH]), chunks=(b"123", b"45678"))
    assert result.status == 200 and result.body == b"12345678"
    assert len(opened) == 1 and opened[0].closed


def test_disconnected_upload_does_not_reach_handler_and_closes_buffer(monkeypatch):
    opened = _track_spools(monkeypatch)

    async def forbidden_handler(scope, receive, send):
        raise AssertionError("Disconnected uploads must not reach a handler")

    result = _invoke(_guard(app=forbidden_handler), _scope(headers=[AUTH]), disconnect=True)
    assert result.sent == []
    assert len(opened) == 1 and opened[0].closed


def test_handler_failure_still_closes_the_buffer(monkeypatch):
    opened = _track_spools(monkeypatch)

    async def broken_handler(scope, receive, send):
        raise RuntimeError("synthetic handler failure")

    with pytest.raises(RuntimeError, match="synthetic handler failure"):
        _invoke(_guard(app=broken_handler), _scope(headers=[AUTH]), chunks=(b"data",))
    assert len(opened) == 1 and opened[0].closed


def test_factory_snapshots_credentials_and_preserves_lifespan(monkeypatch):
    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("startup")
        yield
        events.append("shutdown")

    monkeypatch.setattr(app_module, "_lifespan", lifespan)
    monkeypatch.setenv("RAG_API_TOKEN", TOKEN)
    app = create_app()
    monkeypatch.setenv("RAG_API_TOKEN", OTHER_TOKEN)
    with TestClient(app, base_url="https://testserver") as client:
        assert events == ["startup"]
        assert client.get("/openapi.json").status_code == 401
        result = client.get("/openapi.json", headers={"Authorization": f"Bearer {TOKEN}"})
        assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
        assert client.get("/docs", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        wrong = client.get("/openapi.json", headers={"Authorization": f"Bearer {OTHER_TOKEN}"})
        assert wrong.status_code == 401
    assert events == ["startup", "shutdown"]


@pytest.mark.parametrize("failure", ["startup", "request", None])
def test_real_lifespan_disposes_engine_after_startup_or_request_failure(monkeypatch, failure):
    engine = MagicMock()
    store = MagicMock()
    monkeypatch.setattr(app_module.LMStudioConfig, "from_env",
                        lambda: SimpleNamespace(embed_model="text-embedding-bge-m3"))
    monkeypatch.setattr(app_module.PostgresConfig, "from_env",
                        lambda: SimpleNamespace(url="postgresql+psycopg://synthetic"))
    monkeypatch.setattr(app_module, "create_engine", lambda url: engine)
    monkeypatch.setattr(app_module, "PostgresEmbeddingStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(app_module, "LMStudioClient", lambda config: object())
    monkeypatch.setattr(app_module, "make_lm_studio_embed", lambda client: object())
    monkeypatch.setattr(app_module, "CrossEncoderReranker", lambda: object())
    if failure == "startup":
        store.create_schema.side_effect = RuntimeError("synthetic startup failure")

    async def run():
        async with app_module._lifespan(FastAPI()):
            if failure == "request":
                raise RuntimeError("synthetic request failure")

    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    engine.dispose.assert_called_once_with()
