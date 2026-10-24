# Security model

## API access

Use `py_rag_engine.api:create_app`. The factory validates access configuration
before opening database/model resources. A missing mode prevents startup.

| Mode | Configuration | Boundary |
| --- | --- | --- |
| Shared Bearer | `RAG_API_TOKEN`, 32–4096 ASCII Bearer-token characters | Every route needs the header; HTTPS or an actual loopback peer is required |
| Local development | Token absent and `RAG_API_LOCAL_ONLY=1` | Actual loopback peer, loopback/localhost Host, matching Origin when supplied, and allowed fetch metadata |

A configured token always takes precedence. An empty configured token is an
error. Local mode deliberately trusts clients on the same machine and rejects
an Authorization header. Bind it to loopback. The host/origin checks reduce
browser access from other origins; they do not isolate mutually untrusted local
users or compromised local software. Do not publish local-only mode through a
proxy: a same-host proxy is itself a trusted loopback peer. Use token mode for
any proxied remote access.

One shared token grants every API operation on the entire corpus, including
document ingestion and retrieval. Metadata filters and model selections are
not authorization. There are no accounts, tenant ACLs, JWTs, refresh tokens,
per-user revocation, or sessions. Health, docs, and OpenAPI require the same
access policy. A normal browser navigation cannot attach a Bearer header.

Only the Authorization header carries API credentials. Duplicate headers and
credential query parameters are rejected, including encoded names and empty values.
The access configuration keeps the validated token in an internal fixed-size
buffer, including its length, and omits that buffer from its representation.
Each valid incoming token is normalized the same way and compared with
`hmac.compare_digest`, using equal-sized inputs without content-dependent
short-circuiting. This is direct comparison of a configured service credential;
the padding is not encryption or password hashing. The token remains readable
in the process environment and configuration memory. Do not serialize or dump
the access configuration, and protect process dumps and debug interfaces.
Use a secret manager or a restricted environment configuration, never commit
real `.env` files or place the secret in a URL. Keep request headers out of
proxy/application access logs. The policy's error messages do not echo values.

Replace the token and restart every worker to rotate it. The old credential
snapshot remains valid in a running worker until that worker is stopped; there is no refresh
endpoint or automatic credential reload.

## Transport and proxies

Use TLS with a certificate trusted by clients for any remote connection. Keep
Uvicorn's `--no-proxy-headers` option. Requests containing `Forwarded` or any
`X-Forwarded-*` header are rejected. A TLS proxy on the same host may forward to
a loopback backend after stripping those headers. Do not expose that backend
port to other hosts or rely on forwarding headers to authenticate transport.
The README includes direct TLS and local commands.

Authentication runs before FastAPI reads multipart data. The total HTTP body
limit is 10 MiB, including multipart framing, with a 1 MiB memory spool threshold.
The temporary spool is closed when processing ends. This limit does not bound
PDF expansion, parsing time, concurrent requests, embedding/model cost, stored
corpus size, or generated output. Apply connection/time/concurrency limits and
process resource limits appropriate to your workload. Authorized inputs remain
untrusted data; neither the parser nor the LLM is a sandbox.

## Database, model service, and local tools

PostgreSQL credentials are independent of the API token. The token is not sent
to PostgreSQL or LM Studio. A supplied full database DSN is used as-is; its
transport and authentication settings are the operator's responsibility.
Separate `POSTGRES_*` fields are assembled with SQLAlchemy's escaping, and the
complete DSN is hidden from the configuration representation and demo headers.
The URL string still contains credentials for the database driver; do not log
or serialize it. Use a dedicated database account with the schema permissions
the store needs. Demo Docker ports are published only on loopback. Docker
administrators can inspect environment-based container credentials.

LM Studio defaults to a local HTTP endpoint. Keep it on loopback or a trusted,
protected backend network. The built-in `LMStudioClient` does not configure a
provider API key; it sends document chunks and questions to the configured
service. Do not put passwords or tokens in `LM_STUDIO_BASE_URL`, which appears
in diagnostics and evaluation reports. The `api_key="lm-studio"` example for
an injected OpenAI-compatible SDK client is a local placeholder, not an access
control. External authenticated providers need their own HTTPS configuration
and separate credentials; never reuse the RAG service token.

CLI scripts and library calls bypass the HTTP layer by design and run with the
caller's filesystem/database permissions. Evaluation recreates its dedicated
tables; the optional PostgreSQL integration test drops the `embeddings` table.
Use disposable databases for those tests. Reports can contain private source
text and answers. Protect database files, backups, uploads, and reports according
to the sensitivity of that content.

## Validation scope

The normal CI suite uses fake model/database clients. Access-policy tests cover
startup failure, header ambiguity, actual-peer and browser-origin restrictions,
proxy metadata, authentication before parsing, body limits, and spool cleanup.
Database configuration tests cover special-character round trips and secret
omission. Script tests use synthetic credentials and stop before external I/O.

CI audits the installed API/test dependency set and builds the package on
Python 3.11/3.12. Optional embeddings, reranking, and evaluation environments are
not part of that dependency constraint set or vulnerability gate. Review their
resolved packages and model artifacts separately when deploying those extras.
