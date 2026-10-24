"""Authenticate and bound the complete HTTP body before FastAPI parses it."""
from __future__ import annotations

import tempfile
from typing import BinaryIO

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .access import AccessConfig, AccessDenied, authorize_request, single_header

MAX_BODY_BYTES = 10 * 1024 * 1024
_MEMORY_SPOOL_BYTES = 1024 * 1024
_REPLAY_CHUNK_BYTES = 64 * 1024


class APIAccessMiddleware:
    """Apply one shared access policy to every route, with no public bypass paths."""

    def __init__(self, app: ASGIApp, config: AccessConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return
        try:
            authorize_request(scope, self.config)
            _check_declared_length(scope)
            with tempfile.SpooledTemporaryFile(max_size=_MEMORY_SPOOL_BYTES) as body:
                if not await _buffer_body(receive, body):
                    return
                await run_in_threadpool(body.seek, 0)
                await self.app(scope, _replay_body(receive, body), _no_store(send))
        except AccessDenied as error:
            headers = {"Cache-Control": "no-store"}
            if error.status == 401:
                headers["WWW-Authenticate"] = "Bearer"
            response = JSONResponse({"detail": error.detail}, status_code=error.status,
                                    headers=headers)
            await response(scope, receive, send)


def _check_declared_length(scope: Scope) -> None:
    length = single_header(scope.get("headers", []), b"content-length")
    if length is None:
        return
    if not length.isdigit() or len(length) > 20:
        raise AccessDenied(400, "Invalid Content-Length.")
    if int(length) > MAX_BODY_BYTES:
        raise AccessDenied(413, "Request body exceeds the 10 MiB limit.")


async def _buffer_body(receive: Receive, body: BinaryIO) -> bool:
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return False
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise AccessDenied(413, "Request body exceeds the 10 MiB limit.")
        if chunk:
            await run_in_threadpool(body.write, chunk)
        if not message.get("more_body", False):
            return True


def _replay_body(receive: Receive, body: BinaryIO) -> Receive:
    complete = False

    async def replay() -> Message:
        nonlocal complete
        if complete:
            return await receive()
        chunk = await run_in_threadpool(body.read, _REPLAY_CHUNK_BYTES)
        complete = len(chunk) < _REPLAY_CHUNK_BYTES
        return {"type": "http.request", "body": chunk, "more_body": not complete}

    return replay


def _no_store(send: Send) -> Send:
    async def send_response(message: Message) -> None:
        if message["type"] == "http.response.start":
            message = dict(message)
            headers = [(key, value) for key, value in message.get("headers", [])
                       if key.lower() != b"cache-control"]
            message["headers"] = [*headers, (b"cache-control", b"no-store")]
        await send(message)

    return send_response
