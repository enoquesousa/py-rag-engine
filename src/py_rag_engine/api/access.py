"""API-only access policy; importing the library does not require API credentials."""
from __future__ import annotations

import hmac
import ipaddress
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import SplitResult, parse_qsl, urlsplit

from starlette.types import Scope

_BEARER_TOKEN = re.compile(rb"[A-Za-z0-9._~+/-]+=*")
_MAX_TOKEN_BYTES = 4096
_TOKEN_BUFFER_BYTES = 2 + _MAX_TOKEN_BYTES
_CREDENTIAL_PARAMETERS = {"token", "access_token", "api_key", "authorization", "rag_api_token"}
Headers = Sequence[tuple[bytes, bytes]]


@dataclass(frozen=True, slots=True)
class AccessConfig:
    """Startup snapshot with a plaintext credential buffer excluded from repr."""

    token_buffer: bytes | None = field(repr=False)
    local_only: bool

    def __post_init__(self) -> None:
        if self.token_buffer is not None:
            if self.local_only is not False or not _valid_token_buffer(self.token_buffer):
                raise RuntimeError("Token authentication requires a valid credential buffer.")
        elif self.local_only is not True:
            raise RuntimeError("An API access mode must be explicitly configured.")

    def matches_token(self, token: bytes) -> bool:
        """Compare 4098-byte buffers without branching on the expected token's length."""
        if self.token_buffer is None or not _valid_token(token):
            return False
        return hmac.compare_digest(_token_buffer(token), self.token_buffer)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AccessConfig:
        """Require a Bearer token or an explicit, restricted local-only opt-in."""
        env = os.environ if environ is None else environ
        local = env.get("RAG_API_LOCAL_ONLY", "0")
        if local not in {"0", "1"}:
            raise RuntimeError("RAG_API_LOCAL_ONLY must be 0 or 1.")
        token = env.get("RAG_API_TOKEN")
        if token is not None:
            encoded = token.encode("ascii", errors="replace")
            if not _valid_token(encoded):
                raise RuntimeError(
                    "RAG_API_TOKEN must contain 32 to 4096 ASCII Bearer-token characters."
                )
            return cls(token_buffer=_token_buffer(encoded), local_only=False)
        if local != "1":
            raise RuntimeError("Set RAG_API_TOKEN or explicitly opt in with RAG_API_LOCAL_ONLY=1.")
        return cls(token_buffer=None, local_only=True)


def _valid_token(token: bytes) -> bool:
    return (isinstance(token, bytes) and 32 <= len(token) <= _MAX_TOKEN_BYTES
            and _BEARER_TOKEN.fullmatch(token) is not None)


def _token_buffer(token: bytes) -> bytes:
    """Encode an already-validated token injectively, without truncating its bytes."""
    return len(token).to_bytes(2, "big") + token.ljust(_MAX_TOKEN_BYTES, b"\x00")


def _valid_token_buffer(buffer: bytes) -> bool:
    if not isinstance(buffer, bytes) or len(buffer) != _TOKEN_BUFFER_BYTES:
        return False
    size = int.from_bytes(buffer[:2], "big")
    if not 32 <= size <= _MAX_TOKEN_BYTES:
        return False
    token = buffer[2:2 + size]
    return _valid_token(token) and not any(buffer[2 + size:])


class AccessDenied(Exception):
    """A fixed public error that never contains request credentials or configuration values."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def single_header(headers: Headers, name: bytes) -> bytes | None:
    """Read an unambiguous header, rejecting duplicates instead of merging credentials."""
    values = [value for key, value in headers if key.lower() == name]
    if len(values) > 1:
        raise AccessDenied(400, "Duplicate request header.")
    return values[0] if values else None


def authorize_request(scope: Scope, config: AccessConfig) -> None:
    """Authorize the connection and headers without receiving any request body."""
    headers = scope.get("headers", [])
    if any(key.lower() == b"forwarded" or key.lower().startswith(b"x-forwarded-")
           for key, _ in headers):
        raise AccessDenied(400, "Forwarding headers are not accepted.")
    _reject_url_credentials(scope.get("query_string", b""))
    if config.token_buffer is not None:
        _authorize_bearer(scope, headers, config)
    else:
        _authorize_local(scope, headers)


def _reject_url_credentials(query: bytes) -> None:
    try:
        pairs = parse_qsl(
            query.decode("ascii"), keep_blank_values=True, errors="strict", max_num_fields=128
        )
    except (UnicodeError, ValueError):
        raise AccessDenied(400, "Invalid query string.") from None
    if any(key.lower() in _CREDENTIAL_PARAMETERS for key, _ in pairs):
        raise AccessDenied(400, "Credentials are accepted only in the Authorization header.")


def _authorize_bearer(scope: Scope, headers: Headers, config: AccessConfig) -> None:
    authorization = single_header(headers, b"authorization")
    if authorization is None:
        raise AccessDenied(401, "Bearer authentication required.")
    scheme, separator, token = authorization.partition(b" ")
    if scheme.lower() != b"bearer" or not separator or not _valid_token(token):
        raise AccessDenied(401, "Invalid Bearer credentials.")
    if scope.get("scheme") != "https" and not _loopback_peer(scope):
        raise AccessDenied(403, "HTTPS or a loopback connection is required.")
    if not config.matches_token(token):
        raise AccessDenied(401, "Invalid Bearer credentials.")


def _loopback_address(host: str) -> bool:
    if "%" in host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def _loopback_peer(scope: Scope) -> bool:
    peer = scope.get("client")
    return peer is not None and _loopback_address(peer[0])


def _authorize_local(scope: Scope, headers: Headers) -> None:
    if not _loopback_peer(scope):
        raise AccessDenied(403, "Local-only access requires a loopback connection.")
    if single_header(headers, b"authorization") is not None:
        raise AccessDenied(400, "Bearer credentials require token-configured mode.")
    site = single_header(headers, b"sec-fetch-site")
    if site is not None and site not in {b"same-origin", b"none"}:
        raise AccessDenied(403, "Cross-origin requests are not permitted in local-only mode.")
    scheme = scope.get("scheme", "http")
    host = single_header(headers, b"host")
    if scheme not in {"http", "https"} or host is None:
        raise AccessDenied(403, "A local request authority is required.")
    authority = _local_authority(scheme.encode("ascii") + b"://" + host)
    origin = single_header(headers, b"origin")
    if origin is not None and _local_authority(origin) != authority:
        raise AccessDenied(403, "Cross-origin requests are not permitted in local-only mode.")


def _local_authority(value: bytes) -> tuple[str, str, int]:
    try:
        parsed = _parse_authority_url(value)
        host = parsed.hostname or ""
        if host != "localhost" and not _loopback_address(host):
            raise ValueError
        if not _valid_authority_suffix(parsed.netloc, host):
            raise ValueError
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        return parsed.scheme, host, port
    except (ValueError, UnicodeError):
        raise AccessDenied(403, "Only a loopback or localhost authority is permitted.") from None


def _parse_authority_url(value: bytes) -> SplitResult:
    if any(character <= 32 or character >= 127 for character in value):
        raise ValueError
    parsed = urlsplit(value.decode("ascii"))
    if parsed.scheme not in {"http", "https"} or parsed.username is not None:
        raise ValueError
    if parsed.path or parsed.query or parsed.fragment or b"?" in value or b"#" in value:
        raise ValueError
    return parsed


def _valid_authority_suffix(authority: str, host: str) -> bool:
    expected_host = f"[{host}]" if ":" in host else host
    if not authority.lower().startswith(expected_host):
        return False
    suffix = authority[len(expected_host):]
    return not suffix or (suffix.startswith(":") and suffix[1:].isascii() and suffix[1:].isdigit())
