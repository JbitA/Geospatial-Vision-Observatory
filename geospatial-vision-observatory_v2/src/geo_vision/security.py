from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
from urllib.parse import urlsplit

from .config import Settings


class SecurityViolation(ValueError):
    pass


def validate_outbound_url(url: str, settings: Settings) -> str:
    """Block SSRF: HTTPS only, exact host allowlist, no credentials or odd ports."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or host not in settings.host_allowlist:
        raise SecurityViolation("outbound URL is not allowlisted HTTPS")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise SecurityViolation("credentials and non-standard ports are forbidden")
    return url


def reject_private_resolution(host: str) -> None:
    """Defense in depth against DNS rebinding; call immediately before a request."""
    for result in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise SecurityViolation("allowlisted host resolved to a non-global address")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def chain_digest(previous: str, canonical_event: bytes, key: bytes | None = None) -> str:
    message = previous.encode("ascii") + b"\n" + canonical_event
    return (hmac.new(key, message, hashlib.sha256) if key else hashlib.sha256(message)).hexdigest()


def constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)
