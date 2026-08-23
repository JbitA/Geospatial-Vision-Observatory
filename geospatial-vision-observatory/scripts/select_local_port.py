#!/usr/bin/env python3
"""Select an available loopback TCP port for the native Windows API launcher."""

from __future__ import annotations

import argparse
import socket

DEFAULT_FALLBACKS = (8765, 8888, 8000, 8899)


def port_available(host: str, port: int) -> bool:
    if host != "127.0.0.1":
        raise ValueError("native launcher port probing is restricted to 127.0.0.1")
    if not 1024 <= port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind((host, port))
    except OSError:
        return False
    return True


def select_port(preferred: int, *, allow_fallback: bool = True) -> int:
    candidates = (preferred, *DEFAULT_FALLBACKS) if allow_fallback else (preferred,)
    seen: set[int] = set()
    for port in candidates:
        if port in seen:
            continue
        seen.add(port)
        if port_available("127.0.0.1", port):
            return port
    if allow_fallback:
        # Ask Windows for an allowed ephemeral loopback port when common development ports are
        # reserved or excluded by Hyper-V, VPN, security software, or another local process.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            selected = int(probe.getsockname()[1])
        if 1024 <= selected <= 65535:
            return selected
    raise RuntimeError("no usable loopback API port is available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferred", type=int, default=8080)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()
    try:
        selected = select_port(args.preferred, allow_fallback=not args.no_fallback)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(selected)


if __name__ == "__main__":
    main()
