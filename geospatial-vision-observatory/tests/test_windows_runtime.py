from __future__ import annotations

import pytest

import scripts.select_local_port as ports


def test_port_selection_prefers_requested_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports, "port_available", lambda host, port: port == 8080)
    assert ports.select_port(8080) == 8080


def test_port_selection_falls_back_without_reusing_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[int] = []

    def available(host: str, port: int) -> bool:
        assert host == "127.0.0.1"
        attempted.append(port)
        return port == 8765

    monkeypatch.setattr(ports, "port_available", available)
    assert ports.select_port(8080) == 8765
    assert attempted == [8080, 8765]


def test_explicit_port_selection_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports, "port_available", lambda host, port: False)
    with pytest.raises(RuntimeError, match="no usable loopback API port"):
        ports.select_port(8080, allow_fallback=False)


def test_port_probe_rejects_non_loopback_and_invalid_ports() -> None:
    with pytest.raises(ValueError, match="restricted to 127.0.0.1"):
        ports.port_available("0.0.0.0", 8765)  # noqa: S104 - intentional negative test
    with pytest.raises(ValueError, match="between 1024 and 65535"):
        ports.port_available("127.0.0.1", 80)


def test_port_selection_uses_os_ephemeral_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports, "port_available", lambda host, port: False)

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 49152)

    monkeypatch.setattr(ports.socket, "socket", lambda *args, **kwargs: FakeSocket())
    assert ports.select_port(8080) == 49152
