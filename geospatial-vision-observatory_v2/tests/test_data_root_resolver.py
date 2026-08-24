from __future__ import annotations

from pathlib import Path

import scripts.resolve_data_root as resolver


def test_probe_root_accepts_writable_empty_cache(tmp_path: Path) -> None:
    result = resolver.probe_root(tmp_path / "curated")
    assert result.ok is True


def test_select_root_reuses_healthy_primary(tmp_path: Path) -> None:
    primary = tmp_path / "curated"
    selected = resolver.select_root(primary, tmp_path)
    assert selected == primary.resolve()


def test_select_root_rotates_away_from_unreadable_primary(
    tmp_path: Path, monkeypatch
) -> None:
    primary = (tmp_path / "curated").resolve()
    fallback = (tmp_path / "curated-windows").resolve()

    def fake_probe(path: Path) -> resolver.ProbeResult:
        resolved = path.resolve()
        if resolved == primary:
            return resolver.ProbeResult(False, "PermissionError: access denied")
        if resolved == fallback:
            return resolver.ProbeResult(True)
        return resolver.ProbeResult(False, "unexpected")

    monkeypatch.setattr(resolver, "probe_root", fake_probe)
    assert resolver.select_root(primary, tmp_path) == fallback


def test_select_root_uses_next_healthy_fallback(tmp_path: Path, monkeypatch) -> None:
    primary = (tmp_path / "curated").resolve()
    fallback1 = (tmp_path / "curated-windows").resolve()
    fallback2 = (tmp_path / "curated-windows-2").resolve()

    def fake_probe(path: Path) -> resolver.ProbeResult:
        resolved = path.resolve()
        if resolved in {primary, fallback1}:
            return resolver.ProbeResult(False, "denied")
        if resolved == fallback2:
            return resolver.ProbeResult(True)
        return resolver.ProbeResult(False, "unexpected")

    monkeypatch.setattr(resolver, "probe_root", fake_probe)
    assert resolver.select_root(primary, tmp_path) == fallback2
