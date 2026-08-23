from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from geo_vision import worker
from geo_vision.config import Settings
from geo_vision.source import BudgetExhausted, CircuitOpen


class FakeStore:
    def prune_request_log(self, now: object) -> None:
        self.pruned = True


class FakePipeline:
    def __init__(self, outcome: object = None) -> None:
        self.outcome = outcome

    async def ingest_once(self) -> object:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, db_path=tmp_path / "worker.db")


@pytest.mark.parametrize(
    "outcome",
    [None, BudgetExhausted("budget"), CircuitOpen("circuit"), RuntimeError("unexpected")],
)
def test_worker_once_handles_expected_and_unexpected_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: object
) -> None:
    store = FakeStore()
    pipeline = FakePipeline(outcome)
    monkeypatch.setattr(worker, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(worker, "Store", lambda settings: store)
    monkeypatch.setattr(worker, "Sentinel2StacClient", lambda settings, supplied_store: object())
    monkeypatch.setattr(worker, "Pipeline", lambda settings, supplied_store, source: pipeline)
    asyncio.run(worker.loop(once=True))
    assert store.pruned is True
