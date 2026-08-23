from datetime import UTC, datetime, timedelta
from pathlib import Path

from geo_vision.config import Settings
from geo_vision.storage import Store


def make_store(tmp_path: Path) -> Store:
    return Store(
        Settings(
            data_dir=tmp_path,
            db_path=tmp_path / "test.db",
            min_request_spacing_seconds=15,
            hourly_request_budget=2,
            daily_request_budget=3,
        )
    )


def test_request_spacing_and_budget_include_attempts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    now = datetime.now(UTC)
    first = store.acquire_request_budget("earth-search.aws.element84.com", "stac", now)
    assert first
    assert (
        store.acquire_request_budget(
            "earth-search.aws.element84.com", "stac", now + timedelta(seconds=14)
        )
        is None
    )
    second = store.acquire_request_budget(
        "earth-search.aws.element84.com", "stac", now + timedelta(seconds=15)
    )
    assert second and second != first
    assert (
        store.acquire_request_budget(
            "earth-search.aws.element84.com", "stac", now + timedelta(seconds=30)
        )
        is None
    )
    store.finish_request(first, "200")


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.audit("first", {"value": 1})
    store.audit("second", {"value": 2})
    assert store.verify_audit_chain()
    with store.connect() as conn:
        conn.execute("UPDATE audit_log SET event_json='{}' WHERE id=1")
    assert not store.verify_audit_chain()


def test_request_log_pruning(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    old = datetime.now(UTC) - timedelta(days=40)
    assert store.acquire_request_budget("earth-search.aws.element84.com", "stac", old)
    assert store.prune_request_log(datetime.now(UTC)) == 1
