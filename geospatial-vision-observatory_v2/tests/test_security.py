from pathlib import Path

import pytest
from pydantic import SecretStr

from geo_vision.config import Settings
from geo_vision.security import SecurityViolation, chain_digest, validate_outbound_url


def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, db_path=tmp_path / "test.db")


def test_only_exact_https_hosts_are_allowed(tmp_path: Path) -> None:
    config = settings(tmp_path)
    assert validate_outbound_url("https://earth-search.aws.element84.com/v1/collections", config)
    for unsafe in (
        "http://earth-search.aws.element84.com/v1/collections",
        "https://earth-search.aws.element84.com.evil.example/x",
        "https://user:pass@earth-search.aws.element84.com/x",
        "https://earth-search.aws.element84.com:8443/x",
        "file:///etc/passwd",
    ):
        with pytest.raises(SecurityViolation):
            validate_outbound_url(unsafe, config)


def test_hmac_chain_changes_with_key() -> None:
    event = b'{"type":"test"}'
    plain = chain_digest("0" * 64, event)
    signed = chain_digest("0" * 64, event, b"secret")
    assert len(plain) == len(signed) == 64
    assert plain != signed


def test_secret_is_redacted(tmp_path: Path) -> None:
    config = Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "test.db",
        audit_hmac_key=SecretStr("do-not-print"),
    )
    assert "do-not-print" not in repr(config)


def test_native_runtime_defaults_to_loopback_only(tmp_path: Path) -> None:
    config = Settings(data_dir=tmp_path, db_path=tmp_path / "test.db")
    assert config.bind_host == "127.0.0.1"
    assert config.bind_port == 8080
