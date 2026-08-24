from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


@dataclass(frozen=True)
class ArtifactStat:
    key: str
    bytes: int
    sha256: str


class ArtifactStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> ArtifactStat: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def stat(self, key: str) -> ArtifactStat: ...


def validate_artifact_key(key: str) -> PurePosixPath:
    if not isinstance(key, str) or not key or len(key) > 1024:
        raise ValueError("artifact key must be a non-empty string up to 1024 characters")
    if "\\" in key or "\x00" in key:
        raise ValueError("artifact keys must use portable forward-slash paths")
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact key must be a normalized relative path")
    if ":" in path.parts[0]:
        raise ValueError("artifact key must not contain a Windows drive prefix")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalArtifactStore:
    """Minimal local artifact backend with atomic writes and traversal/symlink protection."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("artifact root must be a real directory, not a symlink")

    def _path(self, key: str, *, create_parent: bool = False) -> Path:
        relative = validate_artifact_key(key)
        current = self.root
        for component in relative.parts[:-1]:
            current = current / component
            if current.exists() and current.is_symlink():
                raise ValueError("artifact path traverses a symlink")
        if create_parent:
            current.mkdir(parents=True, exist_ok=True)
        target = current / relative.name
        if target.exists() and target.is_symlink():
            raise ValueError("artifact target must not be a symlink")
        try:
            target.resolve(strict=False).relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact path escapes the configured root") from error
        return target

    def put_bytes(self, key: str, data: bytes) -> ArtifactStat:
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        target = self._path(key, create_parent=True)
        digest = _sha256(data)
        if target.exists():
            existing = self.get_bytes(key)
            if existing != data:
                raise FileExistsError(f"immutable artifact already exists with different bytes: {key}")
            return ArtifactStat(key=key, bytes=len(data), sha256=digest)

        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # A hard-link create is atomic and refuses to overwrite a concurrently-created
                # immutable key. The temporary file lives in the same directory/filesystem.
                os.link(temp_name, target)
            except FileExistsError:
                existing = self.get_bytes(key)
                if existing != data:
                    raise FileExistsError(
                        f"immutable artifact already exists with different bytes: {key}"
                    ) from None
            finally:
                os.unlink(temp_name)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return ArtifactStat(key=key, bytes=len(data), sha256=digest)

    def get_bytes(self, key: str) -> bytes:
        target = self._path(key)
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(key)
        return target.read_bytes()

    def exists(self, key: str) -> bool:
        try:
            target = self._path(key)
            return target.is_file() and not target.is_symlink()
        except (OSError, ValueError):
            return False

    def stat(self, key: str) -> ArtifactStat:
        data = self.get_bytes(key)
        return ArtifactStat(key=key, bytes=len(data), sha256=_sha256(data))
