"""Safe file enumeration and serialization helpers shared by command-line tools."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

_NATURAL_PARTS = re.compile(r"(\d+)")


def natural_key(path: str | Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in _NATURAL_PARTS.split(str(path))]


def resolve_in(workdir: Path, path: Path) -> Path:
    """Resolve a user path relative to workdir, never process cwd implicitly."""
    workdir = workdir.resolve()
    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def iter_paths(workdir: Path, pattern: str, recursive: bool = False) -> list[Path]:
    workdir = workdir.resolve()
    paths = workdir.rglob(pattern) if recursive else workdir.glob(pattern)
    return sorted((path.resolve() for path in paths if path.is_file()), key=natural_key)


def parse_triplet(value: str, cast: type = float) -> tuple[Any, Any, Any]:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Expected three values, got {value!r}")
    return tuple(cast(part) for part in parts)  # type: ignore[return-value]


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def preflight_outputs(paths: Iterable[Path], force: bool = False) -> list[Path]:
    """Validate every output before any write; reject duplicates and overwrite."""
    result = [path.resolve() for path in paths]
    if len(result) != len(set(result)):
        raise FileExistsError("Output plan contains duplicate paths")
    existing = [path for path in result if path.exists()]
    if existing and not force:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(str(path) for path in existing))
    for path in result:
        path.parent.mkdir(parents=True, exist_ok=True)
    return result


def atomic_write_bytes(path: Path, data: bytes, force: bool = False, *, preflighted: bool = False) -> None:
    path = path.resolve()
    if not preflighted:
        preflight_outputs([path], force)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, data: str, force: bool = False, *, preflighted: bool = False) -> None:
    atomic_write_bytes(path, data.encode("utf-8"), force, preflighted=preflighted)


def write_json(data: Any, path: Path, force: bool = False) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", force)


def relative_strings(paths: Iterable[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]
