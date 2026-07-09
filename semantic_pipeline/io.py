"""Serialization helpers.

Every layer writes its output to disk so the next layer can be developed, run,
and inspected independently. JSONL (one object per line) is the on-disk form of
the immutable object streams; it is append-friendly and streams without loading
the whole file into memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Type, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Write an iterable of pydantic models to JSONL, streaming. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path, model: Type[M]) -> Iterator[M]:
    """Read a JSONL file back into pydantic models, streaming."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def write_json(path: str | Path, obj: dict) -> None:
    """Write a single JSON document (used for run manifests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
