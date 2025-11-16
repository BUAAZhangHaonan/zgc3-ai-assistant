from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List


@dataclass
class DocumentChunk:
    """Single chunk of markdown content."""

    id: str
    content: str
    source: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["metadata"] = data.get("metadata") or {}
        return data


def _iter_markdown_files(data_dir: Path) -> Iterator[Path]:
    for path in sorted(data_dir.glob("**/*.md")):
        if path.is_file():
            yield path


def _chunk_text(content: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    step = chunk_size - overlap
    start = 0
    text = " ".join(line.strip() for line in content.splitlines() if line.strip())
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start += step


def load_markdown_documents(
    data_dir: Path, chunk_size: int = 800, chunk_overlap: int = 120
) -> List[DocumentChunk]:
    """Read markdown files under ``data_dir`` and split into chunks."""
    chunks: List[DocumentChunk] = []
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        return chunks

    for file_path in _iter_markdown_files(data_dir):
        content = file_path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(_chunk_text(content, chunk_size, chunk_overlap)):
            chunk_id = f"{file_path.stem}-{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=chunk,
                    source=str(file_path.relative_to(data_dir)),
                    metadata={"chunk_index": idx},
                )
            )
    return chunks
