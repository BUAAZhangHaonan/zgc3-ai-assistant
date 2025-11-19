from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List

LOGGER = logging.getLogger(__name__)


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


def semantic_chunker(text: str, chunk_size: int) -> Iterator[str]:
    """
    一个平衡的、以语义为核心的文本分块器。
    它将连续的短段落组合在一起，同时对超长段落按句子进行细分。
    """
    if not text or chunk_size <= 0:
        return

    # 1. 初始按段落分割
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current_chunk_paragraphs: List[str] = []

    for paragraph in paragraphs:
        # --- 情况 A: 处理本身就超长的段落 ---
        if len(paragraph) > chunk_size:
            # 1.1 首先，将当前积累的短段落组合成一个 chunk
            if current_chunk_paragraphs:
                yield "\n\n".join(current_chunk_paragraphs)
                current_chunk_paragraphs = []

            # 1.2 然后，对这个长段落按句子进行细分
            sentences = re.split(r"([。\.!\?\n])", paragraph)
            sentences = ["".join(s).strip()
                         for s in zip(sentences[0::2], sentences[1::2])]
            sentences = [s for s in sentences if s]

            if not sentences:  # 如果没有句子，则直接产出这个长段落
                yield paragraph
                continue

            current_sentence_group: List[str] = []
            for sentence in sentences:
                if len(" ".join(current_sentence_group + [sentence])) > chunk_size and current_sentence_group:
                    yield " ".join(current_sentence_group)
                    current_sentence_group = [sentence]
                else:
                    current_sentence_group.append(sentence)

            if current_sentence_group:
                yield " ".join(current_sentence_group)

        # --- 情况 B: 累积短段落 ---
        else:
            if len("\n\n".join(current_chunk_paragraphs + [paragraph])) > chunk_size and current_chunk_paragraphs:
                yield "\n\n".join(current_chunk_paragraphs)
                current_chunk_paragraphs = [paragraph]
            else:
                current_chunk_paragraphs.append(paragraph)

    # 不要忘记最后一个由短段落组成的 chunk
    if current_chunk_paragraphs:
        yield "\n\n".join(current_chunk_paragraphs)


def load_markdown_documents(
    data_dir: Path, chunk_size: int = 400
) -> List[DocumentChunk]:
    """
    读取 Markdown 文件并使用最终的、平衡的语义分割器将其切分为 Chunks。
    """
    chunks: List[DocumentChunk] = []
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        return chunks

    for file_path in _iter_markdown_files(data_dir):
        content = file_path.read_text(encoding="utf-8").strip()

        for idx, chunk_content in enumerate(
            semantic_chunker(content, chunk_size)
        ):
            chunk_id = f"{file_path.stem}-{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    source=str(file_path.relative_to(data_dir)),
                    metadata={"chunk_index": idx},
                )
            )

    # 保存 chunk 预览文件，方便调试
    output_file = Path.cwd() / "chunks_preview.json"
    try:
        with output_file.open("w", encoding="utf-8") as f:
            chunks_as_dicts = [chunk.to_dict() for chunk in chunks]
            json.dump(chunks_as_dicts, f, ensure_ascii=False, indent=2)
        LOGGER.info(f"Chunk 分割结果已保存到: {output_file.resolve()}")
    except Exception as e:
        LOGGER.error(f"无法保存 chunk 预览文件: {e}")

    return chunks
