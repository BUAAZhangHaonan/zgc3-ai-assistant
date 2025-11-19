from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None
import numpy as np

from .loader import DocumentChunk

LOGGER = logging.getLogger(__name__)


class RAGStore:
    """Simple FAISS-backed store to support school RAG scenarios."""

    INDEX_FILENAME = "index.faiss"
    META_FILENAME = "chunks.json"

    def __init__(self, index: faiss.IndexFlatIP, chunks: List[DocumentChunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def build_and_save(
        cls,
        chunks: Iterable[DocumentChunk],
        embeddings: Iterable[Iterable[float]],
        index_dir: Path,
    ) -> "RAGStore":
        if faiss is None:
            raise RuntimeError("faiss is required to build the index. Install faiss-cpu.")
        chunks = list(chunks)
        vectors = np.array(list(embeddings), dtype="float32")
        if not chunks or vectors.size == 0:
            raise ValueError("No data provided to build RAG index")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and embeddings size mismatch")

        dim = vectors.shape[1]
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        index_dir.mkdir(parents=True, exist_ok=True)

        # 写入时的目录切换修复
        origin_cwd = Path.cwd()
        try:
            os.chdir(index_dir)
            faiss.write_index(index, cls.INDEX_FILENAME)
            with open(cls.META_FILENAME, "w", encoding="utf-8") as fp:
                json.dump([c.to_dict() for c in chunks], fp, ensure_ascii=False, indent=2)
        finally:
            os.chdir(origin_cwd)

        LOGGER.info("Saved RAG index with %s chunks to %s", len(chunks), index_dir)
        return cls(index=index, chunks=chunks)

    @classmethod
    def load(cls, index_dir: Path) -> Optional["RAGStore"]:
        if faiss is None:
            LOGGER.warning("faiss not installed; cannot load RAG index.")
            return None
        
        # 先用 Python 的能力检查文件是否存在，这不会报错
        index_path = index_dir / cls.INDEX_FILENAME
        meta_path = index_dir / cls.META_FILENAME
        if not index_path.exists() or not meta_path.exists():
            LOGGER.warning("RAG index directory %s incomplete", index_dir)
            return None

        # 核心修复：在读取 FAISS 索引前，应用与保存时完全相同的目录切换策略
        origin_cwd = Path.cwd()
        try:
            os.chdir(index_dir)
            
            # 现在，FAISS 和 JSON 加载都使用相对路径，避免了中文路径问题
            index = faiss.read_index(cls.INDEX_FILENAME)
            chunk_dicts = json.loads(Path(cls.META_FILENAME).read_text(encoding="utf-8"))

        finally:
            os.chdir(origin_cwd)
        
        chunks = [DocumentChunk(**item) for item in chunk_dicts]
        return cls(index=index, chunks=chunks)

    def search(self, embedding: Iterable[float], top_k: int = 5) -> List[dict]:
        if faiss is None:
            raise RuntimeError("faiss is required to search the index.")
        if self.index.ntotal == 0:
            return []
        vector = np.array([list(embedding)], dtype="float32")
        faiss.normalize_L2(vector)
        limit = min(top_k, len(self.chunks))
        scores, idxs = self.index.search(vector, limit)
        results: List[dict] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.id,
                    "text": chunk.content,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                    "score": float(score),
                }
            )
        return results