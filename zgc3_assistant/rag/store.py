from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import faiss
except ImportError:
    faiss = None
import numpy as np

from zgc3_assistant.rag.chunker import Document

LOGGER = logging.getLogger(__name__)

class RAGStore:
    INDEX_FILENAME = "index.faiss"
    DOCS_FILENAME = "documents.json"

    def __init__(self, index: faiss.Index, documents: List[Document]):
        self.index = index
        self.documents = documents

    @classmethod
    def build_and_save(
        cls,
        documents: Iterable[Document],
        embeddings: np.ndarray,
        index_dir: Path,
    ) -> None:
        if faiss is None:
            raise RuntimeError("faiss is required. Please run 'pip install faiss-cpu'.")
        
        documents = list(documents)
        if embeddings.dtype != "float32":
            embeddings = embeddings.astype("float32")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        LOGGER.info("FAISS index created with %s documents.", index.ntotal)

        index_dir.mkdir(parents=True, exist_ok=True)
        origin_cwd = Path.cwd()
        try:
            os.chdir(index_dir)
            faiss.write_index(index, cls.INDEX_FILENAME)
            with open(cls.DOCS_FILENAME, "w", encoding="utf-8") as fp:
                json.dump([doc.to_dict() for doc in documents], fp, ensure_ascii=False, indent=2)
        finally:
            os.chdir(origin_cwd)
        LOGGER.info("Saved RAG store with %s documents to %s", len(documents), index_dir)

    @classmethod
    def load(cls, index_dir: Path) -> RAGStore:
        if faiss is None:
            raise RuntimeError("faiss is required. Please run 'pip install faiss-cpu'.")
        
        index_path = index_dir / cls.INDEX_FILENAME
        docs_path = index_dir / cls.DOCS_FILENAME

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"RAG store not found or incomplete in {index_dir}")

        LOGGER.info("Loading RAG store from %s", index_dir)
        
        origin_cwd = Path.cwd()
        try:
            os.chdir(index_dir)
            index = faiss.read_index(cls.INDEX_FILENAME)
            with open(cls.DOCS_FILENAME, "r", encoding="utf-8") as fp:
                docs_data = json.load(fp)
        finally:
            os.chdir(origin_cwd)
        
        documents = [Document(**item) for item in docs_data]
        LOGGER.info("Successfully loaded RAG store with %d documents.", len(documents))
        return cls(index=index, documents=documents)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Document]:
        if self.index.ntotal == 0:
            return []
            
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        if query_embedding.dtype != "float32":
            query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)
        
        results: List[Document] = []
        for i in indices[0]:
            if i != -1:
                results.append(self.documents[i])
        return results