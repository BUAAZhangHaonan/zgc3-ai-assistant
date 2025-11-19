from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.rag.loader import load_markdown_documents
from zgc3_assistant.rag.store import RAGStore

LOGGER = logging.getLogger(__name__)


def build_index(
    settings: Settings,
    data_dir: Path | None = None,
    index_dir: Path | None = None,
) -> RAGStore:
    data_dir = (data_dir or settings.rag_data_dir).resolve()
    index_dir = (index_dir or settings.rag_index_dir).resolve()
    chunks = load_markdown_documents(
        data_dir=data_dir,
        chunk_size=settings.rag_chunk_size
    )
    if not chunks:
        raise RuntimeError(f"No markdown documents found under {data_dir}")

    dashscope_client = DashScopeClient(api_key=settings.dashscope_api_key, settings=settings)
    embeddings = dashscope_client.embed_texts(chunk.content for chunk in chunks)
    LOGGER.info("Embedding %s chunks for RAG store", len(chunks))
    return RAGStore.build_and_save(chunks, embeddings, index_dir)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FAISS index for zgc3 assistant.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with markdown files.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Output directory for FAISS index.")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    settings = get_settings()
    build_index(settings, data_dir=args.data_dir, index_dir=args.index_dir)
    LOGGER.info("RAG index successfully created.")


if __name__ == "__main__":
    main()

