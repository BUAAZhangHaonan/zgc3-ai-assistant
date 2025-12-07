from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.rag.chunker import load_and_chunk_documents
from zgc3_assistant.rag.store import RAGStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
LOGGER = logging.getLogger(__name__)

def build_index(settings: Settings, data_dir: Path | None = None, index_dir: Path | None = None):
    data_dir = (data_dir or settings.rag_data_dir).resolve()
    index_dir = (index_dir or settings.rag_index_dir).resolve()

    documents, _ = load_and_chunk_documents(
        data_dir=data_dir,
        index_dir=index_dir,
        chunk_size=settings.rag_chunk_size,
    )
    if not documents:
        raise RuntimeError(f"未从 {data_dir} 下的文档中生成任何有效的文档单元。")

    LOGGER.info("开始调用 DashScope API 为 %d 个文档单元生成向量...", len(documents))
    dashscope_client = DashScopeClient(api_key=settings.dashscope_api_key, settings=settings)
    
    embeddings_list = dashscope_client.embed_texts(doc.content for doc in documents)
    if len(embeddings_list) != len(documents):
        raise RuntimeError(f"向量生成数量 ({len(embeddings_list)}) 与文档数量 ({len(documents)}) 不匹配。")
    
    embeddings_array = np.array(embeddings_list, dtype="float32")
    LOGGER.info("向量生成完成。")

    LOGGER.info("正在构建并保存 FAISS 索引...")
    RAGStore.build_and_save(documents, embeddings_array, index_dir)
    LOGGER.info("RAG 索引成功创建。")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for zgc3 assistant.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with markdown files.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Output directory for FAISS index.")
    args = parser.parse_args()
    
    settings = get_settings()
    build_index(settings, data_dir=args.data_dir, index_dir=args.index_dir)

if __name__ == "__main__":
    main()