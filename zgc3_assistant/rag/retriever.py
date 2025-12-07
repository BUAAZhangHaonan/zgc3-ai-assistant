from __future__ import annotations
import logging
from typing import List, Dict, Any

import numpy as np

from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.config import Settings
from zgc3_assistant.rag.store import RAGStore
from zgc3_assistant.rag.chunker import Document

LOGGER = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, settings: Settings, rag_store: RAGStore, dashscope_client: DashScopeClient, file_trees: Dict[str, Dict[str, Any]]):
        self.settings = settings
        self.rag_store = rag_store
        self.client = dashscope_client
        self.file_trees = file_trees

    def retrieve(self, query: str) -> List[Dict]:
        LOGGER.info(f"--- Starting Hierarchical RAG for query: '{query}' ---")

        # 问题嵌入
        embeddings_list = self.client.embed_texts([query])
        if not embeddings_list: 
            LOGGER.warning("Query embedding failed.")
            return []
        query_embedding = np.array(embeddings_list[0], dtype="float32")

        # 检索chunk
        reranked_child_docs = self._search_and_rerank_docs(query_embedding, query)
        if not reranked_child_docs:
            LOGGER.info("No relevant child documents found after fine-ranking.")
            return []
        
        # 结果去重
        target_paths_to_process: List[Dict] = []
        for doc in reranked_child_docs:
            header_path = doc.metadata.get("header_path", "")
            source_file = doc.source
            if not header_path or source_file not in self.file_trees: continue

            path_parts = header_path.split(" > ")
            target_header_path = " > ".join(path_parts[:2]) if len(path_parts) > 1 else header_path
            
            target_paths_to_process.append({
                "source": source_file,
                "path": target_header_path,
                "score": doc.metadata.get("rerank_score", 0.0)
            })

        unique_targets_map: Dict[tuple[str, str], float] = {}
        sorted_targets = sorted(target_paths_to_process, key=lambda x: len(x["path"]))
        for target in sorted_targets:
            source = target["source"]
            path = target["path"]
            score = target["score"]
            is_subpath = False
            for existing_path_str, _ in unique_targets_map.items():
                existing_source, existing_path = existing_path_str
                if source == existing_source and path.startswith(existing_path + " > "):
                    is_subpath = True
                    LOGGER.info(f"Path '{path}' is a subpath of existing '{existing_path}'. Skipping.")
                    break
            if not is_subpath:
                if (source, path) not in unique_targets_map:
                    unique_targets_map[(source, path)] = score
                    LOGGER.info(f"Adding unique path: '{path}' (Score: {score:.4f})")
        
        # 提取最终参考资料
        final_docs: List[Dict] = []
        for (source_file, path), score in unique_targets_map.items():
            path_parts = path.split(" > ")

            node = self.file_trees.get(source_file, {})
            for part in path_parts:
                node = node.get(part, {})
            
            full_content = node.get("_full_content")
            if full_content:
                final_docs.append({
                    "text": full_content,
                    "source": f"{source_file} ({path})",
                    "score": score
                })
        
        LOGGER.info(f"上下文聚合后，得到 {len(final_docs)} 个唯一的、非嵌套的章节作为最终上下文。")
        return sorted(final_docs, key=lambda x: x['score'], reverse=True)

    def _search_and_rerank_docs(self, query_embedding: np.ndarray, query: str, score_threshold: float = 0.35) -> List[Document]:
        child_docs = self.rag_store.search(query_embedding, top_k=self.settings.rag_top_k)
        if not child_docs: 
            LOGGER.info("Coarse ranking returned no child documents.")
            return []
        
        LOGGER.info(f"--- Coarse Ranking Results (Top {len(child_docs)}) ---")
        for i, doc in enumerate(child_docs):
            LOGGER.info(f"  {i+1}. [ID: {doc.id}] Header: '{doc.metadata.get('header_path')}'; Content: '{doc.content[:80].replace(chr(10), ' ')}...'")
        
        reranked_items = self.client.rerank(
            query=query,
            documents=[doc.content for doc in child_docs],
            top_n=self.settings.rerank_top_k,
        )
        
        final_docs: List[Document] = []
        LOGGER.info(f"--- Fine Ranking (Reranking) Results ---")
        for item in reranked_items:
            score = item.get("score", 0.0)
            original_index = item.get("index")
            if original_index is None or original_index >= len(child_docs): continue
            
            doc = child_docs[original_index]
            log_msg = f"  - Reranked Doc [Original Index: {original_index}]: Score={score:.4f}, Header='{doc.metadata.get('header_path')}'"
            if score >= score_threshold:
                log_msg += " (PASSED threshold)"
                doc.metadata["rerank_score"] = score
                final_docs.append(doc)
            else:
                log_msg += " (FAILED threshold)"
            LOGGER.info(log_msg)
        
        return final_docs