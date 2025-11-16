from __future__ import annotations

import logging
import time
from typing import Iterable, List, Optional

from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.adapters.ytdlp_search import YtDlpSearcher
from zgc3_assistant.cache_manager import CacheManager
from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.rag.store import RAGStore

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    """High-level coordinator used by UI callbacks."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        dashscope_client: Optional[DashScopeClient] = None,
        rag_store: Optional[RAGStore] = None,
        cache_manager: Optional[CacheManager] = None,
        ytdlp_searcher: Optional[YtDlpSearcher] = None,
    ):
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or CacheManager(self.settings.cache_dir)
        self.dashscope = dashscope_client
        if self.dashscope is None and self.settings.dashscope_api_key:
            self.dashscope = DashScopeClient(self.settings.dashscope_api_key, self.settings)
        self.rag_store = rag_store or RAGStore.load(self.settings.rag_index_dir)
        if self.rag_store is None:
            LOGGER.warning("RAG index not found. ask_school will skip retrieval.")
        self.ytdlp_searcher = ytdlp_searcher or YtDlpSearcher(
            executable=self.settings.yt_dlp_binary,
            timeout=self.settings.yt_dlp_timeout,
        )

    def _require_dashscope(self) -> DashScopeClient:
        if not self.dashscope:
            raise RuntimeError("DashScope client is not configured.")
        return self.dashscope

    def ask_school(self, query: str) -> dict:
        """Complete RAG workflow."""
        query = (query or "").strip()
        if not query:
            return {"answer_md": "请输入问题。", "sources": []}

        client = self._require_dashscope()
        sources = self._collect_context(query, client)
        context_text = "\n\n".join(f"[{idx+1}] {src['text']}" for idx, src in enumerate(sources))
        system_prompt = (
            "你是中关村第三小学的校史讲解员，请用小学生能理解的语言回答家长与学生的问题。"
        )
        user_prompt = f"问题：{query}\n\n参考资料：\n{context_text or '（暂无）'}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        answer_md = client.chat_omni(messages)
        return {"answer_md": answer_md, "sources": sources}

    def _collect_context(self, query: str, client: DashScopeClient) -> List[dict]:
        if not self.rag_store:
            return []
        embeddings = client.embed_texts([query])
        if not embeddings:
            return []
        hits = self.rag_store.search(embeddings[0], top_k=self.settings.rag_top_k)
        if not hits:
            return []
        return self._rerank_hits(query, hits, client)

    def _rerank_hits(
        self, query: str, hits: List[dict], client: DashScopeClient
    ) -> List[dict]:
        documents = [item["text"] for item in hits]
        reranked = client.rerank(
            query=query,
            documents=documents,
            top_n=min(self.settings.rerank_top_k, len(documents)),
        )
        reordered: List[dict] = []
        for item in reranked:
            idx = item.get("index")
            if idx is None or idx >= len(hits):
                continue
            hit = hits[idx].copy()
            hit["score"] = item.get("score")
            reordered.append(hit)
        if not reordered:
            reordered = hits[: self.settings.rerank_top_k]
        return reordered

    def search_bilibili(self, keyword: str, top_k: int = 8) -> List[dict]:
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        cached = self.cache_manager.get_bilibili_cache(keyword) or []
        if cached:
            return cached[:top_k]
        if not self.settings.enable_ytdlp:
            return []
        limit = min(top_k, self.settings.bili_max_results)
        videos = self.ytdlp_searcher.search(keyword, limit=limit)
        payload = [video.to_dict() for video in videos]
        if payload:
            self.cache_manager.set_bilibili_cache(keyword, payload)
        return payload[:top_k]

    def gen_image(self, prompt: str, size: str = "1328*1328") -> dict:
        if not self.settings.enable_image_gen:
            raise RuntimeError("Image generation feature is disabled.")
        client = self._require_dashscope()
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt is required for image generation.")
        return client.generate_image(prompt=prompt, size=size)

    def gen_video_from_image(self, image_url: str, prompt: str) -> dict:
        if not self.settings.enable_video_gen:
            raise RuntimeError("Image-to-video feature is disabled.")
        client = self._require_dashscope()
        task_id = client.create_i2v_task(image_url=image_url, prompt=prompt)
        start = time.time()
        while True:
            result = client.get_i2v_task_result(task_id)
            status = (result.get("status") or "").lower()
            if status in {"succeeded", "success"}:
                return result
            if status in {"failed", "error"}:
                return result
            if time.time() - start > self.settings.i2v_timeout:
                result["status"] = "timeout"
                return result
            time.sleep(self.settings.i2v_poll_interval)

    def list_demos(self) -> List[dict]:
        return self.cache_manager.list_demos()

    def get_demo(self, demo_id: str) -> Optional[dict]:
        return self.cache_manager.get_demo(demo_id)

