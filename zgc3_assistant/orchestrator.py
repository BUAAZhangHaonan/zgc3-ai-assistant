from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.adapters.ytdlp_search import YtDlpSearcher
from zgc3_assistant.cache_manager import CacheManager
from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.rag.store import RAGStore

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    """
    UI 回调使用的高级协调器。
    支持流式 RAG 对话和 B 站视频搜索等多种功能。
    """

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
        
        self.rag_store = rag_store
        if self.rag_store is None:
            try:
                self.rag_store = RAGStore.load(self.settings.rag_index_dir)
            except FileNotFoundError:
                LOGGER.warning("RAG index not found at %s. RAG functionality will be disabled.", self.settings.rag_index_dir)
        
        self.ytdlp_searcher = ytdlp_searcher or YtDlpSearcher(
            executable=self.settings.yt_dlp_binary,
            timeout=self.settings.yt_dlp_timeout,
        )
        # 初始化 logger 实例以供内部方法使用
        self._logger = LOGGER

    def _require_dashscope(self) -> DashScopeClient:
        if not self.dashscope:
            raise RuntimeError("DashScope client is not configured. Please check your API key.")
        return self.dashscope

    # --- RAG 对话核心逻辑 (流式) ---

    def stream_ask_school(
        self, query: str, history: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            yield {"type": "error", "content": "请输入问题。"}
            return

        client = self._require_dashscope()
        sources = self._collect_context(query, client)
        yield {"type": "sources", "content": sources}

        context_text = "\n\n".join(f"[{idx+1}] {src['text']}" for idx, src in enumerate(sources))
        system_prompt = (
            "你是中关村第三小学的校史讲解智能助手。"
            "请用耐心、清晰、友好的语气，回答学生的问题。"
        )
        user_prompt = f"问题：{query}\n\n参考资料：\n{context_text or ''}"
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        # --- 核心修改：在这里打印完整的输入 ---
        self._logger.info("向大模型发送的完整消息: %s", messages)

        full_response = ""
        for chunk in client.chat_omni_stream(messages):
            full_response += chunk
            yield {"type": "text_chunk", "content": chunk}
        
        # --- 核心修改：在这里打印完整的输出 ---
        self._logger.info("流式响应完成，收到的完整回答: %s", repr(full_response))

    def _collect_context(self, query: str, client: DashScopeClient) -> List[dict]:
        """执行 RAG 检索和重排以收集上下文。"""
        if not self.rag_store:
            self._logger.warning("RAG store 未加载，跳过上下文检索。")
            return []
        
        self._logger.info(f"正在为查询 '{query}' 检索相关文档...")
        embeddings = client.embed_texts([query])
        if not embeddings:
            self._logger.warning("未能为查询生成 embedding。")
            return []
        
        hits = self.rag_store.search(embeddings[0], top_k=self.settings.rag_top_k)
        if not hits:
            self._logger.info("在向量库中没有找到相关文档。")
            return []
        
        self._logger.info(f"粗排召回 {len(hits)} 个文档，正在进行精排...")
        return self._rerank_hits(query, hits, client)

    def _rerank_hits(
        self, query: str, hits: List[dict], client: DashScopeClient, score_threshold: float = 0.35) -> List[dict]:
        """使用 Reranker 模型对检索结果进行重排序。"""
        documents = [item["text"] for item in hits]
        reranked = client.rerank(
            query=query,
            documents=documents,
            top_n=min(self.settings.rerank_top_k, len(documents)),
        )
        reordered: List[dict] = []
        for item in reranked:
            score = item.get("score", 0.0)
            
            # --- 核心修改：增加分数阈值检查 ---
            if score < score_threshold:
                continue # 如果分数低于阈值，则跳过此文档

            idx = item.get("index")
            if idx is None or idx >= len(hits):
                continue
            
            hit = hits[idx].copy()
            hit["score"] = score # 确保分数被正确赋值
            reordered.append(hit)
        
        # 经过分数过滤后，可能 reordered 为空。
        # 在这种情况下，我们不应再使用粗排结果，而是返回一个空列表，
        # 因为精排模型已经判断所有候选文档的相关性都不足。
        
        if not reordered:
            self._logger.info(f"精排后，没有文档的相关度分数超过阈值 {score_threshold}。")
        else:
            self._logger.info(f"精排并经过分数过滤后，得到 {len(reordered)} 个高质量文档。")
            
        return reordered

    # --- B 站视频搜索核心逻辑 (完整保留) ---

    def search_bilibili(self, keyword: str) -> List[dict]:
        """搜索 B 站视频，并处理缓存。"""
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        
        cached = self.cache_manager.get_bilibili_cache(keyword) or []
        if cached:
            self._logger.info(f"命中了 B 站搜索缓存: '{keyword}'")
            return cached[:self.settings.bili_search_limit]
        
        if not self.settings.enable_ytdlp:
            self._logger.warning("YtDlp 功能未开启，无法搜索 B 站。")
            return []
            
        self._logger.info(f"正在通过 yt-dlp 搜索 B 站: '{keyword}'")
        videos = self.ytdlp_searcher.search(keyword, limit=self.settings.bili_search_limit)
        payload = [video.to_dict() for video in videos]
        
        # 异步思维：可以先返回基本信息，再慢慢加载图片，但在这里我们简化为串行处理
        for item in payload:
            item["cover"] = self._cache_cover_as_base64(item.get("cover", ""))
            
        if payload:
            self.cache_manager.set_bilibili_cache(keyword, payload, max_keys=self.settings.bili_cache_max_keys)
        return payload

    def _cache_cover_as_base64(self, url: str) -> str:
        """带重试机制下载封面图片，并返回 Base64 Data URI。"""
        if not url:
            return ""

        headers = {
            "Referer": "https://www.bilibili.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        total_attempts = 3
        for attempt in range(total_attempts):
            try:
                response = requests.get(url, timeout=10, headers=headers)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', 'image/jpeg')
                encoded_image = base64.b64encode(response.content).decode('utf-8')
                return f"data:{content_type};base64,{encoded_image}"

            except requests.RequestException as e:
                self._logger.warning(
                    f"下载封面失败 (尝试 {attempt + 1}/{total_attempts}) for {url}: {e}"
                )
                if attempt < total_attempts - 1:
                    time.sleep(0.5)

        self._logger.error(f"下载封面 {url} 在 {total_attempts} 次尝试后彻底失败。")
        return "" # 返回空字符串，前端img标签的src为空