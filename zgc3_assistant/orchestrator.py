from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.adapters.ytdlp_search import YtDlpSearcher
from zgc3_assistant.cache_manager import CacheManager
from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.rag.store import RAGStore
from zgc3_assistant.rag.retriever import RAGRetriever

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    """
    UI 回调使用的高级协调器。
    RAG 逻辑已被解耦到 RAGRetriever 中。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        dashscope_client: Optional[DashScopeClient] = None,
        cache_manager: Optional[CacheManager] = None,
        ytdlp_searcher: Optional[YtDlpSearcher] = None,
    ):
        self.settings = settings or get_settings()
        self._logger = LOGGER
        self.cache_manager = cache_manager or CacheManager(
            self.settings.cache_dir)
        self.dashscope = dashscope_client
        if self.dashscope is None and self.settings.dashscope_api_key:
            self.dashscope = DashScopeClient(
                self.settings.dashscope_api_key, self.settings)

        self.rag_retriever = None
        try:
            # 1. 从文件加载 RAGStore
            self._logger.info("正在从 '%s' 加载 RAG Store...", self.settings.rag_index_dir)
            rag_store_instance = RAGStore.load(self.settings.rag_index_dir)
            self._logger.info("RAG Store 加载成功。")
            
            # 2. 从文件加载文档层级树
            self._logger.info("正在加载文档层级树 (file_trees.json)...")
            file_trees_path = self.settings.rag_index_dir / "file_trees.json"
            with file_trees_path.open("r", encoding="utf-8") as f:
                file_trees = json.load(f)
            self._logger.info("文档层级树加载成功。")

            # 3. 如果所有依赖都齐全，则创建 retriever 实例
            if self.dashscope:
                self.rag_retriever = RAGRetriever(
                    settings=self.settings,
                    rag_store=rag_store_instance,
                    dashscope_client=self.dashscope,
                    file_trees=file_trees
                )
                self._logger.info("RAG Retriever 初始化成功。")
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as e:
            self._logger.warning(f"无法加载 RAG 组件 ({e})。RAG 功能将被禁用。请先运行 build_index 脚本。")

        self.ytdlp_searcher = ytdlp_searcher or YtDlpSearcher(
            executable=self.settings.yt_dlp_binary,
            timeout=self.settings.yt_dlp_timeout,
        )

    def _require_dashscope(self) -> DashScopeClient:
        if not self.dashscope:
            raise RuntimeError(
                "DashScope client is not configured. Please check your API key.")
        return self.dashscope

    def stream_ask_school(
        self, query: str, history: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """
        执行完整的 RAG 对话流程，并以流式方式返回结果。
        """
        query = (query or "").strip()
        if not query:
            yield {"type": "error", "content": "请输入问题。"}
            return

        sources = []
        if self.rag_retriever:
            sources = self.rag_retriever.retrieve(query)
        else:
            self._logger.warning("RAG retriever 未初始化，在无 RAG 模式下运行。")
        
        yield {"type": "sources", "content": sources}

        context_text = "\n\n".join(
            f"[{idx+1}] {src['text']}" for idx, src in enumerate(sources))
        system_prompt = (
            "你是中关村第三小学的校史讲解智能助手。"
            "请用温柔、耐心、友好的语气，回答学生的问题。"
        )
        user_prompt = f"问题：{query}\n\n参考资料：\n{context_text or '（无参考资料）'}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        #self._logger.info("向大模型发送的最终 Prompt: %s", user_prompt)
        self._logger.info("用户问题: %s", query)

        client = self._require_dashscope()
        full_response = ""
        for chunk in client.chat_omni_stream(messages):
            full_response += chunk
            yield {"type": "text_chunk", "content": chunk}

        self._logger.info("流式响应完成，收到的完整回答长度: %d", len(full_response))

    def search_bilibili(self, keyword: str) -> List[dict]:
        """
        搜索 B 站视频，并处理缓存。
        """
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
        videos = self.ytdlp_searcher.search(
            keyword, limit=self.settings.bili_search_limit)
        payload = [video.to_dict() for video in videos]

        for item in payload:
            item["cover"] = self._cache_cover_as_base64(item.get("cover", ""))

        if payload:
            self.cache_manager.set_bilibili_cache(
                keyword, payload, max_keys=self.settings.bili_cache_max_keys)
        return payload

    def _cache_cover_as_base64(self, url: str) -> str:
        """
        带重试机制下载封面图片，并返回 Base64 Data URI。
        """
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

                content_type = response.headers.get(
                    'Content-Type', 'image/jpeg')
                encoded_image = base64.b64encode(
                    response.content).decode('utf-8')
                return f"data:{content_type};base64,{encoded_image}"

            except requests.RequestException as e:
                self._logger.warning(
                    f"下载封面失败 (尝试 {attempt + 1}/{total_attempts}) for {url}: {e}"
                )
                if attempt < total_attempts - 1:
                    time.sleep(0.5)

        self._logger.error(f"下载封面 {url} 在 {total_attempts} 次尝试后彻底失败。")
        return ""