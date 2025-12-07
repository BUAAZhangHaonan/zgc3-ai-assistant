from __future__ import annotations

import importlib
import logging
from http import HTTPStatus
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from zgc3_assistant.config import Settings

EMBEDDING_BATCH_SIZE = 10


class DashScopeClient:
    """封装对DashScope SDK的所有调用。"""

    def __init__(self, api_key: str | None, settings: Settings):
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY.")
        self.settings = settings
        self.api_key = api_key
        self._logger = logging.getLogger(__name__)
        self._sdk = self._load_sdk()
        self._sdk.api_key = api_key
        self._generation = getattr(self._sdk, "Generation")
        self._embedding = getattr(self._sdk, "TextEmbedding")
        self._rerank = getattr(self._sdk, "TextReRank")

    def _load_sdk(self):
        try:
            return importlib.import_module("dashscope")
        except ImportError as exc:
            raise RuntimeError("dashscope package is not installed") from exc

    def _call_and_validate_api(self, api_callable, log_message: str, **kwargs) -> Any:
        self._logger.info(f"正在调用 DashScope API: {log_message}...")
        try:
            response = api_callable(api_key=self.api_key, **kwargs)
        except Exception as e:
            self._logger.error(
                f"调用 DashScope API ({log_message}) 时发生异常: %s", e, exc_info=True)
            raise RuntimeError(
                f"DashScope API call ({log_message}) failed with exception: {e}") from e

        if kwargs.get("stream"):
            self._logger.info(f"DashScope API ({log_message}) 以流式模式启动。")
            return response

        if response.status_code != HTTPStatus.OK:
            self._logger.error(
                "DashScope API (%s) 返回失败状态。 Status: %s, Code: %s, Message: %s",
                log_message, response.status_code, response.code, response.message
            )
            self._logger.error(">>> 失败响应详情: %s", response)
            raise RuntimeError(
                f"DashScope API Error ({log_message}): {response.message} (Code: {response.code})"
            )

        self._logger.info(f"DashScope API ({log_message}) 调用成功。")
        return response

    def chat_omni_stream(self, messages: Sequence[dict], **kwargs) -> Iterator[str]:
        """
        以流式方式调用对话模型。
        核心修复：当前使用的是"qwen-flash"，为了适配之前的omni，需要调整为对外暴露一个标准的“增量式”数据流。
        """
        response_generator = self._call_and_validate_api(
            self._generation.call,
            log_message=f"Chat Stream ({self.settings.model_chat})",
            model=self.settings.model_chat,
            messages=list(messages),
            stream=True,
            result_format="text",
            **kwargs,
        )

        # --- 核心修复：在这里处理数据流格式转换 ---
        previous_content = ""
        for response in response_generator:
            if response.status_code == HTTPStatus.OK:
                # 1. 提取当前收到的完整文本
                full_content = self._extract_text(response)
                
                # 2. 计算出本次新增的文本 (delta)
                # 使用 startswith 确保是连续的流
                if full_content.startswith(previous_content):
                    delta_chunk = full_content[len(previous_content):]
                else: # 如果流不连续，则将当前全部内容视为增量
                    delta_chunk = full_content

                # 3. 对外只 yield 增量部分
                if delta_chunk:
                    yield delta_chunk
                
                # 4. 更新“上一次”的内容
                previous_content = full_content
            else:
                error_message = f"DashScope Stream Error: {response.message} (Code: {response.code})"
                self._logger.error(error_message)
                yield f"\n\n **错误**: {error_message}"
                break


    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        payload = [text for text in texts if text and not text.isspace()]
        if not payload:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(payload), EMBEDDING_BATCH_SIZE):
            batch = payload[i: i + EMBEDDING_BATCH_SIZE]
            log_message = f"Embedding ({self.settings.model_embedding}) - Batch {i//EMBEDDING_BATCH_SIZE + 1}"

            response = self._call_and_validate_api(
                self._embedding.call,
                log_message=log_message,
                model=self.settings.model_embedding,
                input=batch,
                text_type="document",
            )

            output_data = response.output
            if not isinstance(output_data, dict) or "embeddings" not in output_data:
                raise ValueError(
                    "Invalid response format from DashScope on batch.")

            batch_embeddings = [item["embedding"]
                                for item in output_data["embeddings"] if item and "embedding" in item]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def rerank(self, query: str, documents: Sequence[str], top_n: int = 5) -> List[dict]:
        response = self._call_and_validate_api(
            self._rerank.call,
            log_message=f"Rerank ({self.settings.model_rerank})",
            model=self.settings.model_rerank,
            query=query,
            documents=list(documents),
            top_n=top_n,
            return_documents=True,
        )
        data = self._ensure_dict(response.output)
        results = data.get("results") or []
        normalized: List[dict] = []
        for item in results:
            index = item.get("index")
            normalized.append({
                "index": index,
                "score": float(item.get("relevance_score") or 0.0),
                "document": item.get("document") or "",
            })
        return normalized[:top_n]

    @staticmethod
    def _ensure_dict(data: Any) -> Dict[str, Any]:
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        if hasattr(data, "to_dict"):
            return data.to_dict()
        return {}

    def _extract_text(self, response: Any) -> str:
        output = getattr(response, "output", response)
        data = self._ensure_dict(output)

        if choices := data.get("choices"):
            if isinstance(choices, list) and choices:
                if message := choices[0].get("message"):
                    if isinstance(message, dict):
                        if content := message.get("content"):
                            if isinstance(content, list) and content:
                                if inner_content := content[0].get("text"):
                                    return str(inner_content)
                            if isinstance(content, str):
                                return content

        if text := data.get("text"):
            return str(text)

        return ""