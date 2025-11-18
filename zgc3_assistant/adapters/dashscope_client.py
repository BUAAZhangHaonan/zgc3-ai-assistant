from __future__ import annotations

import importlib
import json
import logging
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Sequence

from zgc3_assistant.config import Settings

# 模块级常量
EMBEDDING_BATCH_SIZE = 10  # 确认 text-embedding-v4 的限制是 10


class DashScopeClient:
    """Wrap DashScope SDK so the rest of the codebase stays decoupled."""

    def __init__(self, api_key: str | None, settings: Settings):
        if not api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY.")
        self.settings = settings
        self.api_key = api_key
        
        # 初始化 logger
        self._logger = logging.getLogger(__name__)

        self._sdk = self._load_sdk()
        self._sdk.api_key = api_key
        self._multimodal = getattr(self._sdk, "MultiModalConversation")
        self._embedding = getattr(self._sdk, "TextEmbedding")
        self._rerank = getattr(self._sdk, "TextReRank")
        self._image = getattr(self._sdk, "ImageSynthesis")
        self._video = getattr(self._sdk, "VideoSynthesis")

    def _load_sdk(self):
        try:
            return importlib.import_module("dashscope")
        except ImportError as exc:
            raise RuntimeError("dashscope package is not installed") from exc

    def _call_and_validate_api(self, api_callable, log_message: str, **kwargs) -> Any:
        """
        统一的API调用和验证层。
        它负责日志、异常捕获和状态检查，并返回完整的、成功的 response 对象。
        """
        self._logger.info(f"正在调用 DashScope API: {log_message}...")
        try:
            # 确保 api_key 总是被传入
            response = api_callable(api_key=self.api_key, **kwargs)
        except Exception as e:
            self._logger.error(f"调用 DashScope API ({log_message}) 时发生异常: %s", e, exc_info=True)
            raise RuntimeError(f"DashScope API call ({log_message}) failed with exception: {e}") from e

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

    def chat_omni(self, messages: Sequence[dict], **kwargs) -> str:
        # 保留原始逻辑：调用API，然后将完整的 response 传递给 _extract_text
        response = self._call_and_validate_api(
            self._multimodal.call,
            log_message=f"Chat ({self.settings.model_chat})",
            model=self.settings.model_chat,
            messages=list(messages),
            result_format="text",
            **kwargs,
        )
        text = self._extract_text(response)
        return text or ""

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        payload = [text for text in texts if text and not text.isspace()]
        if not payload:
            self._logger.warning("输入的所有文本在清理后为空，无需调用 Embedding API。")
            return []

        self._logger.info("接收到 %d 个文本待处理，将以每批 %d 个进行分批调用。", len(payload), EMBEDDING_BATCH_SIZE)
        
        all_embeddings: List[List[float]] = []
        
        for i in range(0, len(payload), EMBEDDING_BATCH_SIZE):
            batch = payload[i : i + EMBEDDING_BATCH_SIZE]
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            total_batches = (len(payload) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
            log_message = f"Embedding ({self.settings.model_embedding}) - Batch {batch_num}/{total_batches}"

            response = self._call_and_validate_api(
                self._embedding.call,
                log_message=log_message,
                model=self.settings.model_embedding,
                input=batch,
                text_type="document",
            )
            
            output_data = response.output
            if not isinstance(output_data, dict) or "embeddings" not in output_data:
                self._logger.error("API 响应格式不符合预期 (批次 %d): 未找到 'embeddings' 字段。", batch_num)
                raise ValueError(f"Invalid response format from DashScope on batch {batch_num}.")

            batch_embeddings = [item["embedding"] for item in output_data["embeddings"] if item and "embedding" in item]
            all_embeddings.extend(batch_embeddings)

        self._logger.info("所有批次处理完成，共获得 %d 个向量。", len(all_embeddings))
        return all_embeddings

    def rerank(self, query: str, documents: Sequence[str], top_n: int = 5) -> List[dict]:
        # 保留原始逻辑：调用API，然后从 response.output 解析
        response = self._call_and_validate_api(
            self._rerank.call,
            log_message=f"Rerank ({self.settings.model_rerank})",
            model=self.settings.model_rerank,
            query=query,
            documents=list(documents),
            top_n=top_n,
            return_documents=True,
        )
        # --- 以下完全是原始逻辑 ---
        data = self._ensure_dict(response.output if hasattr(response, "output") else response)
        results = data.get("results") or []
        normalized: List[dict] = []
        for item in results:
            index = item.get("index")
            normalized.append(
                {
                    "index": index,
                    "score": float(item.get("relevance_score") or item.get("score") or 0.0),
                    "document": item.get("document")
                    or (documents[index] if index is not None and index < len(documents) else ""),
                }
            )
        return normalized[:top_n]

    def generate_image(self, prompt: str, size: str = "1024*1024") -> dict:
        response = self._call_and_validate_api(
            self._image.call,
            log_message=f"Image Generation ({self.settings.model_image})",
            model=self.settings.model_image,
            prompt=prompt,
            size=size,
        )
        # --- 以下完全是原始逻辑 ---
        output = self._ensure_dict(response.output if hasattr(response, "output") else response)
        results = output.get("results") or output.get("data") or []
        first = results[0] if results else {}
        return {
            "url": first.get("url") or first.get("image_url"),
            "meta": first,
            "raw": output,
        }

    def create_i2v_task(self, image_url: str, prompt: str) -> str:
        response = self._call_and_validate_api(
            self._video.call,
            log_message=f"Create I2V Task ({self.settings.model_i2v})",
            model=self.settings.model_i2v,
            img_url=image_url,
            prompt=prompt,
        )
        # --- 以下完全是原始逻辑 ---
        output = self._ensure_dict(response.output if hasattr(response, "output") else response)
        task_id = output.get("task_id") or output.get("id") or response.request_id
        if not task_id:
            raise RuntimeError("DashScope did not return a task_id for video synthesis")
        return task_id

    def get_i2v_task_result(self, task_id: str) -> dict:
        response = self._call_and_validate_api(
            self._video.get,
            log_message=f"Get I2V Task Result",
            task_id=task_id,
        )
        # --- 以下完全是原始逻辑 ---
        output = self._ensure_dict(response.output if hasattr(response, "output") else response)
        status = output.get("task_status") or output.get("status")
        video_url = None
        if "results" in output:
            results = output["results"]
            if isinstance(results, list) and results:
                video_url = results[0].get("url")
        return {
            "task_id": task_id,
            "status": status,
            "video_url": video_url,
            "raw": output,
        }

    # 静态方法和私有辅助方法保持不变，它们是正确的
    @staticmethod
    def _ensure_dict(data) -> dict:
        if data is None: return {}
        if isinstance(data, dict): return data
        if hasattr(data, "to_dict"): return data.to_dict()
        if hasattr(data, "__dict__"): return {k: v for k, v in data.__dict__.items() if not k.startswith("_")}
        try:
            return json.loads(json.dumps(data))
        except Exception:
            logging.debug("Unable to coerce %s to dict", type(data)) # 使用模块级logger
            return {}

    def _extract_text(self, response) -> str:
        # 这个方法是正确的，因为它能处理各种复杂的响应结构
        output = getattr(response, "output", None) or response
        data = self._ensure_dict(output)
        text = data.get("text")
        if text: return text
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict): return first.get("text") or ""
                if isinstance(content, str): return content
        if hasattr(output, "text"): return output.text
        return str(data)