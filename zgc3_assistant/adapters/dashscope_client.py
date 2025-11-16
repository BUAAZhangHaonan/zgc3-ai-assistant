from __future__ import annotations

import importlib
import json
import logging
import time
from typing import Iterable, List, Sequence

from zgc3_assistant.config import Settings

LOGGER = logging.getLogger(__name__)


class DashScopeClient:
    """Wrap DashScope SDK so the rest of the codebase stays decoupled."""

    def __init__(self, api_key: str | None, settings: Settings):
        if not api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY.")
        self.settings = settings
        self.api_key = api_key
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
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("dashscope package is not installed") from exc

    def chat_omni(self, messages: Sequence[dict], **kwargs) -> str:
        response = self._multimodal.call(
            model=self.settings.model_chat,
            messages=list(messages),
            result_format="text",
            api_key=self.api_key,
            **kwargs,
        )
        text = self._extract_text(response)
        return text or ""

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        payload = [text for text in texts if text]
        if not payload:
            return []
        response = self._embedding.call(
            model=self.settings.model_embedding,
            input=payload,
            text_type="document",
            api_key=self.api_key,
        )
        data = self._ensure_dict(response.output if hasattr(response, "output") else response)
        embeddings = data.get("embeddings") or data.get("vectors") or []
        return [item.get("embedding") or item for item in embeddings]

    def rerank(self, query: str, documents: Sequence[str], top_n: int = 5) -> List[dict]:
        response = self._rerank.call(
            model=self.settings.model_rerank,
            query=query,
            documents=list(documents),
            top_n=top_n,
            return_documents=True,
            api_key=self.api_key,
        )
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
        response = self._image.call(
            model=self.settings.model_image,
            prompt=prompt,
            size=size,
            api_key=self.api_key,
        )
        output = self._ensure_dict(response.output if hasattr(response, "output") else response)
        results = output.get("results") or output.get("data") or []
        first = results[0] if results else {}
        return {
            "url": first.get("url") or first.get("image_url"),
            "meta": first,
            "raw": output,
        }

    def create_i2v_task(self, image_url: str, prompt: str) -> str:
        response = self._video.call(
            model=self.settings.model_i2v,
            img_url=image_url,
            prompt=prompt,
            api_key=self.api_key,
        )
        output = self._ensure_dict(response.output if hasattr(response, "output") else response)
        task_id = output.get("task_id") or output.get("id") or response.request_id
        if not task_id:
            raise RuntimeError("DashScope did not return a task_id for video synthesis")
        return task_id

    def get_i2v_task_result(self, task_id: str) -> dict:
        if not hasattr(self._video, "get"):
            raise RuntimeError("Installed dashscope SDK does not expose VideoSynthesis.get")
        response = self._video.get(task_id=task_id, api_key=self.api_key)
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

    @staticmethod
    def _ensure_dict(data) -> dict:
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "__dict__"):
            return {k: v for k, v in data.__dict__.items() if not k.startswith("_")}
        try:
            return json.loads(json.dumps(data))
        except Exception:
            LOGGER.debug("Unable to coerce %s to dict", type(data))
            return {}

    def _extract_text(self, response) -> str:
        output = getattr(response, "output", None) or response
        data = self._ensure_dict(output)
        text = data.get("text")
        if text:
            return text
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        return first.get("text") or ""
                if isinstance(content, str):
                    return content
        if hasattr(output, "text"):
            return output.text
        return str(data)

