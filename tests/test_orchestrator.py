from __future__ import annotations

import json
from pathlib import Path

from zgc3_assistant.cache_manager import CacheManager
from zgc3_assistant.config import Settings
from zgc3_assistant.orchestrator import Orchestrator


class FakeDashScopeClient:
    def chat_omni(self, messages):
        return "这是一个测试回答。"

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3]]

    def rerank(self, query, documents, top_n):
        return [{"index": 0, "score": 0.99, "document": documents[0]}]


class FakeRAGStore:
    def search(self, embedding, top_k):
        return [
            {
                "text": "学校成立于 1997 年。",
                "source": "history.md",
                "score": 0.9,
                "metadata": {"chunk_index": 0},
            }
        ]


def _prepare_cache(tmp_path: Path) -> CacheManager:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    demos_file = cache_dir / "demos.json"
    demos_file.write_text(
        json.dumps(
            [
                {
                    "id": "demo",
                    "title": "示例",
                    "question": "学校是什么时候成立的？",
                    "answer_md": "学校成立于 1997 年。",
                    "sources": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (cache_dir / "search_cache.json").write_text("{}", encoding="utf-8")
    return CacheManager(cache_dir)


def test_ask_school_returns_structure(tmp_path):
    cache_manager = _prepare_cache(tmp_path)
    settings = Settings(
        dashscope_api_key="dummy",
        cache_dir=cache_manager.cache_dir,
        rag_index_dir=tmp_path / "rag_index",
    )
    orch = Orchestrator(
        settings=settings,
        dashscope_client=FakeDashScopeClient(),
        rag_store=FakeRAGStore(),
        cache_manager=cache_manager,
    )
    result = orch.ask_school("学校成立时间？")
    assert "answer_md" in result
    assert isinstance(result["sources"], list)
    assert result["sources"][0]["source"] == "history.md"


def test_search_bilibili_disabled_returns_empty(tmp_path):
    cache_manager = _prepare_cache(tmp_path)
    settings = Settings(
        dashscope_api_key="dummy",
        cache_dir=cache_manager.cache_dir,
        enable_ytdlp=False,
    )
    orch = Orchestrator(
        settings=settings,
        dashscope_client=FakeDashScopeClient(),
        rag_store=FakeRAGStore(),
        cache_manager=cache_manager,
    )
    assert orch.search_bilibili("足球") == []
