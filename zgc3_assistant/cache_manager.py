from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional


class CacheManager:
    """File-based cache for demos and Bilibili search results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.demos_file = self.cache_dir / "demos.json"
        self.search_cache_file = self.cache_dir / "search_cache.json"
        self._write_lock = threading.Lock()
        self._demos: List[dict] = self._read_json(self.demos_file, default=[])

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
            return deepcopy(default)
        try:
            with path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except json.JSONDecodeError:
            return deepcopy(default)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            with NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
                temp_name = tmp.name
        Path(temp_name).replace(path)

    def reload_demos(self) -> None:
        self._demos = self._read_json(self.demos_file, default=[])

    def get_demo(self, demo_id: str) -> Optional[dict]:
        return next((deepcopy(item) for item in self._demos if item.get("id") == demo_id), None)

    def list_demos(self) -> List[dict]:
        return [deepcopy(item) for item in self._demos]

    def get_bilibili_cache(self, keyword: str) -> Optional[List[dict]]:
        cache = self._read_json(self.search_cache_file, default={})
        return deepcopy(cache.get(keyword))

    def set_bilibili_cache(self, keyword: str, items: List[dict]) -> None:
        cache = self._read_json(self.search_cache_file, default={})
        cache[keyword] = items
        self._write_json(self.search_cache_file, cache)

