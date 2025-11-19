from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional
from collections import OrderedDict
import shutil

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
                # --- 核心修改：使用 OrderedDict 来记录插入顺序 ---
                return json.load(fp, object_pairs_hook=OrderedDict)
        except (json.JSONDecodeError, FileNotFoundError):
            return deepcopy(default)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            # --- MODIFICATION START ---
            # 仍然使用 NamedTemporaryFile 来保证写入的原子性
            with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self.cache_dir) as tmp:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
                temp_name = tmp.name
            # 使用 shutil.move 替代 Path.replace，以支持跨磁盘驱动器移动
            try:
                shutil.move(temp_name, path)
            except Exception:
                Path(temp_name).unlink() # 如果移动失败，清理临时文件
                raise
            # --- MODIFICATION END ---


    def reload_demos(self) -> None:
        self._demos = self._read_json(self.demos_file, default=[])

    def get_demo(self, demo_id: str) -> Optional[dict]:
        return next((deepcopy(item) for item in self._demos if item.get("id") == demo_id), None)

    def list_demos(self) -> List[dict]:
        return [deepcopy(item) for item in self._demos]

    def get_bilibili_cache(self, keyword: str) -> Optional[List[dict]]:
        cache = self._read_json(self.search_cache_file, default=OrderedDict())
        return deepcopy(cache.get(keyword))


    def set_bilibili_cache(self, keyword: str, items: List[dict], max_keys: int) -> None:
        cache = self._read_json(self.search_cache_file, default=OrderedDict())
        cache[keyword] = items
        # --- 核心修改：实现高效的 FIFO 缓存淘汰 ---
        # 当缓存超过上限时，从开头删除最旧的条目
        while len(cache) > max_keys:
            cache.popitem(last=False) # popitem(last=False) 实现 FIFO
        self._write_json(self.search_cache_file, cache)

