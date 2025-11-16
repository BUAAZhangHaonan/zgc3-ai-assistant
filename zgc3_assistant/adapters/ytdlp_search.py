from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class BiliVideo:
    title: str
    url: str
    cover: str
    duration: int
    uploader: str
    raw: dict | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload


class YtDlpSearcher:
    """Thin wrapper around yt-dlp bilisearch."""

    def __init__(self, executable: str = "yt-dlp", timeout: int = 25):
        self.executable = executable
        self.timeout = timeout

    def search(self, keyword: str, limit: int = 8) -> List[BiliVideo]:
        keyword = keyword.strip()
        if not keyword:
            return []
        query = f"bilisearch{limit}:{keyword}"
        cmd = [self.executable, "--dump-json", query]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            LOGGER.error("yt-dlp binary %s not found", self.executable)
            return []
        except subprocess.TimeoutExpired:
            LOGGER.error("yt-dlp search timed out for keyword %s", keyword)
            return []

        if completed.returncode != 0:
            LOGGER.warning("yt-dlp exited with %s: %s", completed.returncode, completed.stderr)
            return []

        videos: List[BiliVideo] = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed line: %s", line[:80])
                continue
            video = self._parse_video(payload)
            if video:
                videos.append(video)
        return videos

    def _parse_video(self, payload: dict) -> Optional[BiliVideo]:
        title = payload.get("title")
        url = payload.get("webpage_url")
        cover = payload.get("thumbnail") or self._first_thumbnail(payload.get("thumbnails"))
        if not title or not url:
            return None
        return BiliVideo(
            title=title,
            url=url,
            cover=cover or "",
            duration=int(payload.get("duration") or 0),
            uploader=payload.get("uploader") or payload.get("channel") or "",
            raw=payload,
        )

    @staticmethod
    def _first_thumbnail(thumbnails) -> str | None:
        if isinstance(thumbnails, list) and thumbnails:
            first = thumbnails[0]
            if isinstance(first, dict):
                return first.get("url")
        elif isinstance(thumbnails, dict):
            return thumbnails.get("url")
        return None
