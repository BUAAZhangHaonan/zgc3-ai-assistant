from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv()


class Settings(BaseSettings):
    """Centralized configuration object used across the stack."""

    dashscope_api_key: Optional[str] = Field(
        default=None, alias="DASHSCOPE_API_KEY")

    #model_chat: str = "qwen3-omni-flash"
    model_chat: str = "qwen-flash"
    model_embedding: str = "text-embedding-v4"
    model_rerank: str = "qwen3-rerank"
    model_image: str = "qwen-image-plus"
    model_i2v: str = "wan2.5-i2v-preview"

    rag_chunk_size: int = 200
    rag_top_k: int = 15
    rerank_top_k: int = 4

    enable_show_sources: bool = False  # 是否在回答末尾显示“参考资料”板块
    enable_ytdlp: bool = True
    enable_image_gen: bool = True
    enable_video_gen: bool = False

    log_level: str = "INFO"
    bili_search_limit: int = 6
    bili_cache_max_keys: int = 10

    base_dir: Path = Field(default=BASE_DIR)
    rag_data_dir: Path = Field(default_factory=lambda: BASE_DIR / "rag_data")
    rag_index_dir: Path = Field(default_factory=lambda: BASE_DIR / "rag_index")
    cache_dir: Path = Field(default_factory=lambda: BASE_DIR / "cache")
    assets_dir: Path = Field(default_factory=lambda: BASE_DIR / "assets")

    demos_file: Path = Field(
        default_factory=lambda: BASE_DIR / "cache" / "demos.json")
    search_cache_file: Path = Field(
        default_factory=lambda: BASE_DIR / "cache" / "search_cache.json")

    yt_dlp_binary: str = "yt-dlp"
    yt_dlp_timeout: int = 25  # seconds

    i2v_poll_interval: float = 2.5
    i2v_timeout: int = 180

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Create known directories when they do not exist yet."""
        for path in (self.rag_data_dir, self.rag_index_dir, self.cache_dir, self.assets_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()  # type: ignore[call-arg]
    settings.ensure_directories()
    return settings
