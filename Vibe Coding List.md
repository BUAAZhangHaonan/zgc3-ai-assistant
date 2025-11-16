# Vibe Coding List

在这套框架下，后面要加「多轮对话」「会话历史」「角色切换」「更多 demo」都只是在 Orchestrator 和 UI 上叠层，不会动到底层适配器和 RAG 存储。这个解耦程度够支撑后面做成「可复用模板仓库」。

1. **创建基础结构**
   - 在仓库中创建上述目录树和空文件。
   - 填充 `.gitignore`、`environment.yml`。
2. **实现配置层**
   - 在 `zgc3_assistant/config.py` 中实现 `Settings` 和 `get_settings()`。
   - 在 `logging_config.py` 中实现 `configure_logging()`。
3. **实现 DashScope 适配器**
   - 在 `adapters/dashscope_client.py` 中为 omni / embedding / rerank / image / i2v 提供方法，使用 DashScope SDK。
4. **实现 yt-dlp 适配器**
   - 在 `adapters/ytdlp_search.py` 中，用 `subprocess.run` 调用 `yt-dlp --dump-json "bilisearch{N}:关键词"`，解析为 `BiliVideo` 列表。
5. **实现 RAG 流水线**
   - `rag/loader.py`：扫描 `rag_data/`，解析 Markdown 为 `DocumentChunk`。
   - `rag/store.py`：`RAGStore.build_and_save` + `RAGStore.load` + `search`。
   - `rag/build_index.py`：串起来，使用 `DashScopeClient.embed_texts` 构建索引。
6. **实现 CacheManager**
   - `cache_manager.py`：读写 `cache/demos.json` 和 `cache/search_cache.json`，注意并发写时加锁或简单覆盖。
7. **实现 Orchestrator**
   - 在 `orchestrator.py` 中按前文定义的四个方法实现编排逻辑。
8. **实现 Gradio UI**
   - 在 `ui/layout.py` 中构建 `Blocks` 布局，绑定：
     - 文本提问 → `orch.ask_school`
     - B 站搜索 → `orch.search_bilibili`
     - 三个示例按钮 → 直接从 `CacheManager` 读 demo。
9. **连接入口**
   - 在 `app.py` 中实例化 `Orchestrator`，调用 `build_app`，然后 `launch()`。
10. **补上测试**
    - 在 `tests/` 下写最小的单测，使用 Fake Client 避免真实 API 调用。

## 1. GitHub 仓库 & Conda 环境清单

### 1.1 仓库初始化步骤

建议仓库名：`zgc3-ai-assistant`

在空目录下依次做：

1. `git init`
2. 创建基础文件：
   - `README.md`
   - `.gitignore`
   - `environment.yml`
   - `pyproject.toml`（可选，用于依赖和工具统一管理）
3. 创建 Python 包目录：`zgc3_assistant/`（不是纯脚本野生项目）

目录初始骨架如下（后面细讲每个文件）：

```bash
zgc3-ai-assistant/
├─ README.md
├─ .gitignore
├─ environment.yml
├─ pyproject.toml          # 可选，但推荐
├─ app.py                  # 启动 Gradio 的入口脚本
├─ zgc3_assistant/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ logging_config.py
│  ├─ orchestrator.py
│  ├─ ui/
│  │  ├─ __init__.py
│  │  └─ layout.py
│  ├─ adapters/
│  │  ├─ __init__.py
│  │  ├─ dashscope_client.py
│  │  └─ ytdlp_search.py
│  ├─ rag/
│  │  ├─ __init__.py
│  │  ├─ loader.py
│  │  ├─ store.py
│  │  └─ build_index.py
│  └─ cache_manager.py
├─ rag_data/
│  └─ README.md            # 提示：把校史 md 丢这里
├─ cache/
│  ├─ demos.json
│  └─ search_cache.json
├─ assets/
│  ├─ style.css
│  ├─ bg_school.jpg
│  └─ placeholder_image.png
├─ scripts/
│  ├─ init_demo_cache.py
│  └─ run_app.bat          # Windows 一键启动
└─ tests/
   ├─ __init__.py
   └─ test_orchestrator.py
```

### 1.2 `.gitignore` 建议内容

让 AI 创建 `.gitignore`，包含至少：

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.log

# Env / IDE
.env
.venv/
.env.*
.vscode/
.idea/

# RAG index & cache
cache/search_cache.json
rag_index/
*.faiss
*.npy

# OS
.DS_Store
Thumbs.db
```

> `cache/demos.json` 建议版本控制（固定示例），`search_cache.json` 可以忽略。

### 1.3 `environment.yml`（CPU + Win10）

```yaml
name: zgc3-assistant
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy
  - tqdm
  - pip:
      - gradio>=4
      - dashscope
      - httpx
      - pydantic>=2
      - faiss-cpu        # 若 Windows 装不上，再手动处理
      - yt-dlp           # B站搜索，可选
      - python-dotenv    # 本地 .env 管理 API Key
      - rich             # 漂亮日志（可选）
      - black            # 代码格式化
      - ruff             # Lint
      - pytest           # 测试
```

运行：

```bash
conda env update -f environment.yml -n zgc3-assistant
```

------

## 2. 顶层脚本与包结构职责

### 2.1 `app.py` — 整个项目的唯一启动入口

**作用**

- 创建 `Orchestrator` 实例。
- 调用 `zgc3_assistant.ui.layout.build_app(orchestrator)` 返回 Gradio `Blocks`。
- `if __name__ == "__main__":` 中跑 `demo()` 或 `queue().launch()`。

**给 AI 的函数目标：**

```python
from zgc3_assistant.orchestrator import Orchestrator
from zgc3_assistant.ui.layout import build_app

def main():
    orch = Orchestrator()
    demo = build_app(orch)
    demo.queue().launch()

if __name__ == "__main__":
    main()
```

不在 `app.py` 里写业务逻辑，只有 wiring。

### 2.2 `zgc3_assistant/config.py` — 全局配置 & 开关

**作用**

- 集中所有模型名、超参数、开关（yt-dlp / 文生图 / 图生视频）。
- 从环境变量 / `.env` 读取 `DASHSCOPE_API_KEY`。
- 用 Pydantic `BaseSettings` 方便扩展。

**关键点**

- 固定模型名：`qwen3-omni-flash`、`text-embedding-v4`、`qwen3-rerank`、`qwen-image-plus`、`wan2.5-i2v-preview`。
- Feature toggle：`ENABLE_YTDLP`、`ENABLE_IMAGE_GEN`、`ENABLE_VIDEO_GEN`。
- 路径：`RAG_DATA_DIR`、`RAG_INDEX_DIR`、`CACHE_DIR`、`ASSETS_DIR`。

**给 AI 的接口草图：**

```python
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    dashscope_api_key: str

    model_chat: str = "qwen3-omni-flash"
    model_embedding: str = "text-embedding-v4"
    model_rerank: str = "qwen3-rerank"
    model_image: str = "qwen-image-plus"
    model_i2v: str = "wan2.5-i2v-preview"

    enable_ytdlp: bool = True
    enable_image_gen: bool = False
    enable_video_gen: bool = False

    base_dir: Path = Path(__file__).resolve().parents[1]
    rag_data_dir: Path = base_dir / "rag_data"
    rag_index_dir: Path = base_dir / "rag_index"
    cache_dir: Path = base_dir / "cache"
    assets_dir: Path = base_dir / "assets"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

_settings: Settings | None = None

def get_settings() -> Settings:
    ...
```

### 2.3 `zgc3_assistant/logging_config.py` — 日志统一

**作用**

- 项目级 logging 配置，避免到处 `basicConfig`。
- 提供 `configure_logging()` 在 `Orchestrator` 初始化时调用。

------

## 3. Orchestrator 层：业务中枢

### 3.1 `zgc3_assistant/orchestrator.py`

**作用**

- 统一编排 RAG、LLM、B 站、文生图、图生视频、缓存。
- 对 UI 暴露极少数、稳定接口（你在需求里已经给出）。

**核心类 & 方法（稳定签名）：**

```python
from zgc3_assistant.adapters.dashscope_client import DashScopeClient
from zgc3_assistant.adapters.ytdlp_search import YtDlpSearcher
from zgc3_assistant.rag.store import RAGStore
from zgc3_assistant.cache_manager import CacheManager

class Orchestrator:
    def __init__(self, ...):
        # 初始化 settings, client, rag_store, cache_manager, ytdlp_searcher
        ...

    def ask_school(self, query: str) -> dict:
        """
        执行完整 RAG 流程：
        1. embedding → faiss 初筛
        2. qwen3-rerank 精排
        3. qwen3-omni-flash 生成“小学生口吻”答案
        返回结构:
        {
          "answer_md": str,
          "sources": list[{"score": float, "text": str, "source": str}]
        }
        """

    def search_bilibili(self, keyword: str, top_k: int = 8) -> list[dict]:
        """
        调用 yt-dlp bilisearch，仅在 ENABLE_YTDLP 为 True 时启用。
        返回:
        [
          {
            "title": str,
            "url": str,
            "cover": str,
            "duration": int,
            "uploader": str
          },
          ...
        ]
        """

    def gen_image(self, prompt: str, size: str = "1328*1328") -> dict:
        """
        调用 qwen-image-plus，文生图。
        返回: {"url": str, "meta": dict}
        """

    def gen_video_from_image(self, image_url: str, prompt: str) -> dict:
        """
        调用 wan2.5-i2v-preview：
        - 提交创建任务
        - 轮询任务状态
        返回:
        {
          "task_id": str,
          "status": str,   # "pending"/"running"/"succeeded"/"failed"
          "video_url": Optional[str],
          "meta": dict
        }
        """
```

**Orchestrator 依赖关系：**

- `DashScopeClient`: 所有云端模型调用统一入口。
- `RAGStore`: 本地向量库检索 + meta。
- `CacheManager`: 读写 `cache/demos.json`、`cache/search_cache.json`。
- `YtDlpSearcher`: B 站搜索，内部可选关闭。

Orchestrator 本身不关心底层 API 细节，只组合。

------

## 4. Adapters 层：对外部依赖做“统一皮肤”

### 4.1 `zgc3_assistant/adapters/dashscope_client.py`

**作用**

- 对 DashScope SDK 做轻量封装，隔离云 API 的具体字段。
- 一个类内提供多种模型调用方法。

**建议接口：**

```python
from typing import Iterable

class DashScopeClient:
    def __init__(self, api_key: str, settings: Settings):
        ...

    def chat_omni(self, messages: list[dict], **kwargs) -> str:
        """
        使用 qwen3-omni-flash，多模态对话。
        messages 结构兼容 OpenAI / DashScope 习惯:
        [{"role": "system"/"user"/"assistant", "content": ...}, ...]
        返回 Markdown string。
        """

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """
        调用 text-embedding-v4，对一批文本向量化。
        返回二维 list[embedding_dim]。
        """

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[dict]:
        """
        调用 qwen3-rerank，返回带 score 的排序结果，附带索引。
        """

    def generate_image(self, prompt: str, size: str) -> dict:
        """
        调用 qwen-image-plus 文生图，返回图片 URL 和 meta。
        """

    def create_i2v_task(self, image_url: str, prompt: str) -> str:
        """调用 wan2.5-i2v-preview 创建任务，返回 task_id。"""

    def get_i2v_task_result(self, task_id: str) -> dict:
        """查询 wan2.5-i2v-preview 任务状态和结果。"""
```

后续如果换 API（比如改走 OpenAI 兼容接口），只改这里。

### 4.2 `zgc3_assistant/adapters/ytdlp_search.py`

**作用**

- 用 `subprocess` 调用 `yt-dlp --dump-json "bilisearch{N}:关键词"`。
- 把多行 JSON 解析成统一数据结构，供 Orchestrator / UI 使用。
- 支持直接关闭：`Settings.enable_ytdlp = False` 时返回空列表或从缓存读。

**接口：**

```python
from dataclasses import dataclass

@dataclass
class BiliVideo:
    title: str
    url: str
    cover: str
    duration: int
    uploader: str

class YtDlpSearcher:
    def __init__(self, settings: Settings):
        ...

    def search(self, keyword: str, limit: int = 8) -> list[BiliVideo]:
        """
        若 yt-dlp 不可执行或被禁用，则返回空列表。
        """
```

------

## 5. RAG 层：文档加载、索引构建与检索

### 5.1 `zgc3_assistant/rag/loader.py`

**作用**

- 读取 `rag_data/` 下的校史 Markdown / txt 文件。
- 按段落或标题切分为文档块。
- 产出统一结构：`DocumentChunk(id, text, source, meta)`。

**接口：**

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str        # 文件名或路径
    meta: dict

def load_markdown_corpus(data_dir: Path) -> list[DocumentChunk]:
    """扫描目录，返回所有分块。"""
```

### 5.2 `zgc3_assistant/rag/store.py`

**作用**

- 封装 FAISS + 元数据存储。
- 提供在线检索接口：给 query，返回 TopK 文档块。

**接口设计：**

```python
import faiss
import numpy as np

class RAGStore:
    def __init__(self, index, id_map: list[DocumentChunk]):
        ...

    @classmethod
    def load(cls, index_dir: Path) -> "RAGStore":
        """从磁盘加载 FAISS 索引和 meta 信息。"""

    @classmethod
    def build_and_save(
        cls,
        chunks: list[DocumentChunk],
        embed_fn,              # 函数：texts -> np.ndarray
        index_dir: Path,
    ) -> "RAGStore":
        """
        构建索引并保存到 index_dir。
        embed_fn 内部可以调用 DashScopeClient.embed_texts。
        """

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[DocumentChunk]:
        """
        输入已经算好的 query 向量，返回 top_k 文档块。
        """
```

### 5.3 `zgc3_assistant/rag/build_index.py`

**作用**

- 离线脚本：构建并保存 RAG 索引。
- 仅在你整理好校史 md 后运行，平时不会被 UI 调用。

**脚本逻辑：**

```python
def main():
    settings = get_settings()
    client = DashScopeClient(settings.dashscope_api_key, settings)
    chunks = load_markdown_corpus(settings.rag_data_dir)
    # texts -> embeddings
    RAGStore.build_and_save(
        chunks=chunks,
        embed_fn=client.embed_texts,
        index_dir=settings.rag_index_dir,
    )

if __name__ == "__main__":
    main()
```

------

## 6. 缓存层：示例任务 & B 站搜索缓存

### 6.1 `zgc3_assistant/cache_manager.py`

**作用**

- 统一读写 `cache/` 下的 JSON 文件。
- “新操场 / 足球赛 / 校园布局” 三个示例任务由这里对外提供。

**建议结构**

`cache/demos.json`：

```json
[
  {
    "id": "playground",
    "title": "新操场故事",
    "query": "介绍一下中关村三小的新操场……（原始问题）",
    "answer_md": "预先生成好的答案 Markdown",
    "sources": [
      {"text": "……", "source": "history_01.md", "score": 0.98}
    ]
  },
  ...
]
```

`cache/search_cache.json`：

```json
{
  "中关村三小 足球 比赛": [
    {
      "title": "...",
      "url": "...",
      "cover": "...",
      "duration": 123,
      "uploader": "..."
    }
  ]
}
```

**接口：**

```python
class CacheManager:
    def __init__(self, cache_dir: Path):
        ...

    def get_demo(self, demo_id: str) -> dict | None:
        ...

    def list_demos(self) -> list[dict]:
        ...

    def get_bilibili_cache(self, keyword: str) -> list[dict] | None:
        ...

    def set_bilibili_cache(self, keyword: str, items: list[dict]) -> None:
        ...
```

Orchestrator 在 `search_bilibili` 里先查缓存再访问 yt-dlp。

### 6.2 `scripts/init_demo_cache.py`

**作用**

- 将你预先写好的 demo 内容写入 `cache/demos.json`。
- 也可预先调用 Orchestrator 生成答案后保存。

------

## 7. UI 层：Gradio Blocks 布局与回调

### 7.1 `zgc3_assistant/ui/layout.py`

**作用**

- 构建 Gradio 单页 UI。
- 这里只负责布局 & 组件绑定，不包含业务逻辑实现细节。

**主接口：**

```python
import gradio as gr
from zgc3_assistant.orchestrator import Orchestrator
from zgc3_assistant.config import get_settings

def build_app(orch: Orchestrator) -> gr.Blocks:
    """
    - 主题: gr.themes.Glass()
    - 自定义 CSS: assets/style.css
    - 左侧: 提问输入框 + 提交按钮
    - 中部: 答案卡片（Markdown + 可折叠 source）
    - 右侧: B站视频卡片墙
    - 顶部: 三个示例按钮，调用 orch + cache 直接读 demos.json
    """
    ...
```

**建议组件结构：**

- 顶部 `Row`：Logo + 标题 + 副标题。
- 中间 `Row` 分三列：
  - 左列：`Textbox` + 提问按钮 + 示例按钮。
  - 中列：`Markdown` + `Accordion`（显示引用片段）。
  - 右列：`Gallery` 或自定义 `HTML` 显示 B 站封面卡片。
- 页面级 CSS：`assets/style.css`。

### 7.2 `assets/style.css`

**作用**

- 自定义背景与卡片样式。

**最低要求内容：**

```css
.gradio-container {
  background: url('/file=assets/bg_school.jpg') center/cover no-repeat fixed;
}

.card {
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  backdrop-filter: blur(12px);
}

.bili-card img {
  border-radius: 10px;
  display: block;
}
```

------

## 8. `scripts/run_app.bat` — 一键启动

**作用**

- 在 Windows 上双击即可启动，包括激活 Conda 环境。

**内容示意：**

```bat
@echo off
call conda activate zgc3-assistant
python app.py
pause
```

------

## 9. 测试骨架

### 9.1 `tests/test_orchestrator.py`

**作用**

- 确认基本接口存在 + 能在假数据上跑。
- 初期可以对 DashScope/yt-dlp 做 Mock，保证 CI 不依赖外部服务。

**内容目标：**

- `Orchestrator` 初始化不报错（在缺少 API Key 时要么抛清晰异常，要么用 FakeClient）。
- `ask_school` 在使用本地 Fake RAG + Fake Client 时返回 dict，包含 `answer_md` 和 `sources` 字段。
- `search_bilibili` 在 `ENABLE_YTDLP=False` 时能正常返回空列表。
