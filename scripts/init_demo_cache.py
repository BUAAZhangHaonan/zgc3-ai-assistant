from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMOS = [
    {
        "id": "history-birth",
        "title": "学校成立时间",
        "question": "学校是哪一年成立的？",
        "answer_md": "中关村第三小学建于 **1997 年**，当时的目标是打造科技文化氛围最浓厚的示范小学。",
        "sources": [],
    },
    {
        "id": "sports",
        "title": "校园足球传统",
        "question": "学校在足球方面有哪些亮点？",
        "answer_md": "校园足球队连续三年夺得海淀区冠军。",
        "sources": [],
    },
]


def main() -> None:
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "demos.json"
    with output.open("w", encoding="utf-8") as fp:
        json.dump(DEFAULT_DEMOS, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(DEFAULT_DEMOS)} demo entries to {output}")


if __name__ == "__main__":
    main()
