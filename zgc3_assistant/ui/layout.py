from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import List

import gradio as gr

from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.orchestrator import Orchestrator


def _format_sources(sources: List[dict]) -> str:
    if not sources:
        return "_暂无引用片段_"
    lines = []
    for idx, item in enumerate(sources, 1):
        lines.append(
            f"**{idx}. {item.get('source', '资料')}**（score={item.get('score', 0):.2f})\n"
            f"> {item.get('text', '')[:400]}"
        )
    return "\n\n".join(lines)


def _format_bili_cards(items: List[dict]) -> str:
    if not items:
        return "<div class='empty-state'>暂无 B 站搜索结果</div>"
    cards = []
    for item in items:
        cards.append(
            dedent(
                f"""
                <div class='bili-card card'>
                    <a href="{item.get('url')}" target="_blank">
                        <img src="{item.get('cover')}" alt="{item.get('title')}"/>
                        <h4>{item.get('title')}</h4>
                        <p>{item.get('uploader')} · {item.get('duration', 0)}s</p>
                    </a>
                </div>
            """
            )
        )
    return "<div class='bili-grid'>" + "".join(cards) + "</div>"


def build_app(orch: Orchestrator, settings: Settings | None = None) -> gr.Blocks:
    settings = settings or get_settings()
    css_path = settings.assets_dir / "style.css"
    custom_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    demos = orch.list_demos()[:3]

    with gr.Blocks(
        theme=gr.themes.Glass(), css=custom_css, title="ZGC3 校园 AI 助手"
    ) as demo:
        gr.Markdown(
            dedent(
                """
                # 中关村三小 · 校史智能助手
                在一个界面里体验问答、校史 RAG、B 站搜索示例。
                """
            )
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                question = gr.Textbox(
                    label="向校史顾问提问",
                    placeholder="例如：学校是哪一年成立的？",
                    lines=5,
                )
                ask_btn = gr.Button("提问", variant="primary")
                demo_buttons = []
                for item in demos:
                    demo_buttons.append(
                        gr.Button(item.get("title") or item.get("question", "示例"))
                    )
                search_box = gr.Textbox(label="B 站搜索关键词", placeholder="足球 比赛")
                search_btn = gr.Button("搜索 B 站")
            with gr.Column(scale=2, min_width=420):
                answer_md = gr.Markdown(label="回答")
                with gr.Accordion("引用片段", open=False):
                    sources_md = gr.Markdown()
            with gr.Column(scale=1, min_width=320):
                bili_panel = gr.HTML(value="<div class='empty-state'>等待搜索</div>")

        def on_ask(user_query: str):
            try:
                payload = orch.ask_school(user_query)
            except Exception as exc:  # pragma: no cover - UI guard
                return f"❌ {exc}", "_出错了_"
            return payload.get("answer_md", ""), _format_sources(payload.get("sources", []))

        def on_search(keyword: str):
            try:
                results = orch.search_bilibili(keyword)
            except Exception as exc:  # pragma: no cover - UI guard
                return f"<div class='error'>搜索失败：{exc}</div>"
            return _format_bili_cards(results)

        def load_demo(demo_id: str):
            demo_payload = orch.get_demo(demo_id)
            if not demo_payload:
                return "", "_示例不存在_", "_示例不存在_"
            return (
                demo_payload.get("question", ""),
                demo_payload.get("answer_md", ""),
                _format_sources(demo_payload.get("sources", [])),
            )

        ask_btn.click(on_ask, inputs=question, outputs=[answer_md, sources_md])
        question.submit(on_ask, inputs=question, outputs=[answer_md, sources_md])
        search_btn.click(on_search, inputs=search_box, outputs=bili_panel)
        search_box.submit(on_search, inputs=search_box, outputs=bili_panel)

        for btn, demo_info in zip(demo_buttons, demos):
            btn.click(
                fn=lambda _=None, demo_id=demo_info["id"]: load_demo(demo_id),
                inputs=None,
                outputs=[question, answer_md, sources_md],
            )

    return demo

