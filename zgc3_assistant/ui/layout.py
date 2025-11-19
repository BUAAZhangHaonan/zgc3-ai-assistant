from __future__ import annotations
from textwrap import dedent
from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path
import base64

import gradio as gr

from zgc3_assistant.config import Settings, get_settings
from zgc3_assistant.orchestrator import Orchestrator

# --- 辅助函数 ---

def _format_sources_as_collapsible_markdown(sources: List[dict]) -> str:
    if not sources: return ""
    markdown_content = "\n\n---\n<details><summary><strong>📚 查看参考资料来源</strong></summary>\n\n"
    for idx, item in enumerate(sources, 1):
        source_name = item.get('source', '资料').replace('.md', '')
        text_preview = item.get('text', '').replace('\n', ' ').strip()
        markdown_content += f"{idx}. {source_name} (相关度: {item.get('score', 0):.2f})\n> {text_preview}\n\n"
    markdown_content += "</details>"
    return markdown_content

def _format_duration(seconds: int) -> str:
    if not isinstance(seconds, (int, float)) or seconds <= 0: return "N/A"
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

def _format_bili_cards(items: List[dict]) -> str:
    if not items: return "<div class='empty-state'>暂无 B 站搜索结果</div>"
    cards = []
    for item in items:
        cards.append(
            dedent(f"""
            <div class='bili-card card'>
                <a href="{item.get('url')}" target="_blank">
                    <img src="{item.get('cover')}" alt="图片加载失败" referrerpolicy="no-referrer"/>
                    <h4>{item.get('title')}</h4>
                    <p><span>@{item.get('uploader')}</span> <span>{_format_duration(item.get('duration', 0))}</span></p>
                </a>
            </div>
            """)
        )
    return "<div class='bili-grid'>" + "".join(cards) + "</div>"

def encode_image_to_base64(image_path: Path) -> str:
    """将图片文件转换为 Base64 字符串"""
    if not image_path.exists():
        print(f"⚠️ 警告: 找不到背景图片 {image_path}")
        return ""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
            encoded_string = base64.b64encode(data).decode("utf-8")
            ext = image_path.suffix.lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"⚠️ 图片编码错误: {e}")
        return ""

# --- App 构建部分 ---

def build_app(orch: Orchestrator, settings: Settings | None = None) -> gr.Blocks:
    settings = settings or get_settings()
    
    # 获取路径
    assets_path = settings.assets_dir
    bg_image_path = assets_path / "zgc3_background.jpg"
    css_path = assets_path / "style.css"

    # 1. 读取外部 CSS
    base_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    # 2. 生成背景 CSS (Base64)
    bg_base64 = encode_image_to_base64(bg_image_path)
    
    if bg_base64:
        background_css = f"""
        .gradio-container {{
            background: url('{bg_base64}') no-repeat center center fixed !important;
            background-size: cover !important;
            background-color: transparent !important; 
        }}
        body, gradio-app {{
            background: transparent !important;
        }}
        """
    else:
        background_css = ""

    # 3. 合并 CSS (定义 final_css)
    final_css = background_css + "\n" + base_css
    
    welcome_message = {"role": "assistant", "content": "你好呀，我是中关三小校史讲解智能助手！有什么可以帮你的吗？"}

    # 使用 Glass 主题
    with gr.Blocks(theme=gr.themes.Glass(), css=final_css, title="ZGC3 校园 AI 助手") as demo:
        
        api_history_state = gr.State([])

        # 主容器
        with gr.Column(elem_id="main-container"):
            
            # 标题区
            gr.Markdown(
                """
                <div style="text-align: center; margin-bottom: 15px;">
                    <h1 style="margin-bottom: 5px; font-size: 1.8rem;">🏫 中关村三小 · 校史 AI 助手</h1>
                    <p style="opacity: 0.9; font-size: 1rem; color: #333;">探索校史 · 发现精彩 · 智能问答</p>
                </div>
                """
            )

            # 聊天框：高度设为 480，适应一屏
            chatbot = gr.Chatbot(
                value=[welcome_message],
                label="对话历史",
                height=480, 
                show_copy_button=True,
                type="messages",
                avatar_images=(None, (assets_path / "zgc3_logo.png").as_posix()),
                elem_id="chat-window"
            )

            # 输入区：按钮宽度设为 120
            with gr.Row(variant="panel"):
                clear_btn = gr.Button("🗑️ 新对话", variant="secondary", scale=0, min_width=120)
                user_input = gr.Textbox(
                    placeholder="在这里输入你关于校史的问题...", 
                    scale=5, 
                    show_label=False, 
                    container=False, 
                    autofocus=True
                )
                submit_btn = gr.Button("🚀 发送", variant="primary", scale=0, min_width=120)
            
            # 扩展工具区
            gr.Markdown("### 🔧 扩展工具")
            with gr.Accordion("📺 B 站视频搜索", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        search_box = gr.Textbox(label="关键词", placeholder="中关村三小", show_label=False)
                        search_btn = gr.Button("🔍 搜索 B 站", variant="primary")
                    with gr.Column(scale=3, min_width=500):
                        bili_panel = gr.HTML(value="<div class='empty-state'>等待搜索...</div>")
    
        # --- 事件绑定 ---
        def handle_chat_submission(user_message: str, chatbot_ui_history: List[Dict[str, Optional[str]]], api_message_history: List[Dict[str, Any]]) -> Iterator[Dict[gr.component, Any]]:
            if not user_message.strip():
                yield {chatbot: chatbot_ui_history}
                return
            chatbot_ui_history.append({"role": "user", "content": user_message})
            chatbot_ui_history.append({"role": "assistant", "content": ""})
            api_message_history.append({"role": "user", "content": user_message})
            yield {chatbot: chatbot_ui_history, api_history_state: api_message_history, user_input: ""}
            
            full_response = ""
            sources = []
            try:
                stream = orch.stream_ask_school(user_message, api_message_history[:-1])
                for event in stream:
                    if event["type"] == "sources":
                        sources = event["content"]
                    elif event["type"] == "text_chunk":
                        full_response += event["content"]
                        chatbot_ui_history[-1]["content"] = full_response + " ▌"
                        yield {chatbot: chatbot_ui_history}
                    elif event["type"] == "error":
                        full_response = f"❌ 抱歉，出错了: {event['content']}"
                        break
            except Exception as e:
                full_response = f"❌ 抱歉，系统出现了一个意外的错误: {e}"
            
            final_answer = full_response.strip()
            if settings.enable_show_sources:
                final_answer += _format_sources_as_collapsible_markdown(sources)

            chatbot_ui_history[-1]["content"] = final_answer
            api_message_history.append({"role": "assistant", "content": full_response})
            yield {chatbot: chatbot_ui_history, api_history_state: api_message_history}

        def on_search(keyword: str):
            try:
                results = orch.search_bilibili(keyword)
            except Exception as exc:
                return f"<div class='error'>搜索失败：{exc}</div>"
            return _format_bili_cards(results)

        def clear_session() -> Tuple[List[Dict[str, str]], List[Any]]:
            return [welcome_message], []

        submit_btn.click(handle_chat_submission, inputs=[user_input, chatbot, api_history_state], outputs=[chatbot, api_history_state, user_input])
        user_input.submit(handle_chat_submission, inputs=[user_input, chatbot, api_history_state], outputs=[chatbot, api_history_state, user_input])
        search_btn.click(on_search, inputs=search_box, outputs=bili_panel)
        search_box.submit(on_search, inputs=search_box, outputs=bili_panel)
        clear_btn.click(fn=clear_session, inputs=None, outputs=[chatbot, api_history_state], queue=False)

    return demo