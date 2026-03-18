# Competition Project Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a competition-facing project description for the current repository that is detailed, accurate, and reusable for submission materials.

**Architecture:** The document should be grounded in the current repository state, the competition notice, and the official AI project submission form. It must distinguish implemented features from reserved extension directions and present the project in a review-friendly narrative.

**Tech Stack:** Markdown, repository source code, Gradio, DashScope, FAISS, yt-dlp

---

### Task 1: Gather grounded source facts

**Files:**
- Read: `README.md`
- Read: `app.py`
- Read: `zgc3_assistant/config.py`
- Read: `zgc3_assistant/orchestrator.py`
- Read: `zgc3_assistant/ui/layout.py`
- Read: `zgc3_assistant/rag/*.py`
- Read: `docs/private/中国宋庆龄基金会第21届发明奖通知(1).docx`
- Read: `docs/private/W020260213540389245790.xlsx`

**Step 1: Extract confirmed repository capabilities**

Record only the capabilities that are already implemented in code:
- RAG-based campus question answering
- Bilibili video metadata search and card rendering
- local knowledge-base build pipeline
- cache and configuration mechanisms

**Step 2: Mark reserved or incomplete directions**

Record the parts that must not be overstated:
- image generation
- image-to-video generation
- demo buttons
- always-on source display

### Task 2: Write the formal review document

**Files:**
- Create: `docs/项目说明文档-比赛评审版.md`

**Step 1: Draft the document in competition-review style**

Use natural Chinese paragraphs and organize the content in this order:
- project background
- project positioning and goals
- architecture and technical route
- knowledge-base construction
- core functions and workflow
- implementation process
- demonstration effects and value
- innovation points
- feasibility and strengths
- limitations and future work
- short reusable text for the submission form

**Step 2: Keep implementation claims conservative**

Every “completed” statement must match the codebase and local assets. Every “future” statement must be labeled as planned or reserved.

### Task 3: Verify consistency

**Files:**
- Review: `docs/项目说明文档-比赛评审版.md`

**Step 1: Check consistency against code and materials**

Verify:
- model wording is not stronger than code reality
- RAG availability reflects the existing index pipeline
- Bilibili search wording reflects current UI and adapter behavior
- submission-form-oriented wording is reusable

**Step 2: Final polish**

Ensure the final document is detailed, complete, logically ordered, and suitable for judges to read directly.
