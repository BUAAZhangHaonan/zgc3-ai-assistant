import logging
from pprint import pprint
import time

from zgc3_assistant.config import get_settings
from zgc3_assistant.logging_config import configure_logging
from zgc3_assistant.orchestrator import Orchestrator

# 配置日志，方便我们看到详细输出
configure_logging("INFO")
LOGGER = logging.getLogger(__name__)


def test_rag_pipeline():
    """
    一个带有详细中间输出的脚本，用于完整测试 RAG 的每一步流程。
    """
    print("="*50)
    print(" RAG 流程端到端测试开始 ".center(50, "="))
    print("="*50)

    # --- 1. 初始化 ---
    print("\n[步骤 1/5] 正在初始化Orchestrator和RAG知识库...")
    start_time = time.time()
    settings = get_settings()
    if not settings.dashscope_api_key:
        LOGGER.error("错误：请先在 .env 文件中配置 DASHSCOPE_API_KEY")
        return

    orchestrator = Orchestrator(settings=settings)
    if not orchestrator.rag_store:
        LOGGER.error(
            "错误：RAG 知识库加载失败，请先运行 `python -m zgc3_assistant.rag.build_index`。")
        return

    client = orchestrator.dashscope
    if not client:
        LOGGER.error("错误：DashScope 客户端初始化失败。")
        return

    print(f"--- 初始化完成 (耗时: {time.time() - start_time:.2f}s) ---")

    # --- 2. 向量化查询 ---
    query = "学校的办学理念是什么？"
    print(f"\n[步骤 2/5] 正在向量化测试问题: '{query}'...")
    start_time = time.time()

    query_embedding = client.embed_texts([query])
    if not query_embedding:
        LOGGER.error("API调用失败：问题向量化失败。")
        return

    print(f"--- 问题向量化成功 (耗时: {time.time() - start_time:.2f}s) ---")
    print(f"向量预览 (前5维): {query_embedding[0][:5]}") # 可选：取消注释以查看向量片段

    # --- 3. 粗排 (Vector Search) ---
    print(f"\n[步骤 3/5] 正在执行粗排 (FAISS 向量搜索), top_k={settings.rag_top_k}...")
    start_time = time.time()

    coarse_pass_hits = orchestrator.rag_store.search(
        query_embedding[0], top_k=settings.rag_top_k)

    print(f"--- 粗排完成 (耗时: {time.time() - start_time:.4f}s) ---")
    print("【粗排结果】:")
    if not coarse_pass_hits:
        LOGGER.warning("粗排没有返回任何结果。")
    else:
        for i, hit in enumerate(coarse_pass_hits):
            print(
                f"  [{i+1}] Score: {hit['score']:.4f}, Source: {hit['source']}, Text: '{hit['text'][:100]}...'")

    # --- 4. 精排 (Rerank) ---
    print(
        f"\n[步骤 4/5] 正在执行精排 (Reranker API 调用), top_k={settings.rerank_top_k}...")
    start_time = time.time()

    # 直接调用内部的精排方法
    reranked_hits = orchestrator._rerank_hits(query, coarse_pass_hits, client)

    print(f"--- 精排完成 (耗时: {time.time() - start_time:.2f}s) ---")
    print("【精排结果】:")
    if not reranked_hits:
        LOGGER.error("精排API调用失败或没有返回结果。")
    else:
        pprint(reranked_hits)

    # --- 5. 构建最终上下文 ---
    print("\n[步骤 5/5] 正在构建最终发送给 LLM 的上下文...")

    context_text = "\n\n".join(
        f"[{idx+1}] {src['text']}" for idx, src in enumerate(reranked_hits))

    print("-" * 20)
    print("【最终上下文预览】:")
    print(context_text)
    print("-" * 20)

    print("\n" + "="*50)
    print(" RAG 流程测试结束 ".center(50, "="))
    print("="*50)


if __name__ == "__main__":
    test_rag_pipeline()
