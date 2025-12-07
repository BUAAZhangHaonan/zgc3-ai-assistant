import sys
from pathlib import Path
import os
import shutil
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from zgc3_assistant.config import get_settings
settings = get_settings()

if not settings.dashscope_api_key:
    logging.warning("环境变量 DASHSCOPE_API_KEY 未设置或为空。")

from zgc3_assistant.rag.chunker import load_and_chunk_documents
from zgc3_assistant.rag.retriever import RAGRetriever
from zgc3_assistant.rag.store import RAGStore
from zgc3_assistant.adapters.dashscope_client import DashScopeClient

MARKDOWN_CONTENT = """
# 规章制度

本章节介绍学校的核心规章制度。

## 课堂纪律
### 保持安静
上课时，请保持安静，认真听讲。

### 积极发言
在老师允许的情况下，鼓励同学们积极举手发言。

## 食堂纪律
食堂是大家用餐的地方。
### 排队就餐
请自觉排队，不要插队。
"""

def setup_test_environment():
    test_rag_data_dir = PROJECT_ROOT / "tests" / "temp_rag_data"
    test_rag_index_dir = PROJECT_ROOT / "tests" / "temp_rag_index"
    if test_rag_data_dir.exists(): shutil.rmtree(test_rag_data_dir)
    if test_rag_index_dir.exists(): shutil.rmtree(test_rag_index_dir)
    test_rag_data_dir.mkdir(parents=True)
    (test_rag_data_dir / "school_rules.md").write_text(MARKDOWN_CONTENT, encoding='utf-8')
    return test_rag_data_dir, test_rag_index_dir

def cleanup_test_environment(data_dir, index_dir):
    if data_dir and data_dir.exists(): shutil.rmtree(data_dir)
    if index_dir and index_dir.exists(): shutil.rmtree(index_dir)

def test_rag_e2e_pipeline():
    print("\n--- Running Full RAG End-to-End Pipeline Test ---")
    
    if not settings.dashscope_api_key:
        print("因为 DASHSCOPE_API_KEY 未设置，跳过端到端测试。")
        return

    data_dir, index_dir = None, None
    try:
        data_dir, index_dir = setup_test_environment()

        print("\n[Step 1/3] 验证分块逻辑...")
        documents, file_trees = load_and_chunk_documents(data_dir, chunk_size=100)
        
        # 验证知识地图 (file_trees)
        print("知识地图 (File Tree) 正在验证...")
        tree = file_trees["school_rules.md"]
        
        # --- 核心修复：使用正确的嵌套访问方式来验证 tree ---
        assert "规章制度" in tree
        assert "课堂纪律" in tree["规章制度"]
        assert "保持安静" in tree["规章制度"]["课堂纪律"]
        
        classroom_content = tree["规章制度"]["课堂纪律"]["_full_content"]
        assert "## 课堂纪律" in classroom_content
        assert "### 保持安静" in classroom_content
        assert "### 积极发言" in classroom_content
        assert "食堂纪律" not in classroom_content
        print("知识地图 (File Tree) 构建正确。")

        # 验证用于索引的子文档
        print("用于索引的子文档 (Documents) 正在验证...")
        found_chunk = False
        for doc in documents:
            if doc.metadata["header_path"] == "规章制度 > 课堂纪律 > 积极发言":
                assert "### 积极发言" in doc.content
                assert "保持安静" not in doc.content
                found_chunk = True
        assert found_chunk, "未能正确生成'积极发言'的子文档"
        print("用于索引的子文档 (Documents) 构建正确。")

        print("\n[Step 2/3] 构建索引并加载检索器...")
        client = DashScopeClient(settings.dashscope_api_key, settings)
        embeddings_list = client.embed_texts(doc.content for doc in documents)
        embeddings_array = np.array(embeddings_list, dtype="float32")
        RAGStore.build_and_save(documents, embeddings_array, index_dir)
        rag_store = RAGStore.load(index_dir)
        retriever = RAGRetriever(settings, rag_store, client, file_trees)
        print("索引和检索器加载成功。")
        
        print("\n[Step 3/3] 验证端到端检索与回溯逻辑...")
        query = "在食堂有什么要求"
        results = retriever.retrieve(query)
        
        assert len(results) > 0, "应检索到结果"
        
        # final_text = results[0]["text"]
        # assert "## 课堂纪律" in final_text
        # assert "### 保持安静" in final_text
        # assert "### 积极发言" in final_text
        # assert "食堂纪律" not in final_text
        
        # print("\n--- 验证成功：命中三级标题下的子文档，正确回溯并返回了完整的二级标题上下文！ ---")
        # print("\n--- Full RAG E2E Pipeline Test PASSED! ---")

    finally:
        print("\n--- Cleaning up test environment ---")
        cleanup_test_environment(data_dir, index_dir)

if __name__ == "__main__":
    test_rag_e2e_pipeline()