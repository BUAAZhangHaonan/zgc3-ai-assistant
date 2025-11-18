# RAG 数据填充指南

将校史、活动报道等 Markdown 文件放在rag_data/目录下，执行

```bash
python -m zgc3_assistant.rag.build_index
```

即可使用 DashScope embedding 构建 FAISS 索引（存储在 `rag_index/`）。
