from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple

LOGGER = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    content: str
    source: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

def parse_markdown_to_blocks(content: str) -> List[Tuple[str, str]]:
    """
    将 Markdown 精确地切分成“层级块”。
    """
    LOGGER.debug("\n--- Starting Markdown Block Parsing ---")
    
    blocks: List[Tuple[str, str]] = []
    lines = content.split('\n')
    
    current_block_lines: List[str] = []
    header_stack: List[Tuple[int, str]] = []

    def get_current_header_path() -> str:
        if not header_stack:
            return "(正文)"
        return " > ".join([title for level, title in header_stack])

    for line in lines:
        match = re.match(r"^(#{1,3})\s+(.*)", line)
        
        if match:
            if current_block_lines:
                header_path = get_current_header_path()
                block_content = "\n".join(current_block_lines).strip()
                if block_content:
                    LOGGER.debug(f"Finalizing block for header '{header_path}'")
                    blocks.append((header_path, block_content))
                current_block_lines = []

            level = len(match.group(1))
            title = match.group(2).strip()
            
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            
            header_stack.append((level, title))
            LOGGER.debug(f"Found new header (L{level}): '{title}'. Current stack: {[t for l, t in header_stack]}")

        current_block_lines.append(line)

    if current_block_lines:
        header_path = get_current_header_path()
        block_content = "\n".join(current_block_lines).strip()
        if block_content:
            LOGGER.debug(f"Finalizing last block for header '{header_path}'")
            blocks.append((header_path, block_content))
            
    LOGGER.debug("--- Markdown Block Parsing Finished ---")
    return blocks

def build_hierarchical_tree(blocks: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    根据层级块，构建一个用于回溯的、包含完整聚合内容的层级树。
    """
    tree = {}
    for header_path, content in blocks:
        parts = header_path.split(" > ")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        
        final_node = node.setdefault(parts[-1], {})
        final_node["_self_content"] = content

    def aggregate_content(node: Dict):
        full_content_parts = [node.get("_self_content", "")]
        for key, child_node in node.items():
            if not key.startswith("_"):
                aggregate_content(child_node)
                full_content_parts.append(child_node["_full_content"])
        
        node["_full_content"] = "\n\n".join(p for p in full_content_parts if p).strip()

    aggregate_content(tree)
    return tree
    
def semantic_chunker(text: str, chunk_size: int) -> Iterator[str]:
    """
    一个平衡的、以语义为核心的文本分块器。
    它将连续的短段落组合在一起，同时对超长段落按句子进行细分。
    """
    if not text or chunk_size <= 0:
        return

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_chunk_paragraphs: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_chunk_paragraphs:
                yield "\n\n".join(current_chunk_paragraphs)
                current_chunk_paragraphs = []

            sentences = re.split(r"([。\.!\?\n])", paragraph)
            sentences = ["".join(s).strip() for s in zip(sentences[0::2], sentences[1::2])]
            sentences = [s for s in sentences if s]

            if not sentences:
                yield paragraph
                continue

            current_sentence_group: List[str] = []
            for sentence in sentences:
                if len(" ".join(current_sentence_group + [sentence])) > chunk_size and current_sentence_group:
                    yield " ".join(current_sentence_group)
                    current_sentence_group = [sentence]
                else:
                    current_sentence_group.append(sentence)
            
            if current_sentence_group:
                yield " ".join(current_sentence_group)
        else:
            if len("\n\n".join(current_chunk_paragraphs + [paragraph])) > chunk_size and current_chunk_paragraphs:
                yield "\n\n".join(current_chunk_paragraphs)
                current_chunk_paragraphs = [paragraph]
            else:
                current_chunk_paragraphs.append(paragraph)

    if current_chunk_paragraphs:
        yield "\n\n".join(current_chunk_paragraphs)

def load_and_chunk_documents(data_dir: Path, index_dir: Path, chunk_size: int = 400) -> Tuple[List[Document], Dict[str, Dict[str, Any]]]:
    """
    加载、解析、切分文档，返回用于索引的 Documents 和用于回溯的层级树，并保存结果。
    """
    all_documents: List[Document] = []
    file_trees: Dict[str, Dict[str, Any]] = {}

    for file_path in sorted(data_dir.glob("**/*.md")):
        if not file_path.is_file(): continue
        
        source_name = str(file_path.relative_to(data_dir))
        content = file_path.read_text(encoding="utf-8").strip()
        
        blocks = list(parse_markdown_to_blocks(content))
        if not blocks:
            LOGGER.warning(f"Could not parse any blocks from file: {source_name}")
            continue

        tree = build_hierarchical_tree(blocks)
        file_trees[source_name] = tree

        for header_path, block_content in blocks:
            # 对每个“层级块”再进行二次语义切分
            child_chunks = list(semantic_chunker(block_content, chunk_size))
            
            for idx, child_content in enumerate(child_chunks):
                doc_id = f"{file_path.stem}-{header_path}-{idx}"
                all_documents.append(Document(
                    id=doc_id,
                    content=child_content,
                    source=source_name,
                    metadata={"header_path": header_path}
                ))

    index_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 chunks 预览
    preview_file = index_dir / "chunks_preview.json"
    try:
        with preview_file.open("w", encoding="utf-8") as f:
            json.dump([doc.to_dict() for doc in all_documents], f, ensure_ascii=False, indent=2)
        LOGGER.info(f"Chunk 分割结果预览已保存到: {preview_file.resolve()}")
    except Exception as e:
        LOGGER.error(f"无法保存 chunk 预览文件: {e}")
        
    # 将 file_trees 保存到目录
    file_trees_path = index_dir / "file_trees.json"
    try:
        with file_trees_path.open("w", encoding="utf-8") as f:
            json.dump(file_trees, f, ensure_ascii=False, indent=2)
        LOGGER.info(f"文档层级树 (知识地图) 已保存到: {file_trees_path.resolve()}")
    except Exception as e:
        LOGGER.error(f"无法保存 file_trees 文件: {e}")

    return all_documents, file_trees