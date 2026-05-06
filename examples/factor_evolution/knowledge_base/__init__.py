"""
量化因子知识库模块

提供：
- KnowledgeBase: 知识库管理器（CRUD、嵌入缓存）
- KBEmbedder: 嵌入向量生成器
- KnowledgeRetriever: 混合检索器（规则过滤 + 向量相似度）
- KnowledgeWriter: 知识回写器（演化改进 → 知识提取 → 写入KB）
"""

from .kb_manager import KnowledgeBase
from .kb_embedder import KBEmbedder
from .kb_retriever import KnowledgeRetriever
from .kb_writer import KnowledgeWriter

__all__ = ["KnowledgeBase", "KBEmbedder", "KnowledgeRetriever", "KnowledgeWriter"]
