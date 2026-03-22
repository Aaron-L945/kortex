"""
LlamaIndex Agent 工具集。

提供给 ReActAgent 使用的 FunctionTool：
  - knowledge_retrieval: 从知识库检索相关信息
  - summarize_context: 对检索结果做摘要（Claude 内部处理，此处为占位）

工具通过闭包绑定当前用户，保证权限隔离。
"""

from typing import List
from llama_index.core.tools import FunctionTool

from models.schemas import UserInfo
from rag.retriever import PermissionAwareRetriever


def build_tools(user: UserInfo, top_k: int = 5) -> List[FunctionTool]:
    retriever = PermissionAwareRetriever(top_k=top_k)

    def knowledge_retrieval(query: str) -> str:
        """
        在企业知识库中检索与问题相关的文档片段。
        当需要回答专业问题、查找公司政策、产品文档或内部资料时调用此工具。
        参数：
          query: 搜索关键词或问题描述
        返回：相关文档片段及来源
        """
        results = retriever.retrieve(query, user)
        if not results:
            return "知识库中未找到相关内容。"

        parts = []
        for i, (meta, score) in enumerate(results, 1):
            parts.append(
                f"[来源 {i}] 《{meta.title}》（相关度: {score:.3f}）\n{meta.content}"
            )
        return "\n\n---\n\n".join(parts)

    def list_accessible_topics() -> str:
        """
        列出当前用户可访问的知识库主题概览。
        当用户想知道知识库里有什么内容时调用。
        """
        from rag.indexer import FAISSIndexer
        indexer = FAISSIndexer.get()

        from permissions.filter import PermissionFilter
        accessible = PermissionFilter.filter(indexer.metadata, user)

        if not accessible:
            return "当前没有可访问的知识库内容。"

        titles = list({m.title for m in accessible})
        return "可访问的文档主题：\n" + "\n".join(f"  • {t}" for t in titles[:30])

    return [
        FunctionTool.from_defaults(fn=knowledge_retrieval),
        FunctionTool.from_defaults(fn=list_accessible_topics),
    ]
