"""
权限感知的 RAG 检索层。

流程：
  query → embed → FAISS 搜索 top-K*N → PermissionFilter → 截断 top-K → 返回
"""

from typing import List, Tuple

from models.schemas import DocumentMetadata, UserInfo
from permissions.filter import PermissionFilter
from rag.embedder import Embedder
from rag.indexer import FAISSIndexer


class PermissionAwareRetriever:

    def __init__(self, top_k: int = 5, candidate_multiplier: int = 4):
        self.indexer = FAISSIndexer.get()
        self.embedder = Embedder.get()
        self.top_k = top_k
        # 先多取一些候选，过滤后再截断
        self.candidate_k = top_k * candidate_multiplier

    def retrieve(
        self,
        query: str,
        user: UserInfo,
    ) -> List[Tuple[DocumentMetadata, float]]:
        """
        返回当前用户有权限访问的 top-K 相关 chunks。
        """
        query_vec = self.embedder.embed_query(query)
        candidates = self.indexer.search(query_vec, top_k=self.candidate_k)

        # 仅保留 metadata 部分进行权限过滤
        metas = [m for m, _ in candidates]
        scores = [s for _, s in candidates]

        filtered_metas = PermissionFilter.filter(metas, user)
        filtered_set = {m.chunk_id for m in filtered_metas}

        # 重新组合 (meta, score)，保持分数排序，截断 top_k
        results = [
            (m, s)
            for m, s in zip(metas, scores)
            if m.chunk_id in filtered_set
        ][: self.top_k]

        return results
