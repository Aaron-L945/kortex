import jieba
import os
from rank_bm25 import BM25Okapi
from loguru import logger
from config import Config

class HybridRetriever:
    def __init__(self, vector_db, k_rrf=Config.RRF_K):
        """
        初始化混合检索器
        :param vector_db: 加载好的 FAISS 对象
        :param k_rrf: RRF 常数，通常取 60
        """
        self.vector_db = vector_db
        self.k_rrf = k_rrf

        # 1. 自动从向量库提取所有文档对象
        # docstore 是 LangChain FAISS 的存储字典
        self.all_docs = list(vector_db.docstore._dict.values())

        # 2. 预构建 BM25 索引 (稀疏检索)
        logger.info(f"🧠 正在构建 BM25 稀疏索引，样本数: {len(self.all_docs)}")
        tokenized_corpus = [jieba.lcut(doc.page_content) for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.success("✅ HybridRetriever 初始化完成")

    def _rrf_fusion(self, vector_results, bm25_results, top_n=3):
        """内部私有方法：执行 RRF 融合算法"""
        scores = {}
        doc_map = {}

        # 处理向量检索排名 (Score 越小越靠前)
        for rank, (doc, _) in enumerate(vector_results):
            doc_id = hash(doc.page_content)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.k_rrf + rank + 1)
            doc_map[doc_id] = doc

        # 处理 BM25 排名
        for rank, doc in enumerate(bm25_results):
            doc_id = hash(doc.page_content)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.k_rrf + rank + 1)
            doc_map[doc_id] = doc

        # 按 RRF 得分降序排列
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回最终 Top-N 和对应的 RRF 分数
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs[:top_n]]

    def search(self, query, top_k=None):
        """
        统一检索接口
        :param query: 用户问题
        :param top_k: 返回几个结果
        :return: List[(Document, rrf_score)]
        """
        if not query.strip():
            return []
    
        search_k = top_k if top_k else Config.RETRIEVAL_TOP_K

        # A. 执行稠密检索 (为了融合，我们多拿几个，取前 10)
        vector_results = self.vector_db.similarity_search_with_score(query, k=search_k)

        # B. 执行稀疏检索 (BM25，同样多拿几个)
        query_tokens = jieba.lcut(query)
        bm25_top_docs = self.bm25.get_top_n(query_tokens, self.all_docs, n=search_k)

        # C. 融合
        final_results = self._rrf_fusion(vector_results, bm25_top_docs, top_n=search_k)

        return final_results
