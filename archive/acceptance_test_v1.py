"""
引入BM25 + 向量检索
"""

import os
import time
import jieba
from rank_bm25 import BM25Okapi
from embedder import load_vector_store
from config import Config
from loguru import logger

# --- 新增：BM25 初始化工具 ---
def prepare_bm25(db):
    """从 FAISS 中提取所有文本并构建 BM25 索引"""
    # 假设你使用的是 LangChain 的 FAISS 包装器
    all_docs = list(db.docstore._dict.values())
    # 对所有文档内容进行分词
    tokenized_corpus = [jieba.lcut(doc.page_content) for doc in all_docs]
    return BM25Okapi(tokenized_corpus), all_docs

def run_acceptance_test():
    print("\n🚀 " + "="*20 + " RAG 混合检索验收测试 (Hybrid) " + "="*20 + "\n")

    # 1. 加载 FAISS
    try:
        db = load_vector_store()
        total_vectors = db.index.ntotal
        print(f"📊 [库状态] 向量切片总量: {total_vectors}")
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return

    # 2. 构建内存 BM25 (即你说的稀疏检索能力)
    print("🧠 正在同步构建稀疏检索索引 (BM25)...")
    bm25, doc_list = prepare_bm25(db)

    # 3. 验收案例
    TEST_CASES = [
        {"query": "如何做好logging？", "expected_source": "logging", "desc": "日志规范"},
        {"query": "RPA 的核心定义是什么？", "expected_source": "RPA", "desc": "业务定义"},
        {"query": "微软的软件工程价值观", "expected_source": "微软", "desc": "企业文化"},
        {"query": "如何升级VSD？", "expected_source": "VSD", "desc": "技术文档"},
    ]
    
    hits = 0
    total = len(TEST_CASES)
    
    for case in TEST_CASES:
        query = case["query"]
        expected = case["expected_source"]
        print(f"\n👉 提问: '{query}' ({case['desc']})")
        
        # --- A. 稠密检索 (FAISS) ---
        vector_results = db.similarity_search_with_score(query, k=2)
        
        # --- B. 稀疏检索 (BM25) ---
        query_tokens = jieba.lcut(query)
        # 获取得分最高的前 2 个
        bm25_top_docs = bm25.get_top_n(query_tokens, doc_list, n=2)

        # --- C. 合并结果 (Hybrid) ---
        # 只要有一方命中就算命中
        is_hit = False
        
        print(f"   [稠密检索(Vector) 结果]:")
        for doc, score in vector_results:
            success = expected.lower() in doc.page_content.lower() or expected.lower() in doc.metadata.get('source','').lower()
            mark = "✅" if success else "❌"
            if success: is_hit = True
            print(f"      {mark} Score: {score:.4f} | {doc.metadata.get('source')}...")

        print(f"   [稀疏检索(BM25) 结果]:")
        for doc in bm25_top_docs:
            success = expected.lower() in doc.page_content.lower() or expected.lower() in doc.metadata.get('source','').lower()
            mark = "✅" if success else "❌"
            if success: is_hit = True
            print(f"      {mark} | {doc.metadata.get('source')} | 片段: {doc.page_content[:30].strip()}...")

        if is_hit: hits += 1

    # 4. 统计
    hit_rate = (hits / total) * 100
    print("\n" + "="*50)
    print(f"📈 最终验收结果: {hit_rate:.1f}% (目标 ≥ 70%)")
    if hit_rate >= 70:
        print("🎉 【混合检索验收通过】 哪怕向量距离远，BM25 也帮你抓到了关键词！")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_acceptance_test()