import os
from embedder import load_vector_store
from core.retriever import HybridRetriever
from loguru import logger


def run_acceptance_test():
    print("\n🚀 " + "=" * 20 + " RAG 混合检索验收测试 (工程化版) " + "=" * 20 + "\n")

    # 1. 初始化
    try:
        db = load_vector_store()
        # 这里实例化我们的新检索类
        retriever = HybridRetriever(db)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    # 2. 验收案例
    TEST_CASES = [
        {"query": "如何做好logging？", "expected": "logging"},
        {"query": "RPA 的核心定义是什么？", "expected": "RPA"},
        # {"query": "微软的软件工程价值观", "expected": "微软"},
        {"query": "如何升级VSC？", "expected": "VSC"},
        {"query": "如何升级VSD？", "expected": "VSD"},
    ]

    hits = 0
    for case in TEST_CASES:
        query = case["query"]
        expected = case["expected"]
        print(f"\n👉 提问: '{query}'")

        # 直接调用封装好的 search
        final_hybrid_results = retriever.search(query, top_k=3)

        is_hit = False
        print(f"   [RRF 最终排序]:")
        for i, (doc, rrf_score) in enumerate(final_hybrid_results):
            success = (
                expected.lower() in doc.page_content.lower()
                or expected.lower() in doc.metadata.get("source", "").lower()
            )
            mark = "✅" if success else "❌"
            if success:
                is_hit = True

            print(
                f"      [{i+1}] {mark} RRF: {rrf_score:.4f} | {doc.metadata.get('source')} | {doc.page_content[:30].strip()}..."
            )

        if is_hit:
            hits += 1

    print(f"\n📈 最终命中率: {(hits/len(TEST_CASES))*100:.1f}%")


if __name__ == "__main__":
    run_acceptance_test()
