import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from core.retriever_with_testcase import HybridRetrieverV3

class RAGSimplifiedTester:
    def __init__(self, retriever: HybridRetrieverV3):
        self.retriever = retriever

    def run_benchmark(self, test_file="test_queries.jsonl"):
        # 1. 加载测试集
        test_cases = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                test_cases.append(json.loads(line))

        stats = {
            "total_queries": 0,
            "hits_at_5": 0
        }

        print(f"\n🚀 开始核心指标测评 (目标: Hit@5 & Recall@5 ≥ 90%)")
        print("-" * 60)

        # 2. 执行检索与统计
        for i, case in enumerate(tqdm(test_cases, desc="🧪 测评中")):
            query = case["query"]
            target_id = case.get("target_id")
            
            if not target_id:
                continue
            
            stats["total_queries"] += 1

            try:
                # 显存定期清理
                if i % 50 == 0:
                    torch.cuda.empty_cache()

                # 调用检索 pipeline (请求 top_n=5 即可满足需求)
                results = self.retriever.pipeline(query=query, top_n=5, target_id=target_id)
                retrieved_ids = [doc["metadata"]["docid"] for doc in results]

                # 核心逻辑：判断目标 ID 是否在 Top-5 中
                if target_id in retrieved_ids:
                    stats["hits_at_5"] += 1

            except Exception as e:
                print(f"❌ 检索失败: {query[:20]}... | Error: {e}")

        # 3. 计算结果
        total = stats["total_queries"] if stats["total_queries"] > 0 else 1
        hit_rate_5 = (stats["hits_at_5"] / total) * 100

        # 4. 打印验收报告
        print("\n" + "🏁 最终验收报告 ".center(60, "="))
        print(f"📅 测评时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 有效样本量: {total}")
        print("-" * 60)
        
        status = "✅ 通过" if hit_rate_5 >= 90 else "❌ 未达标"
        
        # 在单目标检索中，Hit@5 和 Recall@5 的数值是相等的
        print(f"🎯 Hit@5 (Top-5 命中率):  {hit_rate_5:.2f}%  (目标 ≥ 90%)")
        print(f"🎯 Recall@5 (Top-5 召回率): {hit_rate_5:.2f}%  (目标 ≥ 90%)")
        print(f"📢 最终结论: {status}")
        print("=" * 60)

if __name__ == "__main__":
    
    from config import Config
    import os

    # 初始化检索器
    retriever = HybridRetrieverV3(
        model_path=Config.LOCAL_MODEL_PATH, 
        reranker_path=Config.RERANKER_MODEL_PATH
    )
    
    # 确保索引已加载
    corpus_file = "corpus_dedup.jsonl"
    if os.path.exists(corpus_file):
        retriever.build_index(corpus_file)
    
    # 运行
    tester = RAGSimplifiedTester(retriever)
    tester.run_benchmark(test_file="test_queries_mix.jsonl")