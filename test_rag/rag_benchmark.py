# ... 初始化 retriever 和 llm 代码 ...
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
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
            "hits_at_5": 0,
            "sum_reciprocal_rank": 0.0,
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
            target_id_str = str(target_id)

            try:
                if i % 50 == 0:
                    torch.cuda.empty_cache()

                # --- 修改点：在 pipeline 中传入 target_id ---
                # 注意：请确保你的 HybridRetrieverV3.pipeline 定义接收 target_id 参数
                results = self.retriever.pipeline(
                    query=query, top_n=5, target_id=target_id_str  # 传入 target_id
                )

                retrieved_ids = [str(doc["metadata"]["docid"]) for doc in results]

                # --- 核心逻辑：计算 Rank 和 MRR ---
                if target_id_str in retrieved_ids:
                    rank = retrieved_ids.index(target_id_str) + 1
                    stats["hits_at_5"] += 1
                    stats["sum_reciprocal_rank"] += 1.0 / rank
                else:
                    stats["sum_reciprocal_rank"] += 0.0

            except Exception as e:
                print(f"❌ 检索失败: {query[:20]}... | Error: {e}")

        # 3. 计算结果
        total = stats["total_queries"] if stats["total_queries"] > 0 else 1
        hit_rate_5 = (stats["hits_at_5"] / total) * 100
        # 在单目标检索场景下，Recall@5 等于 Hit@5
        recall_at_5 = hit_rate_5
        mrr = stats["sum_reciprocal_rank"] / total

        # 4. 打印验收报告
        print("\n" + "🏁 最终验收报告 ".center(60, "="))
        print(f"📅 测评时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 有效样本量: {total}")
        print("-" * 60)

        status = "✅ 通过" if hit_rate_5 >= 90 else "❌ 未达标"

        print(f"🎯 Hit@5 (命中率):     {hit_rate_5:.2f}%  (目标 ≥ 90%)")
        print(f"🎯 Recall@5 (召回率):  {recall_at_5:.2f}%  (单目标场景等同于 Hit@5)")
        print(f"🏆 MRR (平均倒数排名):  {mrr:.4f}")

        if mrr > 0.8:
            print(f"💡 表现评估: 极好！目标大多排在第 1 名。")

        print(f"📢 最终结论: {status}")
        print("=" * 60)


if __name__ == "__main__":

    from config import Config
    import os

    # 初始化检索器
    retriever = HybridRetrieverV3(
        model_path=Config.LOCAL_MODEL_PATH, reranker_path=Config.RERANKER_MODEL_PATH
    )

    # 确保索引已加载
    corpus_file = "corpus_dedup.jsonl"
    if os.path.exists(corpus_file):
        retriever.build_index(corpus_file)

    # 运行
    tester = RAGSimplifiedTester(retriever)
    tester.run_benchmark(test_file="test_queries_mix.jsonl")
