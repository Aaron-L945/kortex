import json
import random
import time
import torch
from loguru import logger
from core.retriever_v3 import HybridRetrieverV3

class RAGVisualTester:
    def __init__(self, corpus_path: str, retriever: HybridRetrieverV3):
        self.corpus_path = corpus_path
        self.retriever = retriever
        self.all_data = []

    def prepare_data(self):
        """初始化索引并加载原始语料用于比对"""
        # 内部会自动判断加载磁盘索引还是重建
        self.retriever.build_index(self.corpus_path)
        logger.info(f"读取原始语料库以供性能比对...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                self.all_data.append(json.loads(line))

    def run_detail_test(self, sample_size: int = 100):
        if not self.all_data:
            logger.error("请先执行 prepare_data() 加载数据")
            return

        samples = random.sample(self.all_data, sample_size)
        top1_hits = 0
        top5_hits = 0
        total = len(samples)

        print("\n" + " RAG 检索精度深度分析 (精排增强版) ".center(70, "="))

        for i, sample in enumerate(samples):
            target_id = sample["docid"]
            # 构造 Query：取原句中间一段
            raw_text = sample["text"]
            query = (raw_text[30:80] if len(raw_text) > 80 else raw_text[:50])

            # 执行检索流水线 (包含 BM25 + Vector + Rerank + Window)
            # 注意：top_n=5 表示我们要看最终输出的 5 个块
            final_context = self.retriever.pipeline(query=query, top_n=5)
            
            # 提取检索出来的 docid 列表
            retrieved_ids = [doc["metadata"]["docid"] for doc in final_context]

            # 统计命中情况
            is_top1 = False
            is_top5 = False

            if retrieved_ids:
                if retrieved_ids[0] == target_id:
                    is_top1 = True
                    top1_hits += 1
                
                if target_id in retrieved_ids:
                    is_top5 = True
                    top5_hits += 1

            # 打印 Bad Case (只打印非 Top-1 的，方便分析)
            if not is_top1:
                print(f"\n🔍 [样例 {i+1}] Query: {query[:50]}...")
                if is_top5:
                    pos = retrieved_ids.index(target_id) + 1
                    print(f"【评价】: ⚠️ 已召回但排序靠后 (第 {pos} 位)")
                    print(f"   - 目标内容: {raw_text[:70]}...")
                    print(f"   - Top-1内容: {final_context[0]['content'][:70]}...")
                elif not retrieved_ids:
                    print(f"【评价】: ❌ 结果为空")
                else:
                    print(f"【评价】: ❌ 目标未在 Top-5 中出现")
                    print(f"   - 目标内容: {raw_text[:70]}...")
                    print(f"   - Top-1内容: {final_context[0]['content'][:70]}...")

        # 最终汇总
        print("\n" + "=" * 70)
        print(f"📊 测试总结 (样本量: {total})")
        print(f"🎯 Top-1 准确率 (Precision@1): {(top1_hits/total)*100:.2f}%")
        print(f"🎯 Top-5 召回率 (Recall@5):    {(top5_hits/total)*100:.2f}%")
        print(f"💡 提示: 如果 Top-1 依然较低，建议检查 Reranker 模型路径或增加 RRF 中 BM25 的权重。")
        print("=" * 70)


if __name__ == "__main__":
    # 假设你的配置类里有这些路径
    # Config.LOCAL_MODEL_PATH = "BAAI/bge-m3"
    # Config.RERANKER_MODEL_PATH = "BAAI/bge-reranker-v2-m3"
    from config import Config

    # 1. 初始化集成精排的检索器
    my_retriever = HybridRetrieverV3(
        model_path=Config.LOCAL_MODEL_PATH,
        reranker_path=getattr(Config, "RERANKER_MODEL_PATH", None) 
    )

    # 2. 运行测试
    tester = RAGVisualTester("corpus.jsonl", my_retriever)
    tester.prepare_data()
    
    # 3. 建议先跑 100 个看看分布
    tester.run_detail_test(sample_size=100)