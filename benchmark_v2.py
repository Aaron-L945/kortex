import json
import random
import time
import re
from core.retriever_v2 import HybridRetrieverV2
from loguru import logger
from opencc import OpenCC  # pip install opencc-python-reimplemented

class RAGSamplerTester:
    def __init__(self, corpus_path: str, retriever: HybridRetrieverV2):
        self.corpus_path = corpus_path
        self.retriever = retriever
        self.all_data = []
        self.cc = OpenCC('t2s')  # 繁体转简体器

    def clean_text_for_query(self, text: str) -> str:
        """清洗 Query：去除特殊符号，保留纯文本，统一转为简体"""
        # 1. 繁转简
        text = self.cc.convert(text)
        # 2. 去除括号内的干扰信息和特殊标点
        text = re.sub(r'（.*?）|\(.*?\)|[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        return text

    def prepare_data(self):
        """流式读取，构建索引"""
        logger.info(f"正在预读数据并构建索引...")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.all_data.append(json.loads(line))
        
        # 构建索引
        self.retriever.build_index(self.corpus_path)

    def run_sample_test(self, sample_size: int = 50):
        """核心抽检逻辑（带故障诊断）"""
        samples = random.sample(self.all_data, sample_size)
        results = []
        logger.info(f"开始抽检测试，样本量: {sample_size}")

        for i, sample in enumerate(samples):
            target_id = sample['docid']
            raw_text = sample['text']
            
            # --- 优化 Query 构造 ---
            # 从文本中间截取一段较长的、无干扰的纯文本
            clean_full_text = self.clean_text_for_query(raw_text)
            if len(clean_full_text) > 60:
                query = clean_full_text[20:70]  # 取中间 50 字
            else:
                query = clean_full_text[:40]

            logger.info(f"[{i+1}/{sample_size}] 测试 Query: {query}")
            
            start_time = time.time()
            
            # --- 诊断模式：获取中间过程 ---
            # 我们直接调用 pipeline，并手动观察召回情况
            final_context = self.retriever.pipeline(
                query=query, 
                top_k=100,  # 粗排给够空间
                top_n=15,   # 精排保留 15 个锚点
                window_size=3
            )
            latency = time.time() - start_time

            # 检查目标 docid 是否命中
            retrieved_ids = [doc['metadata']['docid'] for doc in final_context]
            hit = target_id in retrieved_ids
            
            results.append({"hit": hit, "latency": latency})

            if hit:
                # 找到正确答案在结果中的位置
                pos = retrieved_ids.index(target_id) + 1
                logger.success(f"  ✅ 命中！(排在第 {pos} 位) | 耗时: {latency:.3f}s")
            else:
                # ❌ 诊断：如果没命中，看看它死在哪一步
                logger.error(f"  ❌ 未命中 ID: {target_id}")
                # 检查是否在 Top-K 粗排里（这里需要修改 retriever2 暴露一个 debug 方法，或者看日志）
                logger.debug(f"     [建议检查]：文档内容是否包含繁体？模型是否支持该领域？")

        self._summary(results)

    def _summary(self, results):
        hits = sum(1 for r in results if r['hit'])
        avg_lat = sum(r['latency'] for r in results) / len(results)
        print("\n" + "="*50)
        print(f"📊 抽样验收报告 (V3 诊断版)")
        print(f"🎯 召回成功率 (Recall): {(hits/len(results))*100:.1f}%")
        print(f"⏱️ 平均响应耗时: {avg_lat:.3f}s")
        if (hits/len(results)) < 0.5:
            print("💡 警告：召回率极低！请检查 Embedding 模型是否支持繁体，或尝试调大 Top_K。")
        print("="*50)

# --- 启动测试 ---
if __name__ == "__main__":
    from core.retriever_v2 import HybridRetrieverV2
    # 这里建议传入你的本地模型路径或 HuggingFace ID
    # 推荐使用 BGE-M3 (支持多语言和繁体)
    from config import Config
    print(f"{Config.LOCAL_MODEL_PATH=}")
    my_retriever = HybridRetrieverV2(model_path=Config.LOCAL_MODEL_PATH) 
    
    tester = RAGSamplerTester("corpus.jsonl", my_retriever)
    tester.prepare_data()
    tester.run_sample_test(sample_size=50)