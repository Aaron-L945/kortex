from loguru import logger
import os
import json
from tqdm import tqdm
import pandas as pd
import time
import random
from llm_validation_testcase import RAGAnswerEngine


def run_batch_test(engine, input_file, output_file="final_audit_results.jsonl"):
    # 1. 加载带标签的测试集
    if not os.path.exists(input_file):
        logger.error(f"找不到文件: {input_file}")
        return
    with open(input_file, "r", encoding="utf-8-sig") as f:
        all_queries = [json.loads(line) for line in f]

    # 2. 断点续传检查
    processed_queries = set()
    last_id = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8-sig") as f_exist:
            for line in f_exist:
                try:
                    data = json.loads(line)
                    processed_queries.add(data["query"])
                    last_id = max(last_id, data.get("id", 0))
                except:
                    continue

    pending_tasks = [
        item for item in all_queries if item["query"] not in processed_queries
    ]
    if not pending_tasks:
        logger.success("全部任务已完成。")
    else:
        logger.info(f"🚀 剩余任务: {len(pending_tasks)} 条")

        # 3. 循环测试
        with open(output_file, "a", encoding="utf-8-sig") as f_out:
            for item in tqdm(pending_tasks):
                query = item["query"]
                cat = item.get("category", "pos")  # 读取标签
                last_id += 1

                # 调用带 category 的引擎
                answer, metrics = engine.generate_answer(query, category=cat)

                record = {
                    "id": last_id,
                    "query": query,
                    "category": cat,
                    "answer": answer,
                    **metrics,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                # 随机冷却 1-10s
                time.sleep(random.uniform(1, 10))

    # --- 4. 分层验收统计看板 ---
    full_results = []
    with open(output_file, "r", encoding="utf-8-sig") as f_res:
        for line in f_res:
            full_results.append(json.loads(line))

    df = pd.DataFrame(full_results)

    print("\n" + " 🏆 RAG 系统分层验收报告 ".center(60, "="))
    print(f"| 类别 | 样本 | 核心指标 (准确率/拒答率) | 一致性 | 状态 |")
    print(f"|------|------|-------------------------|--------|------|")

    for cat in ["pos", "neg", "blur"]:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue

        count = len(sub)
        success_sub = sub[sub["status"] == "success"]

        # 判定标准：
        # POS: 必须有回答且带引用
        # NEG: 必须拒答且无引用
        if cat == "pos":
            core_metric = (success_sub["citation_count"] > 0).sum() / count
        elif cat == "neg":
            core_metric = (sub["is_refusal"] == True).sum() / count
        else:
            core_metric = (
                success_sub["citation_count"] > 0
            ).sum() / count  # 模糊项暂定同 POS

        consist_rate = success_sub["consistency_ok"].sum() / count

        status = "✅" if core_metric >= 0.9 and consist_rate >= 0.9 else "❌"
        if cat == "neg":
            status = "✅" if core_metric >= 0.95 else "❌"

        print(
            f"| {cat.upper():<4} | {count:>4} | {core_metric*100:>21.2f}% | {consist_rate*100:>5.2f}% |  {status}  |"
        )

    print("=" * 66)
    # 统计总耗时等信息...
    avg_dur = df["duration"].mean() if "duration" in df else 0
    print(
        f"平均响应时间: {avg_dur:.2f}s  | 总样本: {len(df)} | 超时数: {len(df[df['status']=='timeout'])}"
    )


# ==========================================
# 启动批量测试
# ==========================================
if __name__ == "__main__":
    # ... 初始化 retriever 和 llm 代码 ...
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 添加到 sys.path
    if project_root not in sys.path:
        sys.path.append(project_root)
    from core.retriever_with_testcase import HybridRetrieverV3
    from config import Config
    from llm_service import get_qwen_llm

    logger.info("--- 步骤 1: 加载检索模型 ---")
    retriever = HybridRetrieverV3(
        model_path=Config.LOCAL_MODEL_PATH, reranker_path=Config.RERANKER_MODEL_PATH
    )

    # 2. 构建/加载索引
    corpus_file = "corpus_dedup.jsonl"
    if os.path.exists(corpus_file):
        logger.info(f"--- 步骤 2: 构建索引 (文件: {corpus_file}) ---")
        build_start = time.time()
        retriever.build_index(corpus_file)
        logger.info(f"索引加载成功，耗时: {time.time() - build_start:.2f}s")
    else:
        logger.error(f"未找到语料文件: {corpus_file}")

    # 3. 初始化本地 LLM
    logger.info("--- 步骤 3: 连接本地 LLM 服务 ---")
    llm = get_qwen_llm(streaming=False)

    # 4. 运行引擎
    engine = RAGAnswerEngine(retriever=retriever, llm_client=llm, request_timeout=60)

    # 执行 100 条批量测试
    run_batch_test(
        engine,
        input_file="/root/tests/aaron/kortex/test_llm/matched_results.jsonl",
        output_file="/root/tests/aaron/kortex/test_llm/final_audit_results.jsonl",
    )
