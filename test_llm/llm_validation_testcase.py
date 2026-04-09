# ==========================================
# 1. 严谨的 Prompt 定义
# ==========================================
PROMPT_TEMPLATE = """
你是一个严格的文档核查员。你的任务是根据【参考资料】提取信息并回答问题。

### 核心铁律（违反任何一条将导致任务失败）：
1. 【原文提取】：**严禁对数值、日期、金额、百分比进行任何形式的改写、取整、转换或四舍五入。** 必须 100% 摘抄原文符号。
2. 【禁止推理】：禁止根据资料进行逻辑推导。如果资料说“A和B相关”，你不能回答“A影响了B”。
3. 【原子化引用】：每一个事实陈述后必须紧跟引用的 [docid]。禁止在段落末尾汇总标注。
4. 【死板拒答】：若资料中未出现问题的主体（实体词），或资料无法直接推导答案，必须统一回复：“抱歉，根据已知资料无法回答该问题。”，不得输出任何其他文字。
5.  请直接引用资料中的原始数值，不要进行任何形式的加总、平均或换算。

### 回答结构（严格执行）：
[结论]：用一句话直接回答问题。
[证据]：摘抄资料中的关键原句，必须包含原始数值。 [docid]
[详情]：若有多个维度，请分点列出。 [docid]

### 参考资料：
{context}

### 用户问题：
{query}

### 回答模板：
[结论]：(一句话总结)
[证据]：(摘抄原文关键句) [docid]
[详情]：(展开描述，必须包含原始数值) [docid]
"""

# ==========================================
# 2. 增强版 RAG 引擎
# ==========================================
import re
import time
import json
import pandas as pd
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langchain_core.messages import SystemMessage, HumanMessage

# import re
# import time
# import json
# import pandas as pd
# from tqdm import tqdm
# from loguru import logger
# from concurrent.futures import ThreadPoolExecutor, TimeoutError

class RAGAnswerEngine:
    def __init__(self, retriever, llm_client, request_timeout=60):
        self.retriever = retriever
        self.llm = llm_client
        self.request_timeout = request_timeout  # 单次 LLM 推理的最高容忍时间


    def _verify_metrics(self, answer, docs, category="unknown"):
        """
        最终优化版审计逻辑
        修复点：
        1. 细化拒答判定：只有在没有引用且包含拒答词时才判定为拒绝。
        2. 增强引用提取：解决 re.findall 分组导致的 ID 截断问题。
        3. 优化一致性匹配：排除引用干扰，提高数字比对健壮性。
        """
        
        # --- 0. 预处理：提取引用 (使用非捕获组确保提取完整 ID，如 123#1) ---
        # 先提取带括号的全文本，再剥离括号，避免 findall 分组捕获问题
        raw_citations = re.findall(r"\[\d+(?:#\d+)?\]", answer)
        claimed_ids = [c.strip("[]") for c in raw_citations]
        citation_count = len(claimed_ids)

        # --- 1. 拒答判定 (逻辑升级：有引用即代表尝试回答) ---
        refusal_keywords = ["抱歉", "无法回答", "没有提到", "未提供", "不详"]
        # 只有当：含有拒答词 且 引用数为0 且 长度较短时，才判定为真拒答
        is_refusal_detected = any(word in answer for word in refusal_keywords)
        
        # 核心修正：如果模型给出了引用，说明它在资料中找到了内容，不应判定为 is_refusal
        actual_refusal = is_refusal_detected and citation_count == 0

        # --- 2. 处理 NEG (无关类) 的判定 ---
        if category == "neg":
            return {
                "is_refusal": actual_refusal,
                "hallucination": citation_count > 0, # NEG 组如果写了引用，反而是幻觉
                "consistency_ok": True,
                "citation_count": citation_count,
                "status": "success"
            }

        # --- 3. 处理 POS/BLUR 的判定 ---
        if actual_refusal:
            return {
                "is_refusal": True, 
                "hallucination": False, 
                "consistency_ok": True, 
                "citation_count": 0, 
                "status": "success"
            }

        # --- 4. 一致性检查 (数值校验) ---
        # 剔除引用 ID 干扰，避免将文档 ID 误认为事实数字
        clean_answer_for_audit = re.sub(r"\[\d+(?:#\d+)?\]", "", answer)
        
        # 提取数字（含百分比）
        numbers_in_ans = re.findall(r"\d+(?:\.\d+)?%?", clean_answer_for_audit)
        # 预处理源码：合并内容并统一剔除千分位逗号
        combined_source = "".join([d['content'] for d in docs]).replace(",", "")
        
        consistency_risk = False
        for num in numbers_in_ans:
            clean_n = num.replace(",", "").replace("%", "")
            # 长度 > 1 的数字（排除掉 0-9 或单纯的年份 2024）才进行深度校验
            if len(clean_n) > 1: 
                # 校验逻辑：
                # 1. 字符串直接包含
                # 2. 或者 整数部分包含（处理 85.5% 被模型简写为 85.5 的情况）
                if clean_n not in combined_source and clean_n.split('.')[0] not in combined_source:
                    # 特殊逻辑：如果是计算出来的总和数字，在基础 RAG 中通常会判错
                    # 除非原文确实存在该数字
                    consistency_risk = True
                    break

        # --- 5. 引用幻觉检查 (ID 是否合法) ---
        # 这里的 source_ids 通常只有主 ID，需要处理带 # 的匹配
        source_ids = [str(d['metadata']['docid']) for d in docs]
        id_hallucination = False
        for cid in claimed_ids:
            main_id = cid.split('#')[0]
            if main_id not in source_ids:
                id_hallucination = True
                break

        return {
            "is_refusal": False,
            "hallucination": id_hallucination,
            "consistency_ok": not consistency_risk,
            "citation_count": citation_count,
            "status": "success"
        }

    def generate_answer(self, query: str, category: str = "pos"):
        """
        带超时监控和类别感知的生成逻辑
        category: pos(有答案), neg(无关), blur(模糊)
        """
        start_time = time.time()
        
        # 1. 检索阶段
        try:
            results = self.retriever.pipeline(query, top_n=5)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return "检索异常", {"status": "error", "is_refusal": True, "consistency_ok": False, "hallucination": False, "citation_count": 0}

        # 2. LLM 推理阶段 (带线程池超时控制)
        context = "\n\n".join([f"--- 文档 [{d['metadata']['docid']}] ---\n{d['content']}" for d in results])
        logger.warning(f"{context=}")
        messages = [
            SystemMessage(content=PROMPT_TEMPLATE.format(context=context, query=query)),
            HumanMessage(content=query)
        ]

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.llm.invoke, messages)
            try:
                response = future.result(timeout=self.request_timeout)
                answer = response.content.strip()
                duration = time.time() - start_time
                
                # 3. 调用增强版审计逻辑
                m = self._verify_metrics(answer, results, category)
                m["status"] = "success"
                m["duration"] = duration
                return answer, m
            
            except TimeoutError:
                logger.error(f"⚠️ 任务超时: {query[:15]}...")
                return "抱歉，系统响应超时。", {"status": "timeout", "is_refusal": True, "consistency_ok": False, "hallucination": False, "citation_count": 0}
            except Exception as err:
                print(f"{query[:15]} --- Got Exception: {err=}")
                return "抱歉，查询被拒绝。", {"is_refusal": True, "hallucination": False, "consistency_ok": False, "citation_count": 0, "status": "timeout"}
    
    def _print_audit_table(self, query, m, duration):
        logger.info(f"📊 审计报告 | Query: {query[:15]}...")
        print(f"| 幻觉率: {'❌ 风险' if m['hallucination'] else '✅ 通过'} "
              f"| 证据一致性: {'✅ 达标' if m['consistency_ok'] else '❌ 风险'} "
              f"| 引用数: {m['citation_count']} "
              f"| 耗时: {duration:.2f}s |")

# ==========================================
# 3. 执行脚本 (Main)
# ==========================================
if __name__ == "__main__":
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 添加到 sys.path
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from core.retriever_with_testcase import HybridRetrieverV3
    from config import Config
    from llm_service import get_qwen_llm

    # 1. 初始化检索器（带日志追踪）
    logger.info("--- 步骤 1: 加载检索模型 ---")
    retriever = HybridRetrieverV3(
        model_path=Config.LOCAL_MODEL_PATH, 
        reranker_path=Config.RERANKER_MODEL_PATH
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
    engine = RAGAnswerEngine(retriever=retriever, llm_client=llm)
    
    # 测试 Query
    test_query = "谁写了《网络独立宣言》？"
    final_res = engine.generate_answer(test_query)
    
    print("\n" + "="*50)
    print(f"最终回答：\n{final_res}")
    print("="*50)