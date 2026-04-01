import os
import time
import json
from typing import List, Dict
from loguru import logger

# 基础依赖
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 假设你的 retriever 和 config 已经在 core 文件夹中
# from core.retriever_with_testcase import HybridRetrieverV3
# from config import Config

# ==========================================
# 1. 严谨的 Prompt 定义
# ==========================================
PROMPT_TEMPLATE = """
你是一个高度严谨的知识库问答助手。你的唯一任务是根据提供的【参考资料】回答用户问题。

### 严格约束：
1. **证据为本**：仅使用【参考资料】中的信息。禁止引入任何外部知识或常识。
2. **拒答原则**：如果【参考资料】中没有直接、明确的答案，或者资料与问题无关，必须回答：“抱歉，根据已知资料无法回答该问题。”，禁止进行任何推测。
3. **引用规范**：在回答的每个要点末尾，必须标注对应文档的 [docid]。
4. **非检索限制**：禁止解释、禁止寒暄、禁止输出与资料无关的引导性文字。

### 参考资料：
{context}

### 用户问题：
{query}

### 回答格式：
[结论] + [证据陈述] + [引用ID]
"""

# ==========================================
# 2. 增强版 RAG 引擎
# ==========================================
import re
from loguru import logger

class RAGAnswerEngine:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm = llm_client
        self.min_score_threshold = 0.4
        
        # 统计计数器，用于计算长期指标
        self.stats = {
            "total_queries": 0,
            "refusal_count": 0,
            "has_citation_count": 0
        }

    def _format_context(self, docs):
        return "\n\n".join([f"--- 文档 [{d['metadata']['docid']}] ---\n{d['content']}" for d in docs])

    def _audit_response(self, query, answer, docs, duration):
        """
        自动化审计：对应 5 项验收标准
        """
        # 1. 拒答判定
        is_refusal = "抱歉" in answer or "无法回答" in answer
        
        # 2. 引用检查 (正则匹配 [docid] 或 [数字#数字])
        citations = re.findall(r"\[\d+(?:#\d+)?\]", answer)
        has_citation = len(citations) > 0
        
        # 3. 非检索内容占比估算 (简单逻辑：检查是否包含大模型常见废话词汇)
        hallucination_triggers = ["根据我所知", "一般来说", "常识告诉我", "希望对你有帮助"]
        contains_external_info = any(word in answer for word in hallucination_triggers)
        
        # 4. 打印审计日志
        logger.info("=" * 30 + " 🚩 验收审计报告 " + "=" * 30)
        print(f"| 指标项             | 运行值                | 验收标准           | 状态   |")
        print(f"|--------------------|-----------------------|--------------------|--------|")
        print(f"| 1. 拒答状态        | {'已拒答' if is_refusal else '已回答'}             | 准确认 > 95%       | {'-' if is_refusal else 'OK'}     |")
        print(f"| 2. 引用准确率      | 发现 {len(citations)} 处引用        | 引用率 > 90%       | {'✅' if (has_citation or is_refusal) else '❌'}     |")
        print(f"| 3. 非检索内容占比  | {'存在风险' if contains_external_info else '极低'}               | 占比 < 5%          | {'✅' if not contains_external_info else '⚠️'}    |")
        print(f"| 4. 响应耗时        | {duration:.2f}s               | -                  | -      |")
        
        # 记录统计
        self.stats["total_queries"] += 1
        if is_refusal: self.stats["refusal_count"] += 1
        if has_citation: self.stats["has_citation_count"] += 1
        
        # 打印证据一致性核对 (人工抽查辅助)
        if not is_refusal:
            logger.debug(f"证据核对：当前回答引用的 ID 列表为 {list(set(citations))}")
        logger.info("=" * 76)

    def generate_answer(self, query: str):
        start_time = time.time()
        
        # 1. 检索
        results = self.retriever.pipeline(query, top_n=5)
        
        # 2. 无结果处理
        if not results:
            ans = "抱歉，根据已知资料无法回答该问题。"
            self._audit_response(query, ans, [], time.time() - start_time)
            return ans

        # 3. LLM 生成
        context = self._format_context(results)
        messages = [
            SystemMessage(content=PROMPT_TEMPLATE.format(context=context, query=query)),
            HumanMessage(content=query)
        ]
        
        try:
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            # 4. 自动审计并打印
            duration = time.time() - start_time
            self._audit_response(query, answer, results, duration)
            
            return answer
        except Exception as e:
            logger.error(f"LLM 运行异常: {e}")
            return "系统异常"

# ==========================================
# 3. 执行脚本 (Main)
# ==========================================
if __name__ == "__main__":
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
    test_query = "領展的女性董事比例是多少？"
    final_res = engine.generate_answer(test_query)
    
    print("\n" + "="*50)
    print(f"最终回答：\n{final_res}")
    print("="*50)