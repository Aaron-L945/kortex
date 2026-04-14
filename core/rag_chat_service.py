import os
import time
import sys
from typing import Dict, List, Tuple
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from build_milvus_index import EnterpriseSecureRAG

# ==========================================
# 0. Loguru 配置
# ==========================================
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# ==========================================
# 1. LLM 加载配置
# ==========================================
def get_qwen_llm(streaming=True):
    return ChatOpenAI(
        model="qwen3-max",
        openai_api_key="sk-b51aac8fea8dcb4fb574275c123f960e",
        openai_api_base="https://apis.iflow.cn/v1",
        temperature=0,
        streaming=streaming,
        verbose=False
    )

# ==========================================
# 2. 对话服务类 (优化版)
# ==========================================
class SecureChatService:
    def __init__(self):
        logger.info("正在初始化企业级安全 RAG 服务...")
        self.rag_backend = EnterpriseSecureRAG()
        self.llm = get_qwen_llm(streaming=True) # 确保开启流式
        logger.success("系统初始化成功!")

    def ask_question_stream(self, query: str, user_context: Dict):
        """
        使用流式生成答案，解决 29 秒延迟焦虑
        """
        # --- 阶段 1: 向量检索 (减少 Top-K) ---
        start_retrieve = time.perf_counter()
        # 注意：这里需要在你的 secure_search 方法里把 limit 改小，或者在此处传参
        # 建议直接在 secure_search 的实现里把 RETRIEVAL_TOP_K 设为 5
        search_results = self.rag_backend.secure_search(query, user_context)
        retrieve_time = time.perf_counter() - start_retrieve
        
        # --- 阶段 2: 构建上下文 ---
        context_parts = []
        sources = set()
        for hits in search_results:
            # 即使检索返回了更多，我们也只取前 5 个最相关的
            for hit in hits[:10]: 
                content = hit.entity.get('text')
                fname = hit.entity.get('file_name')
                context_parts.append(f"文件 [{fname}]:\n{content}")
                sources.add(fname)
        
        if not context_parts:
            yield "未发现相关参考资料。", []
            return

        context_text = "\n\n".join(context_parts)
        
        # --- 阶段 3: LLM 流式生成 ---
        system_prompt = SystemMessage(content="你是一个企业助手。请严格基于资料回答，不得编造。")
        user_message = HumanMessage(content=f"【参考资料】:\n{context_text}\n\n【用户问题】: {query}")

        logger.info(f"检索完成({retrieve_time:.2f}s)，包含 {len(context_parts)} 个精选片段。开始流式生成...")
        
        full_answer = ""
        start_gen = time.perf_counter()
        first_token_time = None

        # 使用 stream 方法替代 invoke
        for chunk in self.llm.stream([system_prompt, user_message]):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_gen
                logger.debug(f"首字产出耗时: {first_token_time:.2f}s")
            
            content = chunk.content
            full_answer += content
            yield content, list(sources) # 实时返回每一个字

        total_gen_time = time.perf_counter() - start_gen
        logger.success(f"回答生成完毕 | 总耗时: {retrieve_time + total_gen_time:.2f}s | 片段数: {len(context_parts)}")

# ==========================================
# 3. 运行测试 (流式显示)
# ==========================================
if __name__ == "__main__":
    chat_svc = SecureChatService()
    u_context = {"user_id": "aaron_admin", "dept": "Tech", "role": "internal"}
    
    print(f"\nQ: VSC 如何做升级？\nA: ", end="", flush=True)
    
    last_sources = []
    # 通过生成器获取流式输出
    for text_chunk, sources in chat_svc.ask_question_stream("VSC 如何做升级？", u_context):
        print(text_chunk, end="", flush=True)
        last_sources = sources
    
    print(f"\n\n[参考文件]: {last_sources}")