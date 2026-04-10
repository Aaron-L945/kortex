import sys
import os
import time
from loguru import logger

# 导入你的核心组件
from llm_service import get_qwen_llm

# 导入 LangChain 相关组件
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from test_llm.llm_validation_testcase import RAGAnswerEngine
from langchain_core.prompts import load_prompt
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

from core.retriever_with_testcase import HybridRetrieverV3
from config import Config
from core.tools import TOOLS, get_weather


# --- 初始化 (全局) ---

logger.info("--- 步骤 1: 加载检索模型 ---")
retriever = HybridRetrieverV3(
    model_path=Config.LOCAL_MODEL_PATH, reranker_path=Config.RERANKER_MODEL_PATH
)
corpus_file = "corpus_dedup.jsonl"
if os.path.exists(corpus_file):
    retriever.build_index(corpus_file)

logger.info("--- 步骤 2: 连接本地 LLM 服务 ---")
llm = get_qwen_llm(streaming=True)
router_llm = get_qwen_llm(streaming=False)

# --- 智能路由组件 ---


class SmartRouter:
    def __init__(self, llm):
        self.route_prompt = load_prompt("prompts/router/intent_classifier.yaml")
        self.route_chain = self.route_prompt | llm | StrOutputParser()

    def decide(self, user_input, history=""):
        # 修改点：invoke 字典中补齐 'history' 变量
        res = self.route_chain.invoke({"input": user_input, "history": history}).strip()

        if res.startswith("TOOL:"):
            return "TOOL", res
        return ("SEARCH" if "SEARCH" in res.upper() else "CHAT"), res


# --- 核心运行逻辑 ---
@traceable(run_type="chain", name="Main_Conversation_Loop") # 关键：声明这是一个整体链
def run_conversation():
    # 1. 初始化窗口记忆 (k=5)
    memory = ConversationBufferWindowMemory(k=3, memory_key="history", return_messages=True, input_key="input")

    # 2. 将工具绑定到 LLM (语义驱动的核心)
    llm_with_tools = llm.bind_tools(TOOLS)

    # 3. 初始化 RAG 引擎
    engine = RAGAnswerEngine(retriever=retriever, llm_client=llm)

    logger.info("--- 语义驱动引擎已启动 ---")

    while True:
        user_query = input("\n[用户]: ").strip()
        if not user_query or user_query.lower() in ["exit", "quit"]:
            break

        start_time = time.time()

        # A. 加载历史
        history_data = memory.load_memory_variables({})["history"]

        # 2. 核心修复逻辑：确保 history 是 list
        if isinstance(history_data, str):
            # 如果是字符串且为空，给空列表；如果不为空，说明配置有误，强制转为消息格式（或者检查初始化）
            logger.warning("Memory 返回了字符串而非消息列表，请检查 return_messages=True 配置")
            history_messages = [] 
        else:
            history_messages = history_data

        # B. 构造当前请求
        system_msg = SystemMessage(
            content="你是一个全能助手。根据用户需求，自主选择调用工具或直接回答。"
        )
        current_msg = HumanMessage(content=user_query)

        # C. 第一次模型调用：决定做什么
        prompt = [system_msg] + history_messages + [current_msg]
        ai_msg = llm_with_tools.invoke(
            prompt, 
            config={"run_name": "Main_Agent_Decision"} 
        )

        response_content = ""

        # D. 语义判断：是否触发了工具？
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                logger.info(f"语义触发工具: {t_name} 参数: {t_args}")

                if t_name == "get_weather":
                    result = get_weather.invoke(t_args, config={"run_name": "Call_Weather_API"})
                    # 润色输出
                    response_content = llm.predict(
                        f"用户问：{user_query}\n天气结果：{result}\n请友好回复。"
                    )

                elif t_name == "search_knowledge_base":
                    # 拦截并执行你原有的 RAG 逻辑
                    history_str = memory.load_memory_variables({})[
                        "history"
                    ]  # 转文本格式给 RAG
                    response_content, _ = engine.generate_answer(
                        t_args.get("query", user_query), history=str(history_str)
                    )
        else:
            # 纯聊天：模型直接给出了回答
            response_content = ai_msg.content

        # E. 更新记忆
        memory.save_context({"input": user_query}, {"output": response_content})

        print("-" * 30)
        print(f"[AI]: {response_content}")
        print(f"(耗时: {time.time() - start_time:.2f}s)")
        print("-" * 30)


if __name__ == "__main__":
    run_conversation()
