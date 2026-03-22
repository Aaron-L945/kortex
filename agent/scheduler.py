"""
Agent 调度层：LlamaIndex ReActAgent + Claude claude-opus-4-6

职责：
  1. Planning：拆解复杂问题为子步骤
  2. 判断是否需要检索（调用 knowledge_retrieval 工具）
  3. 判断是否需要多轮工具调用
  4. 最终综合生成答案

支持流式输出（streaming=True）。
"""

import asyncio
from typing import AsyncIterator

from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.llms.anthropic import Anthropic as AnthropicLLM

from agent.tools import build_tools
from config import settings
from models.schemas import UserInfo


SYSTEM_PROMPT = """\
你是一个企业级知识库智能助手。

你的职责：
1. 准确理解用户的问题
2. 判断是否需要从知识库检索信息
3. 如果需要检索，使用 knowledge_retrieval 工具获取相关内容
4. 综合检索结果，给出清晰、专业、有依据的回答
5. 回答时注明信息来源

注意：
- 只回答与企业业务相关的问题
- 不确定的信息请明确告知用户
- 涉及机密信息时，请谨慎处理
- 回答语言与用户提问语言保持一致（默认中文）
"""


def build_agent(user: UserInfo) -> ReActAgent:
    """为指定用户构建一个带权限的 ReActAgent"""
    llm = AnthropicLLM(
        model="claude-opus-4-6",
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=4096,
    )
    LlamaSettings.llm = llm

    tools = build_tools(user)

    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=10,
        context=SYSTEM_PROMPT,
    )
    return agent


async def run_agent(user: UserInfo, query: str) -> str:
    """同步执行 Agent，返回完整答案"""
    agent = build_agent(user)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: agent.chat(query)
    )
    return str(response)


async def run_agent_stream(
    user: UserInfo, query: str
) -> AsyncIterator[str]:
    """
    流式执行 Agent。
    LlamaIndex ReActAgent 的 stream_chat 返回 StreamingAgentChatResponse。
    我们在 executor 中运行并逐 token yield。
    """
    agent = build_agent(user)

    def _stream():
        return agent.stream_chat(query)

    loop = asyncio.get_event_loop()
    streaming_response = await loop.run_in_executor(None, _stream)

    # streaming_response.response_gen 是同步生成器，在线程中逐块读取
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _consume():
        try:
            for token in streaming_response.response_gen:
                asyncio.run_coroutine_threadsafe(queue.put(token), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    import threading
    threading.Thread(target=_consume, daemon=True).start()

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token
