import asyncio
from loguru import logger

from core.model_pool import router as llm_router
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class SecureChatService:
    def __init__(self, rag_backend):
        self.rag_backend = rag_backend

    async def ask_question_stream(self, query: str, user_context: dict):
        # 1. 检索 RAG 资料
        search_results = await asyncio.to_thread(
            self.rag_backend.secure_search, query, user_context
        )

        context_parts = []
        sources = set()
        for hits in search_results:
            for hit in hits[:5]:
                context_parts.append(hit.entity.get("text"))
                sources.add(hit.entity.get("file_name"))

        source_list = list(sources)
        context_text = "\n".join(context_parts)

        # 2. 调度 LLM 节点
        node = llm_router.get_available_node()

        # 3. 在信号量保护下调用
        async with node.semaphore:
            logger.info(f"🚀 调度成功 | 节点: {node.name} | 模型: {node.model_name}")

            llm = ChatOpenAI(
                model=node.model_name,
                openai_api_key=node.api_key,
                openai_api_base=node.url,  # 这里 LangChain 会自动补全 /chat/completions
                streaming=True,
                temperature=0.1,
            )

            messages = [
                SystemMessage(content=f"请基于资料回答：\n{context_text}"),
                HumanMessage(content=query),
            ]

            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content, source_list
