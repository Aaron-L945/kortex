import asyncio
import json
import hashlib
from loguru import logger

from core.model_pool import router as llm_router
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from core.semantic_cache import SemanticCache

class SecureChatService:
    def __init__(self, rag_backend):
        self.rag_backend = rag_backend
        # 初始化语义缓存（复用 RAG 后端的 redis 和 embedding 模块）
        self.semantic_cache = SemanticCache(
            redis_client=rag_backend.emb_cache_manager.redis,
            embedding_manager=rag_backend.emb_cache_manager
        )

#     import json
# import hashlib
# from loguru import logger
# from langchain_openai import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage
# 假设 llm_router 已在外部定义并导入

    async def ask_question_stream(self, query: str, user_context: dict):
        # --- [1] 生成 Cache Key ---
        # 建议加上前缀，区分 Embedding 缓存和答案缓存
        q_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()
        cache_key = f"full_ans:{q_hash}"

        # --- [2] 尝试从语义缓存直接获取 (精准匹配层) ---
        try:
            cached_res = await self.rag_backend.emb_cache_manager.redis.get(cache_key)
            if cached_res:
                logger.info(f"🔥 [ULTRA HIT] 语义缓存完全命中: {query[:20]}...")
                data = json.loads(cached_res)
                # 模拟流式：直接 yield 整个答案
                yield data['answer'], data['sources']
                return 
        except Exception as e:
            logger.warning(f"⚠️ 读取语义缓存失败: {e}")

        # --- [3] 缓存未命中：执行 RAG 检索 ---
        # 这一步已经受惠于你之前的 Embedding Cache
        search_results = await self.rag_backend.secure_search(query, user_context)

        context_parts = []
        sources = set()
        for hits in search_results:
            for hit in hits[:5]:
                # 兼容实体获取
                text = hit.entity.get("text") if hasattr(hit, "entity") else hit.get("text")
                fname = hit.entity.get("file_name") if hasattr(hit, "entity") else hit.get("file_name")
                context_parts.append(text)
                sources.add(fname)

        source_list = list(sources)
        context_text = "\n".join(context_parts)

        # --- [4] 调度 LLM 节点并生成答案 ---
        node = llm_router.get_available_node()
        
        # 用于收集完整回答，以便存入缓存
        full_answer_content = []

        async with node.semaphore:
            logger.info(f"🚀 调度成功 | 节点: {node.name} | 模型: {node.model_name}")

            llm = ChatOpenAI(
                model=node.model_name,
                openai_api_key=node.api_key,
                openai_api_base=node.url,
                streaming=True,
                temperature=0.1,
            )

            messages = [
                SystemMessage(content=f"请基于资料回答：\n{context_text}"),
                HumanMessage(content=query),
            ]

            async for chunk in llm.astream(messages):
                if chunk.content:
                    # 实时累加内容
                    full_answer_content.append(chunk.content)
                    # 实时推送给前端
                    yield chunk.content, source_list

        # --- [5] 异步存入语义缓存 (关键步骤) ---
        if full_answer_content:
            try:
                complete_answer = "".join(full_answer_content)
                # 存储结构：答案 + 来源文档列表
                cache_payload = {
                    "answer": complete_answer,
                    "sources": source_list
                }
                # 设置过期时间为 24 小时 (86400秒)，可根据 1TB 内存剩余空间调整
                await self.rag_backend.emb_cache_manager.redis.set(
                    cache_key, 
                    json.dumps(cache_payload), 
                    ex=86400 
                )
                logger.debug(f"💾 结果已缓存: {cache_key}")
            except Exception as e:
                logger.error(f"❌ 写入语义缓存异常: {e}")