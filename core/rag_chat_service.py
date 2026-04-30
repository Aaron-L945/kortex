import asyncio
import json
import hashlib
from loguru import logger
import yaml

from core.model_pool import router as llm_router
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.semantic_cache import SemanticCache
from core.embedding_manager import EmbeddingCacheManager

class SecureChatService:
    def __init__(self, rag_backend):
        self.rag_backend = rag_backend
        self.emb_cache_manager = EmbeddingCacheManager(rag_backend)  # 复用 RAG 后端的模型实例    
        # 初始化语义缓存（复用 RAG 后端的 redis 和 embedding 模块）
        self.semantic_cache = SemanticCache(
            redis_client=self.emb_cache_manager.redis,
            embedding_manager=self.emb_cache_manager
        )
        self.prompt_template = self._load_prompt_template()
        logger.info("SecureChatService initialized and prompt_template loaded.")

    def _load_prompt_template(self):
        with open("/home/aaron/kortex/prompts/rag/answer_with_ref.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get("template", "")


    async def ask_question_stream(self, query: str, history: list, user_context: dict):
        # --- [1] 尝试从语义缓存获取 (按用户隔离) ---
        try:
            cached_res = await self.semantic_cache.get_cache(query, user_context)
            if cached_res:
                logger.info(f"🔥 [ULTRA HIT] 语义缓存命中: {query[:20]}...")
                # 模拟流式：直接 yield 整个答案
                yield cached_res['answer'], cached_res.get('sources', [])
                return 
        except Exception as e:
            logger.warning(f"⚠️ 读取语义缓存失败: {e}")

        # --- [3] 缓存未命中：执行 RAG 检索 ---
        # 这一步已经受惠于你之前的 Embedding Cache
        search_results = await self.rag_backend.secure_search(query, user_context)

        context_parts = []
        sources = set()
        
        # search_results 是 Milvus 返回的批量结果，通常是 [[hit1, hit2, ...], []]
        # 我们只需要第一个批次的结果
        if search_results and len(search_results) > 0:
            hits = search_results[0] if isinstance(search_results[0], list) else search_results
            for hit in hits[:5]:
                # 兼容实体获取
                text = hit.entity.get("text") if hasattr(hit, "entity") else hit.get("text", "")
                fname = hit.entity.get("file_name") if hasattr(hit, "entity") else hit.get("file_name", "")
                dept = hit.entity.get("department") if hasattr(hit, "entity") else hit.get("department", "")
                # 将文本和来源信息组合
                source_info = f"[文件: {fname}, 部门: {dept}]"
                context_parts.append(f"{text}\n来源: {source_info}")
                sources.add(f"{fname} ({dept})")

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
                SystemMessage(content=self.prompt_template.format(context=context_text, history=history, query=query)),
            ]

            async for chunk in llm.astream(messages):
                if chunk.content:
                    # 实时累加内容
                    full_answer_content.append(chunk.content)
                    # 实时推送给前端
                    yield chunk.content, source_list

        # --- [5] 异步存入语义缓存 (使用 SemanticCache 存储)
        if full_answer_content:
            try:
                complete_answer = "".join(full_answer_content)
                # 使用 semantic_cache 存储（按用户隔离）
                await self.semantic_cache.set_cache(query, complete_answer, source_list, user_context)
                logger.debug(f"💾 结果已存入语义缓存")
            except Exception as e:
                logger.error(f"❌ 写入语义缓存异常: {e}")
