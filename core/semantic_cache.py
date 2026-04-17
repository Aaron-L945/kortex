import json
import hashlib
from loguru import logger
import numpy as np

class SemanticCache:
    def __init__(self, redis_client, embedding_manager, threshold=0.98):
        """
        :param redis_client: 异步 Redis 客户端
        :param embedding_manager: 之前实现的 EmbeddingCacheManager 实例
        :param threshold: 语义命中阈值 (0.0~1.0)，建议 0.95~0.99 之间
        """
        self.redis = redis_client
        self.emb_manager = embedding_manager
        self.threshold = threshold
        self.cache_prefix = "sem_ans:"

    def _get_query_hash(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    async def get_cache(self, query: str):
        """
        尝试获取缓存。
        1. 首先尝试极其快速的字符串精确匹配。
        2. (可选) 如果精确匹配失败，可以扩展为在 Redis 向量库中进行相似度检索。
        """
        q_hash = self._get_query_hash(query)
        
        # 1. 精确匹配检查 (O(1) 速度)
        cached_data = await self.redis.get(f"{self.cache_prefix}{q_hash}")
        if cached_data:
            logger.info(f"🚀 [EXACT HIT] 精确匹配命中缓存: {query[:20]}")
            return json.loads(cached_data)

        # 2. 语义匹配逻辑 (此处为简化版，建议在高阶版本中接入 RedisVL)
        # 目前先实现精确匹配以解决你的并发脚本重复请求问题
        return None

    async def set_cache(self, query: str, answer: str, sources: list, expire=86400):
        """
        存入缓存
        """
        q_hash = self._get_query_hash(query)
        data = {
            "answer": answer,
            "sources": sources,
            "query": query
        }
        try:
            await self.redis.set(
                f"{self.cache_prefix}{q_hash}", 
                json.dumps(data), 
                ex=expire
            )
            logger.debug(f"💾 答案已存入语义缓存: {q_hash}")
        except Exception as e:
            logger.error(f"写入语义缓存失败: {e}")