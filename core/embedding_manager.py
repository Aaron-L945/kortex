import hashlib
import json
import redis.asyncio as aioredis
from cachetools import TTLCache
from loguru import logger
from config import Config

class EmbeddingCacheManager:
    def __init__(self, model_instance):
        self.model = model_instance
        self.l1_cache = TTLCache(maxsize=2000, ttl=3600)
        self.redis = aioredis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD
        )

    def _get_query_hash(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    async def get_embedding(self, text: str):
        query_hash = self._get_query_hash(text)

        # 1. 查 L1
        if query_hash in self.l1_cache:
            logger.debug("🎯 L1 Embedding Cache Hit")
            return self.l1_cache[query_hash]

        # 2. 查 L2 (Redis)
        redis_val = await self.redis.get(f"emb:{query_hash}")
        if redis_val:
            logger.debug("🥈 L2 Embedding Cache Hit")
            vector = json.loads(redis_val)
            self.l1_cache[query_hash] = vector # 回填 L1
            return vector

        # 3. 缓存穿透，计算 Embedding
        logger.info("⚡ Embedding Cache Miss - Calling Model...")
        # 假设你原来的模型方法是 model.embed_query
        vector = await self.model.aembed_query(text)

        # 4. 异步写入缓存
        self.l1_cache[query_hash] = vector
        await self.redis.set(f"emb:{query_hash}", json.dumps(vector), ex=86400 * 7) # 存7天
        
        return vector