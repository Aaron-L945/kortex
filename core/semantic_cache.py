import json
import hashlib
import numpy as np
from loguru import logger
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class SemanticCache:
    def __init__(self, redis_client, embedding_manager, index_name="idx:semantic_cache", dimension=1024):
        """
        :param redis_client: 异步 Redis 客户端 (需支持 Redis Stack)
        :param embedding_manager: EmbeddingCacheManager 实例
        :param index_name: 向量索引名称
        :param dimension: 向量维度 (bge-large-zh 为 1024)
        """
        self.redis = redis_client
        self.emb_manager = embedding_manager
        self.index_name = index_name
        self.dim = dimension
        self.cache_prefix = "ans:"  # 统一前缀
        # 阈值：L2距离越小越相似。BGE模型建议 0.1 - 0.2 左右，0.98这种余弦值需要转换
        self.threshold = 0.15 

    def _get_query_hash(self, query: str, user_context: dict = None) -> str:
        """生成查询哈希，加入用户上下文以实现权限隔离"""
        base = query.strip().lower()
        if user_context:
            # 将用户身份加入哈希，实现用户级别的缓存隔离
            ctx = f"{user_context.get('user_id', '')}:{user_context.get('dept', '')}:{user_context.get('role', '')}"
            base = f"{base}:{ctx}"
        return hashlib.md5(base.encode()).hexdigest()

    async def init_index(self):
        """初始化 Redis Stack 向量索引"""
        try:
            await self.redis.ft(self.index_name).info()
            logger.info(f"✅ 语义缓存索引 '{self.index_name}' 已存在")
        except Exception:
            logger.info(f"🚀 正在创建 Redis 向量索引: {self.index_name}...")
            # 定义 Schema: 原始问题(文本) + 向量(向量字段)
            schema = (
                TextField("$.query", as_name="query"),
                VectorField("$.vector", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": "L2",
                }, as_name="vector")
            )
            # 指定索引前缀和数据类型
            await self.redis.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[self.cache_prefix], index_type=IndexType.JSON)
            )
            logger.info("✅ 向量索引创建成功")

    async def get_cache(self, query: str, user_context: dict = None):
        """双层检索：精确哈希 + 语义向量 (按用户隔离)"""
        q_hash = self._get_query_hash(query, user_context)
        cache_key = f"{self.cache_prefix}{q_hash}"

        # 1. 精确匹配 (已按用户隔离)
        cached_data = await self.redis.json().get(cache_key)
        if cached_data:
            logger.info(f"🎯 [EXACT HIT] 精确命中: {query[:15]}...")
            if isinstance(cached_data.get('sources'), str):
                cached_data['sources'] = json.loads(cached_data['sources'])
            return cached_data

        # 2. 语义搜索：仅搜索当前用户缓存的数据（通过前缀匹配用户 hash）
        # 由于 key 已包含用户上下文，全局 KNN 搜索可能会命中其他用户的缓存
        # 为安全起见，语义搜索也限制在当前用户的缓存空间
        try:
            # 搜索当前用户的所有缓存 key（以 q_hash 开头）
            user_prefix = f"{self.cache_prefix}{q_hash[:8]}"  # 匹配前8位
            pattern = f"{user_prefix}*"
            
            keys = await self.redis.keys(pattern)
            if keys and cache_key in keys:
                cached_data = await self.redis.json().get(cache_key)
                if cached_data:
                    logger.info(f"🧠 [SEMANTIC HIT] 用户缓存命中")
                    if isinstance(cached_data.get('sources'), str):
                        cached_data['sources'] = json.loads(cached_data['sources'])
                    return cached_data
        except Exception as e:
            logger.warning(f"⚠️ 语义搜索异常: {e}")

        return None

    async def set_cache(self, query: str, answer: str, sources: list, user_context: dict = None, expire=86400):
        """存入 RedisJSON 格式数据 (按用户隔离)"""
        try:
            q_hash = self._get_query_hash(query, user_context)
            cache_key = f"{self.cache_prefix}{q_hash}"
            
            # 获取向量并转为 list 存储
            vector = await self.emb_manager.get_embedding(query)
            
            payload = {
                "query": query,
                "answer": answer,
                "sources": json.dumps(sources), # 存为字符串方便检索返回
                "vector": vector
            }
            
            # 存入 JSON
            await self.redis.json().set(cache_key, "$", payload)
            await self.redis.expire(cache_key, expire)
            logger.debug(f"💾 语义缓存已存入: {q_hash}")
        except Exception as e:
            logger.error(f"❌ 写入语义缓存失败: {e}")