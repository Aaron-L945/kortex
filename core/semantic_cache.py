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

        # 1. 尝试极速精确匹配 (RedisJSON get)
        cached_data = await self.redis.json().get(cache_key)
        if cached_data:
            logger.info(f"🎯 [EXACT HIT] 精确匹配命中: {query[:15]}...")
            # RedisJSON 返回的是字典，直接处理 sources
            if isinstance(cached_data.get('sources'), str):
                cached_data['sources'] = json.loads(cached_data['sources'])
            return cached_data

        # 2. 语义模糊匹配 (KNN 搜索)
        try:
            # 获取当前问题的向量
            query_vec = await self.emb_manager.get_embedding(query)
            query_vec_np = np.array(query_vec, dtype=np.float32).tobytes()

            # 构造 K-最近邻查询 (寻找最像的 1 个)
            q = (
                Query("*=>[KNN 1 @vector $vec_param AS score]")
                .sort_by("score")
                .return_fields("$.answer", "$.sources", "$.query", "score")
                .dialect(2)
            )
            
            res = await self.redis.ft(self.index_name).search(
                q, query_params={"vec_param": query_vec_np}
            )

            if res.docs:
                best_match = res.docs[0]
                score = float(best_match.score)
                
                # 判断是否在语义误差范围内
                if score <= self.threshold:
                    logger.info(f"🧠 [SEMANTIC HIT] 语义命中 (Distance: {score:.4f})")
                    return {
                        "answer": getattr(best_match, "$.answer"),
                        "sources": json.loads(getattr(best_match, "$.sources")),
                        "query": getattr(best_match, "$.query")
                    }
        except Exception as e:
            logger.error(f"❌ 语义检索异常: {e}")

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