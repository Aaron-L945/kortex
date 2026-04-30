import os
import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
from dotenv import load_dotenv
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. 核心整合类
# ==========================================
class EnterpriseSecureRAG:
    def __init__(self, host="127.0.0.1", port="19530"):
        load_dotenv()
        self.embedding_model_path = os.getenv("EMBEDDING_MODEL_NAME")
        self.dim = 1024  # 需确保与模型匹配
        self.collection_name = "enterprise_knowledge_vault"
        self.top_k = int(os.getenv("RETRIEVAL_TOP_K", 10))

        # 初始化向量模型
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={"device": "cpu"},
        )

        # 建立连接
        try:
            connections.connect("default", host=host, port=port)
            logger.success(f"✅ 已连接到 Milvus: {host}:{port}")
        except Exception as e:
            logger.error(f"❌ 无法连接到 Milvus: {e}")
            raise

        self._setup_collection()

    # 异步 embedding 方法（供 EmbeddingCacheManager 调用）
    async def aembed_query(self, text: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embedder.embed_query(text))

    async def aembed_documents(self, texts: List[str]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embedder.embed_documents(texts))

    def _setup_collection(self):
        """定义 Schema 并创建适配 CPU 的索引"""
        if utility.has_collection(self.collection_name):
            logger.info(f"📦 发现现有集合: {self.collection_name}")
            self.collection = Collection(self.collection_name)
        else:
            logger.info(f"🆕 正在创建新集合: {self.collection_name}")
            
            # 定义字段结构
            fields = [
                # 主键：自动递增 ID
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                
                # 向量字段：维度需与 Embedding 模型一致
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                
                # 基础文本：VARCHAR 长度根据 CHUNK_SIZE 调整，建议不要设得极端大
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
                
                # 元数据字段
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="owner_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
                
                # 权限隔离核心字段：ARRAY 类型
                FieldSchema(
                    name="department",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_capacity=50,
                    max_length=100,
                ),
                FieldSchema(
                    name="role_access",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_capacity=50,
                    max_length=100,
                ),
            ]
            
            schema = CollectionSchema(fields, description="企业级三层权限隔离知识库")
            self.collection = Collection(self.collection_name, schema)
            
            # ==========================================
            # CPU 优化索引配置 (IVF_FLAT)
            # ==========================================
            # 相比 HNSW，IVF_FLAT 加载极快，内存占用更低
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}, # 将空间划分为 1024 个聚类桶
            }
            
            logger.info("⚡ 正在创建 IVF_FLAT 索引...")
            self.collection.create_index("vector", index_params)
            logger.success("✅ 索引创建成功")

        # ==========================================
        # 关键加载步骤
        # ==========================================
        # 在 CPU 环境下，IVF_FLAT 会让这一步非常迅速
        try:
            logger.info("🚀 正在加载 Collection 进内存...")
            self.collection.load()
            logger.success("📖 Collection 已就绪 (Memory Loaded)")
        except Exception as e:
            logger.error(f"❌ 加载失败，可能是索引类型冲突，请尝试 drop 后重建: {e}")

    def _build_auth_expr(self, user_context: Dict) -> Optional[str]:
        """封装权限表达式构建逻辑"""
        if user_context.get("role") == "admin":
            return None
            
        u_id = user_context.get('user_id')
        u_dept = user_context.get('dept')
        u_role = user_context.get('role')

        # 使用你第二个代码片段中更简洁的拼接方式
        return (
            f"(owner_id == '{u_id}' or "
            f"ARRAY_CONTAINS(department, '{u_dept}') or "
            f"ARRAY_CONTAINS(role_access, '{u_role}'))"
        )

    async def secure_search(self, query: str, user_context: Dict, semantic_filters: Optional[Dict] = None):
        """
        整合了异步执行和权限控制的搜索方法
        """
        try:
            # 1. 生成查询向量 (同步转异步)
            loop = asyncio.get_running_loop()
            query_vec = await loop.run_in_executor(None, lambda: self.embedder.embed_query(query))

            # 2. 构建表达式
            auth_expr = self._build_auth_expr(user_context)
            
            # 叠加业务过滤
            if semantic_filters:
                biz_expr = " and ".join([f"{k} == '{v}'" for k, v in semantic_filters.items()])
                expr = f"{auth_expr} and ({biz_expr})" if auth_expr else biz_expr
            else:
                expr = auth_expr

            logger.debug(f"🔍 检索表达式: {expr}")

            # 3. 在线程池中执行 Milvus 检索 (继承自 MilvusSearcher 的核心逻辑)
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.search(
                    data=[query_vec],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                    limit=self.top_k,
                    expr=expr,
                    output_fields=["text", "file_name", "department"],
                ),
            )

            return results[0]

        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            return []