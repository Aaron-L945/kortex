import os
from typing import List, Dict, Any
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 加载环境变量
load_dotenv()

# ==========================================
# 1. 配置参数 (从 .env 读取)
# ==========================================
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))


# ==========================================
# 2. 企业级安全 RAG 类
# ==========================================
class EnterpriseSecureRAG:
    def __init__(self, host="10.66.196.31", port="19530"):
        # 初始化 BGE-M3 模型
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_PATH}...")
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cuda"},  # 如果没 GPU 改成 'cpu'
        )

        from core.embedding_manager import EmbeddingCacheManager

        self.emb_cache_manager = EmbeddingCacheManager(model_instance=self.embedder)

        # BGE-M3 默认维度是 1024
        self.dim = 1024
        self.collection_name = "enterprise_knowledge_vault"

        # 连接 Milvus
        connections.connect("default", host=host, port=port)
        self._setup_collection()

    def _setup_collection(self):
        """定义三层元数据 Schema"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                # 第一层：基础 (Basic)
                FieldSchema(
                    name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=200),
                # 第二层：访问控制 (Access Control)
                FieldSchema(name="owner_id", dtype=DataType.VARCHAR, max_length=100),
                # 修正：ARRAY 类型的 VARCHAR 元素也必须有 max_length
                FieldSchema(
                    name="department",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_capacity=50,
                    max_length=100,
                ),
                FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=100),
                # 修正：同上
                FieldSchema(
                    name="role_access",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_capacity=50,
                    max_length=100,
                ),
                # 第三层：业务语义 (Business Semantic)
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            ]
            schema = CollectionSchema(fields, "企业级三层权限隔离知识库")
            self.collection = Collection(self.collection_name, schema)

            # 创建 HNSW 索引
            # 提示：既然你有 1TB 内存和 AVX512，可以适当调大 M 和 efConstruction 提升检索精度
            index_params = {
                "metric_type": "L2",  # 或者用 "IP" (内积)，取决于 BGE-M3 的训练方式
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            }
            self.collection.create_index("vector", index_params)

        self.collection.load()

    def ingest_pdf(self, file_path: str, access_info: Dict, business_tags: Dict):
        """解析 PDF 并存入 Milvus"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(loader.load())

        if not chunks:
            return

        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        data = [
            embeddings,
            texts,
            [os.path.basename(file_path)] * len(chunks),
            [access_info["owner_id"]] * len(chunks),
            [access_info["department"]] * len(chunks),
            [access_info["project_id"]] * len(chunks),
            [access_info["role_access"]] * len(chunks),
            [business_tags["domain"]] * len(chunks),
            [business_tags["doc_type"]] * len(chunks),
        ]

        self.collection.insert(data)
        self.collection.flush()
        print(f"✅ 已入库: {os.path.basename(file_path)} ({len(chunks)} chunks)")

    async def secure_search(
        self, query: str, user_context: Dict, semantic_filters: Dict = None
    ):
        """执行带权限卡控的向量检索"""

        # 1. 获取向量 (确保 cache 内部处理了 await)
        query_vec = await self.emb_cache_manager.get_embedding(query)

        # 2. 构造权限表达式
        ac_expr = ""
        is_admin = user_context.get("role") == "admin"

        if not is_admin:
            # 普通用户三路过滤
            ac_expr = (
                f"(owner_id == '{user_context['user_id']}') or "
                f"(ARRAY_CONTAINS(department, '{user_context['dept']}')) or "
                f"(ARRAY_CONTAINS(role_access, '{user_context['role']}'))"
            )

        # 3. 叠加业务标签过滤 (修正空字符串拼接问题)
        if semantic_filters:
            filter_parts = []
            for key, val in semantic_filters.items():
                filter_parts.append(f"{key} == '{val}'")

            filter_expr = " and ".join(filter_parts)

            if ac_expr:
                ac_expr = f"({ac_expr}) and ({filter_expr})"
            else:
                ac_expr = filter_expr

        # 4. 执行检索 (Milvus search 是同步 I/O，建议放在 to_thread 中防止阻塞)
        # 如果 ac_expr 最终仍为空字符串，传 None 给 Milvus
        final_expr = ac_expr if ac_expr else None

        results = self.collection.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 64}},
            limit=RETRIEVAL_TOP_K,
            expr=final_expr,
            output_fields=["text", "file_name"]
        )

        return results
