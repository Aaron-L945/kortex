# 企业级 RAG 系统开发文档

## 📋 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [技术栈](#3-技术栈)
4. [目录结构](#4-目录结构)
5. [快速开始](#5-快速开始)
6. [核心模块详解](#6-核心模块详解)
7. [API 接口文档](#7-api-接口文档)
8. [部署指南](#8-部署指南)
9. [常见问题](#9-常见问题)

---

## 1. 系统概述

### 1.1 项目简介

企业级 RAG（检索增强生成）系统是一个安全、高效的智能问答平台，支持：
- 📄 PDF 文档入库与向量化
- 🔍 基于 Milvus 的向量检索
- 🧠 多轮对话与语义缓存
- 🔐 细粒度权限控制
- 📊 来源追溯（文件+部门）

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| 多轮对话 | 支持上下文理解，代词解析 |
| 语义缓存 | Redis 双层缓存（精确+向量） |
| 权限隔离 | 部门/角色/所有者三重控制 |
| 流式输出 | 实时返回，良好体验 |
| 来源标注 | 回答引用具体文件和部门 |

---

## 2. 架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端层                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Streamlit 前端 (client.py)                ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  ││
│  │  │ 登录认证 │  │ 文档入库  │  │       智能问答           │  ││
│  │  └──────────┘  └──────────┘  └──────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API 网关层                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              FastAPI 后端 (app/main.py)                      ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   ││
│  │  │ /v1/auth/* │  │/v1/chat/*  │  │   认证中间件        │   ││
│  │  └────────────┘  └────────────┘  └────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       业务逻辑层                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              SecureChatService (core/rag_chat_service.py)   ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    ││
│  │  │  语义缓存    │  │   多轮对话   │  │   提示词引擎   │    ││
│  │  │SemanticCache │  │   History   │  │    Template    │    ││
│  │  └──────────────┘  └──────────────┘  └────────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   向量数据库     │ │   缓存层         │ │   LLM 调度      │
│   Milvus        │ │   Redis         │ │   Model Pool    │
│   (向量检索)     │ │   (语义缓存)     │ │   (多模型路由)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 用户问答流程图

```
用户提问: "费曼是谁？"
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  1. 缓存查询                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SemanticCache.get_cache(query)                         │  │
│  │  ├── 精确匹配: MD5(query) → Redis GET                   │  │
│  │  └── 语义匹配: KNN向量搜索 → L2距离 < 0.15?              │  │
│  │                                                          │  │
│  │  命中? ──是──→ 返回缓存答案 [ULTRA HIT]                  │  │
│  │     │                                                    │  │
│  │     否                                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  2. 权限过滤表达式构建                                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  auth_expr = owner_id == 'user1'                        │  │
│  │           OR ARRAY_CONTAINS(department, 'Tech')          │  │
│  │           OR ARRAY_CONTAINS(role_access, 'user')         │  │
│  │                                                          │  │
│  │  admin 用户 → expr = None (不过滤)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  3. 向量检索 (Milvus)                                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  query_vec = embedder.embed_query("费曼是谁？")           │  │
│  │                                                          │  │
│  │  collection.search(                                      │  │
│  │    data=[query_vec],                                     │  │
│  │    expr=auth_expr,           # 权限过滤                  │  │
│  │    output_fields=["text", "file_name", "department"]     │  │
│  │  )                                                       │  │
│  │                                                          │  │
│  │  返回 Top-5 相关文档片段                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  4. 构建提示词                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  {context}:                                             │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │ 文档片段1内容                                     │   │  │
│  │  │ 来源: [文件: 费曼学习法.pdf, 部门: 研发部]         │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │ 文档片段2内容...                                   │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  │                                                          │  │
│  │  {history}:                                            │  │
│  │  用户: 费曼是谁？                                        │  │
│  │  助手: 理查德·费曼是...                                  │  │
│  │                                                          │  │
│  │  {query}: 谁是他的学生？                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  5. LLM 调度与生成                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  ModelPool.get_available_node() → 选择模型节点            │  │
│  │                                                          │  │
│  │  LLM.astream(prompt) → 流式返回                           │  │
│  │                                                          │  │
│  │  [结论]: 理查德·费曼的学生包括...                         │  │
│  │  [证据]: 他的学生包括 Robert B. Leighton... [文件: xxx]    │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  6. 结果缓存                                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SemanticCache.set_cache(                                │  │
│  │    query="谁是费曼的学生",                               │  │
│  │    answer=complete_answer,                               │  │
│  │    sources=["费曼学习法.pdf (研发部)"]                    │  │
│  │  )                                                       │  │
│  │                                                          │  │
│  │  Redis JSON: {query, answer, sources, vector}            │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
              返回给前端显示
```

### 2.3 文档入库流程图

```
用户上传 PDF
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  1. 文件解析                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  PyPDFLoader(tmp_path) → 读取 PDF 内容                    │  │
│  │  RecursiveCharacterTextSplitter(chunk_size=800) → 切分   │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  2. 向量化                                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  embedder.embed_documents(chunks) → 生成向量             │  │
│  │                                                          │  │
│  │  批处理: 每批 4 个 chunks，避免内存溢出                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  3. 元数据准备                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  {                                                       │  │
│  │    "text": chunk_content,                               │  │
│  │    "file_name": "xxx.pdf",                              │  │
│  │    "owner_id": "admin",                                 │  │
│  │    "department": ["public"],                            │  │
│  │    "role_access": ["user"],                            │  │
│  │    "domain": "tech"                                     │  │
│  │  }                                                       │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  4. Milvus 插入                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  collection.insert([                                     │  │
│  │    {vector, text, file_name, ...}                       │  │
│  │  ])                                                      │  │
│  │                                                          │  │
│  │  分批插入: 每批 10 条                                      │  │
│  │  插入后 flush() 确保数据落盘                               │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
      │
      ▼
         入库完成
```

---

## 3. 技术栈

### 3.1 核心依赖

| 组件 | 技术 | 用途 |
|------|------|------|
| **前端** | Streamlit | 可视化界面 |
| **后端** | FastAPI | REST API |
| **向量库** | Milvus | 向量存储与检索 |
| **缓存** | Redis Stack | 语义缓存+Sentinel |
| **Embedding** | BAAI/bge-m3 | 多语言向量化 |
| **LLM** | Qwen/GPT-4 | 答案生成 |

### 3.2 Python 依赖

```
fastapi>=0.100.0
uvicorn>=0.23.0
pymilvus>=2.3.0
redis[hiredis,search]>=5.0.0
langchain>=0.1.0
langchain-huggingface>=0.0.1
sentence-transformers>=2.2.0
pypdf>=3.15.0
streamlit>=1.28.0
loguru>=0.7.0
PyJWT>=2.8.0
passlib>=1.7.4
bcrypt>=4.0.0
```

---

## 4. 目录结构

```
kortex/
├── app/                    # FastAPI 后端
│   ├── main.py            # API 入口
│   ├── auth.py            # 认证逻辑
│   ├── user_manager.py    # 用户管理
│   └── service.py         # 业务服务
├── core/                   # RAG 核心模块
│   ├── embedding_manager.py   # Embedding 缓存管理
│   ├── rag_chat_service.py    # RAG 对话服务
│   ├── semantic_cache.py       # 语义缓存
│   ├── model_pool.py          # LLM 模型池
│   ├── retrieve.py            # 检索模块
│   └── tools.py               # 工具函数
├── client.py               # Streamlit 前端
├── build_milvus_index.py   # Milvus RAG 核心
├── config.py              # 配置管理
├── prompts/               # 提示词模板
│   └── rag/
│       └── answer_with_ref.yaml
├── scripts/                # 工具脚本
│   ├── test_concurrency.py
│   └── verify_ingestion.py
├── test_rag/              # RAG 测试
│   ├── test_milvus_rag.py
│   └── rag_benchmark.py
├── test_llm/              # LLM 测试
├── test_langchain/        # LangChain 测试
├── milvus_service/        # Milvus Docker
│   └── docker-compose.yaml
├── redis-stack/           # Redis Docker
│   └── docker-compose.yaml
├── docs/                  # 文档
├── archive/               # 废弃文件备份
├── sql_data/              # SQLite 数据库
├── data/                  # 数据目录
├── volumes/               # Docker 卷
├── .env                   # 环境变量
└── requirements.txt       # Python 依赖
```

---

## 5. 快速开始

### 5.1 环境要求

- Python 3.10+
- Docker & Docker Compose
- 16GB+ RAM (推荐)

### 5.2 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/Aaron-L945/kortex.git
cd kortex

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\\Scripts\\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填写必要配置
```

### 5.3 启动服务

```bash
# 1. 启动 Docker 服务 (Milvus + Redis)
cd milvus_service && docker-compose up -d
cd ../redis-stack && docker-compose up -d

# 2. 启动 FastAPI 后端
cd ..
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动 Streamlit 前端 (新终端)
streamlit run client.py --server.port 8501
```

### 5.4 验证服务

```bash
# 测试 Milvus 连接
python test_rag/test_milvus_rag.py

# 测试 RAG 功能
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

---

## 6. 核心模块详解

### 6.1 SecureChatService

位置: `core/rag_chat_service.py`

**功能**: RAG 对话核心服务

```python
class SecureChatService:
    def __init__(self, rag_backend):
        self.rag_backend = rag_backend
        self.emb_cache_manager = EmbeddingCacheManager(rag_backend)
        self.semantic_cache = SemanticCache(...)
        self.prompt_template = self._load_prompt_template()

    async def ask_question_stream(self, query, history, user_context):
        # 1. 尝试语义缓存
        cached = await self.semantic_cache.get_cache(query)
        if cached:
            yield cached['answer'], cached['sources']
            return

        # 2. RAG 检索
        results = await self.rag_backend.secure_search(query, user_context)

        # 3. 构建 context
        context = self._build_context(results)

        # 4. 调用 LLM
        async for chunk in llm.astream(prompt):
            yield chunk, sources

        # 5. 存入缓存
        await self.semantic_cache.set_cache(query, answer, sources)
```

### 6.2 SemanticCache

位置: `core/semantic_cache.py`

**功能**: Redis 语义缓存，支持精确+向量双层匹配

```python
class SemanticCache:
    def __init__(self, redis_client, embedding_manager):
        self.redis = redis_client
        self.emb_manager = embedding_manager
        self.threshold = 0.15  # L2 距离阈值

    async def get_cache(self, query):
        # 精确匹配
        cached = await self.redis.json().get(cache_key)
        if cached:
            return cached  # [EXACT HIT]

        # 语义匹配 (KNN)
        query_vec = await self.emb_manager.get_embedding(query)
        results = await self.redis.ft(self.index_name).search(
            Query("*=>[KNN 1 @vector $vec_param AS score]")
        )
        if results.docs and results.docs[0].score <= self.threshold:
            return result  # [SEMANTIC HIT]

    async def set_cache(self, query, answer, sources):
        vector = await self.emb_manager.get_embedding(query)
        await self.redis.json().set(cache_key, "$", {
            "query": query,
            "answer": answer,
            "sources": sources,
            "vector": vector
        })
```

### 6.3 EnterpriseSecureRAG

位置: `build_milvus_index.py`

**功能**: Milvus RAG 核心

```python
class EnterpriseSecureRAG:
    def _build_auth_expr(self, user_context):
        """构建权限表达式"""
        if user_context.get("role") == "admin":
            return None  # admin 不过滤

        return (
            f"(owner_id == '{user_id}' or "
            f"ARRAY_CONTAINS(department, '{dept}') or "
            f"ARRAY_CONTAINS(role_access, '{role}'))"
        )

    async def secure_search(self, query, user_context):
        """带权限控制的向量检索"""
        query_vec = await self.aembed_query(query)
        expr = self._build_auth_expr(user_context)

        return collection.search(
            data=[query_vec],
            expr=expr,
            output_fields=["text", "file_name", "department"]
        )

    async def aembed_query(self, text):
        """异步 embedding"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.embedder.embed_query(text)
        )
```

### 6.4 ModelPool

位置: `core/model_pool.py`

**功能**: 多模型调度

```python
class ModelRouter:
    def get_available_node(self):
        # 优先选择空闲节点
        for node in self.nodes:
            if node.is_available():
                return node
        # 等待或负载均衡
```

---

## 7. API 接口文档

### 7.1 认证接口

#### POST /v1/auth/login

**请求**:
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**响应**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### 7.2 对话接口

#### POST /v1/chat/completions

**Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**请求**:
```json
{
  "query": "费曼是谁？",
  "stream": true,
  "history": [
    {"role": "user", "content": "之前我问过你一个问题"},
    {"role": "assistant", "content": "是的，我记得..."}
  ]
}
```

**响应** (SSE 流式):
```
data: {"choices": [{"delta": {"content": "理查德"}}]}

data: {"choices": [{"delta": {"content": "·费曼"}}]}

data: [DONE]
```

---

## 8. 部署指南

### 8.1 Docker 部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - milvus
      - redis
    environment:
      - MILVUS_HOST=milvus
      - REDIS_HOST=redis

  web:
    build: .
    command: streamlit run client.py
    ports:
      - "8501:8501"

  milvus:
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"

  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
```

### 8.2 环境变量配置

```bash
# .env
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_HOST=localhost
REDIS_PORT=6379
EMBEDDING_MODEL_NAME=/path/to/bge-large-zh
OPENAI_API_KEY=sk-xxx
```

### 8.3 Redis Sentinel 高可用部署

本系统使用 Redis Sentinel 模式实现高可用，架构如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Redis Sentinel 高可用架构                    │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Sentinel 1  │     │  Sentinel 2  │     │  Sentinel 3  │       │
│  │  (26379)    │     │  (26379)    │     │  (26379)    │       │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                    ┌─────────▼─────────┐                       │
│                    │    Master (主)     │                       │
│                    │   172.20.0.10:6379 │                       │
│                    │   (写/读命令)       │                       │
│                    └─────────┬─────────┘                       │
│                              │                                   │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                 │
│     ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│     │  Slave 1   │   │  Slave 2   │   │  Slave 3   │         │
│     │  (从)       │   │  (从)       │   │  (从)       │         │
│     └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                 │
│  Sentinel 选举机制:                                              │
│  - 3 个 Sentinel 节点，quorum=2                                  │
│  - 任何 2 个 Sentinel 认为 Master 宕机则触发故障转移               │
│  - 新 Master 由多数 Sentinel 选举产生                            │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.3.1 Docker Compose 配置

位置: `redis-stack/docker-compose.yaml`

```yaml
version: '3.8'

services:
  # 主节点
  redis-master:
    container_name: redis-master
    image: redis/redis-stack:latest
    environment:
      - REDIS_ARGS=--requirepass yourpassword --masterauth yourpassword
    ports:
      - "6379:6379"
      - "8001:8001"  # Redis Insight 可视化界面
    volumes:
      - ./data-master:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "yourpassword", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      redis-network:
        ipv4_address: 172.20.0.10  # 固定 IP 避免 DNS 解析问题
    restart: always

  # 从节点 1
  redis-slave1:
    container_name: redis-slave1
    image: redis/redis-stack:latest
    environment:
      - REDIS_ARGS=--replicaof 172.20.0.10 6379 --requirepass yourpassword --masterauth yourpassword
    depends_on:
      redis-master:
        condition: service_healthy
    volumes:
      - ./data-slave1:/data
    networks:
      - redis-network
    restart: always

  # 从节点 2
  redis-slave2:
    container_name: redis-slave2
    image: redis/redis-stack:latest
    environment:
      - REDIS_ARGS=--replicaof 172.20.0.10 6379 --requirepass yourpassword --masterauth yourpassword
    depends_on:
      redis-master:
        condition: service_healthy
    volumes:
      - ./data-slave2:/data
    networks:
      - redis-network
    restart: always

  # Sentinel 节点 (至少 3 个)
  sentinel1:
    container_name: redis-sentinel1
    image: redis:latest
    command: [
      "sh", "-c",
      "until redis-cli -h 172.20.0.10 -a yourpassword ping 2>/dev/null | grep -q PONG; do sleep 1; done && \
      printf 'port 26379\\nsentinel monitor mymaster 172.20.0.10 6379 2\\nsentinel auth-pass mymaster yourpassword\\n' > /tmp/sentinel.conf && \
      redis-sentinel /tmp/sentinel.conf"
    ]
    depends_on:
      redis-master:
        condition: service_healthy
    networks:
      - redis-network
    restart: always

  sentinel2:
    container_name: redis-sentinel2
    image: redis:latest
    # ... 同 sentinel1
    restart: always

  sentinel3:
    container_name: redis-sentinel3
    image: redis:latest
    # ... 同 sentinel1
    restart: always

networks:
  redis-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### 8.3.2 启动 Redis Sentinel

```bash
# 进入 Redis 配置目录
cd redis-stack

# 创建必要的目录
mkdir -p data-master data-slave1 data-slave2

# 启动所有容器
docker-compose up -d

# 查看容器状态
docker ps | grep redis

# 查看 Sentinel 日志
docker logs redis-sentinel1 --tail 10
```

#### 8.3.3 验证 Sentinel 集群

```bash
# 连接 Sentinel
redis-cli -p 26379

# 查看 Sentinel 信息
SENTINEL masters

# 查看 Master 状态
SENTINEL master mymaster

# 查看 Slaves
SENTINEL slaves mymaster

# 测试故障转移
# 1. 停止 Master
docker stop redis-master

# 2. 等待 Sentinel 选举新 Master (约 10-30 秒)
# 3. 查看新 Master
SENTINEL get-master-addr-by-name mymaster

# 4. 重启原 Master (自动变为 Slave)
docker start redis-master
```

#### 8.3.4 应用连接配置

应用使用固定 IP `172.20.0.10` 连接 Redis Master：

```python
# config.py
REDIS_HOST = "172.20.0.10"  # Master 固定 IP
REDIS_PORT = 6379
REDIS_PASSWORD = "yourpassword"
```

---

## 9. 常见问题

### Q1: 语义缓存未命中？

**原因**: 
- 两次问题向量相似度不够高 (L2 > 0.15)
- Redis Stack 向量索引未创建

**解决**:
```bash
# 检查索引
redis-cli FT.INFO idx:semantic_cache
# 重建索引
redis-cli FT.DROPINDEX idx:semantic_cache
# 重启应用自动创建
```

### Q2: Milvus 检索报错 `metric type not match`？

**原因**: 搜索参数与索引类型不匹配

**解决**: 确保使用正确的参数
```python
# IVF_FLAT 索引用 nprobe
param = {"metric_type": "COSINE", "params": {"nprobe": 16}}

# HNSW 索引用 ef
param = {"metric_type": "COSINE", "params": {"ef": 64}}
```

### Q3: 多轮对话未生效？

**原因**: 未传递 history 参数

**解决**: 确保 client.py 发送 history
```python
payload = {
    "query": prompt,
    "stream": True,
    "history": st.session_state.messages[:-1]  # 排除当前消息
}
```

---

## 📄 附录

### A. 提示词模板

位置: `prompts/rag/answer_with_ref.yaml`

关键规则:
- 禁止推理，只引用原文
- 引用格式: `[文件: xxx, 部门: xxx]`
- 死板拒答: "抱歉，根据已知资料无法回答该问题"

### B. Milvus Schema

```python
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="department", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50),
    FieldSchema(name="role_access", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50),
]
```

### C. Redis 数据结构

```
# 精确缓存
ans:{md5_hash} → JSON {query, answer, sources}

# 语义缓存索引
idx:semantic_cache → 向量索引 (HNSW)
```

---

**文档版本**: v1.0  
**更新日期**: 2026-04-30  
**维护者**: Aaron
