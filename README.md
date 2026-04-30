# 🚀 企业级 RAG 智能问答系统

企业级检索增强生成（RAG）系统，支持 PDF 文档入库、多轮对话、语义缓存和细粒度权限控制。

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 📄 **文档入库** | PDF 文件上传，自动切分向量化 |
| 🔍 **向量检索** | 基于 Milvus 的语义搜索 |
| 💬 **多轮对话** | 支持上下文理解，代词解析 |
| ⚡ **语义缓存** | Redis 双层缓存，相似问题秒回 |
| 🔐 **权限控制** | 部门/角色/所有者三重隔离 |
| 📊 **来源追溯** | 回答标注具体文件和部门 |

## 📈 效果展示

```
用户: 费曼是谁？
助手: 理查德·费曼是著名物理学家...
     [文件: 费曼学习法.pdf, 部门: 研发部]

用户: 他的学生有哪些？
助手: 他的学生包括 Robert B. Leighton 等...
     [文件: 费曼学习法.pdf, 部门: 研发部]
```

**缓存命中效果：**
- 首次提问：500-2000ms（检索+生成）
- 缓存命中：< 50ms（直接返回）

## 🚀 Quick Start

### 1. 环境准备

```bash
# 克隆项目
git clone https://gitee.com/aaron_945/kortex.git
cd kortex

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置
cp .env.example .env

# 编辑 .env 填写模型路径和密钥
```

### 3. 启动服务

```bash
# 启动 Docker 服务
cd milvus_service && docker-compose up -d
cd ../redis-stack && docker-compose up -d

# 启动后端 (终端 1)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 启动前端 (终端 2)
streamlit run client.py --server.port 8501
```

### 4. 验证

浏览器打开 `http://localhost:8501`

1. 使用 `admin/admin123` 登录
2. 上传 PDF 文档
3. 开始提问

## 📁 项目结构

```
kortex/
├── app/                    # FastAPI 后端
│   ├── main.py            # API 入口
│   └── user_manager.py    # 用户管理
├── core/                   # RAG 核心
│   ├── rag_chat_service.py    # 对话服务
│   └── semantic_cache.py       # 语义缓存
├── client.py               # Streamlit 前端
├── build_milvus_index.py   # Milvus RAG
├── milvus_service/        # Milvus Docker
├── redis-stack/           # Redis Docker
└── docs/                 # 开发文档
```

## 🔧 技术栈

- **前端**: Streamlit
- **后端**: FastAPI + Uvicorn
- **向量库**: Milvus (IVF_FLAT 索引)
- **缓存**: Redis Stack + Sentinel
- **Embedding**: BAAI/bge-m3
- **LLM**: Qwen / GPT-4

## 📖 文档

详细开发文档：[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

---

**版本**: v1.0 | **维护**: Aaron | **日期**: 2026-04-30
