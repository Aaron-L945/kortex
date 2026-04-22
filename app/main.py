"""
main.py
"""

import time
import json
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Depends, status, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Optional, List
from loguru import logger

# 导入业务模块
from app.user_manager import UserManager

# 🚩 导入我们新写的模型池路由模块
from core.rag_chat_service import SecureChatService
from build_milvus_index import EnterpriseSecureRAG

# ==========================================
# 1. 初始化服务
# ==========================================
user_db = UserManager()
security = HTTPBearer()
rag_backend = EnterpriseSecureRAG()
rag_service = SecureChatService(rag_backend=rag_backend)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    针对 1TB 内存 + Redis Stack 的生命周期管理
    """
    logger.info("🚀 正在启动 Enterprise RAG 系统...")
    
    # 🚩 [新增] 初始化语义缓存索引 (Redis Stack 向量库)
    try:
        # 确保 semantic_cache 已经正确挂载在 rag_service 中
        await rag_service.semantic_cache.init_index()
        logger.info("🎯 语义缓存向量索引已就绪")
    except Exception as e:
        logger.error(f"❌ 语义缓存索引初始化失败: {e}。请检查 Redis Stack 是否正常运行。")

    yield

    # 🚩 [新增] 优雅关闭 Redis 连接
    logger.info("🛑 正在释放资源...")
    try:
        await rag_backend.emb_cache_manager.redis.close()
        logger.info("✅ Redis 连接已安全关闭")
    except Exception as e:
        logger.error(f"释放资源异常: {e}")
    
    logger.info("🏠 服务已完全停止")


app = FastAPI(title="Enterprise Secure RAG API", lifespan=lifespan)


# ==========================================
# 2. 安全鉴权中间件
# ==========================================
async def get_current_active_user(
    auth: HTTPAuthorizationCredentials = Security(security),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="身份凭证无效、已过期或缺失",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = user_db.decode_token(auth.credentials)
    if payload is None:
        raise credentials_exception
    if payload == "EXPIRED":
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    return payload


# ==========================================
# 3. 数据模型
# ==========================================
class ChatRequest(BaseModel):
    query: str
    stream: bool = True


class LoginRequest(BaseModel):
    username: str
    password: str


# ==========================================
# 4. API 路由实现
# ==========================================


@app.post("/v1/auth/login", tags=["Auth"])
async def login(req: LoginRequest):
    token = user_db.authenticate_user(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    logger.info(f"用户 {req.username} 登录成功")
    return {"access_token": token, "token_type": "bearer"}


@app.post("/v1/chat/completions", tags=["RAG"])
async def chat_endpoint(
    request: ChatRequest, user_context: dict = Depends(get_current_active_user)
):
    """
    经过模型池调度的 RAG 接口：
    1. 接收请求并构造 Payload
    2. 由 Router 决定分发到本地 vLLM 还是云端 API
    3. 实时流式转发
    """
    logger.info(f"收到请求 | 用户: {user_context['user_id']} | 调度开始")

    # 构造发送给模型节点的标准 OpenAI 格式 Payload
    # 可以在这里注入 RAG 检索到的 Context（如果 RAG 逻辑在 main 之外）
    model_payload = {
        "model": "qwen-rag",  # 逻辑模型名
        "messages": [
            {
                "role": "system",
                "content": f"你是一个企业助手。用户信息: {user_context['dept']}",
            },
            {"role": "user", "content": request.query},
        ],
        "stream": True,
        "temperature": 0.1,
    }

    async def event_generator():
        # 🚩 调用业务层逻辑
        try:
            async for chunk, sources in rag_service.ask_question_stream(
                request.query, user_context
            ):
                payload = {
                    "choices": [{"delta": {"content": chunk}}],
                    "sources": sources,  # 可以在第一帧带上来源
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"模型池转发异常: {e}")
            err_payload = {
                "choices": [{"delta": {"content": f"\n[调度系统异常: {str(e)}]"}}]
            }
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓存
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


# ==========================================
# 5. 启动配置
# ==========================================
if __name__ == "__main__":
    # 针对 1TB 内存环境：
    # 1. 虽然内存大，但 uvicorn worker 建议设为 1，内部利用 asyncio 处理并发更安全
    # 2. 如果 CPU 核心极多且是多租户场景，可考虑 workers=2~4，但注意单例模式下的内存占用
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,  # 这里的端口要对应你之前在 Streamlit 里填写的
        workers=1,
        reload=False,
        timeout_keep_alive=120,
    )
