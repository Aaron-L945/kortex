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
from fastapi.concurrency import run_in_threadpool
from fastapi.concurrency import iterate_in_threadpool

# 导入你之前定义的业务模块
from app.user_manager import UserManager
from core.rag_chat_service import SecureChatService

# ==========================================
# 1. 初始化服务
# ==========================================


# 1. 定义一个全局变量占位，先不初始化
rag_service = None
user_db = None


# 2. 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    # 这里才是真正加载模型的地方
    if rag_service is None:
        from core.rag_chat_service import SecureChatService

        rag_service = SecureChatService()
    yield
    # 这里可以写清理逻辑（如关闭 Milvus 连接）
    del rag_service


# 3. 将 lifespan 传入 FastAPI
app = FastAPI(title="Enterprise Secure RAG API", lifespan=lifespan)

# 单例初始化：用户管理与RAG引擎
user_db = UserManager()

# 定义安全认证方案
security = HTTPBearer()


# ==========================================
# 2. 安全鉴权中间件 (身份认证与权限决策层)
# ==========================================
async def get_current_active_user(
    auth: HTTPAuthorizationCredentials = Security(security),
):
    """
    核心防护层：校验 JWT Token，确保用户身份真实且未过期
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="身份凭证无效、已过期或缺失",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 解码 Token，内部包含 SECRET_KEY 签名校验
    payload = user_db.decode_token(auth.credentials)

    if payload is None:
        raise credentials_exception
    if payload == "EXPIRED":
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")

    # 返回的内容包含：user_id, dept, role
    # 这确保了后续检索使用的 user_context 是绝对可信的
    return payload


# ==========================================
# 3. 数据模型定义
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
@app.post("/login", tags=["Auth"])
async def login(req: LoginRequest):
    """
    用户登录入口：校验 SQLite 中的哈希密码，签发 JWT
    """
    token = user_db.authenticate_user(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    logger.info(f"用户 {req.username} 登录成功，签发 Token")
    return {"access_token": token, "token_type": "bearer"}


@app.post("/v1/chat/completions", tags=["RAG"])
async def chat_endpoint(
    request: ChatRequest, 
    user_context: dict = Depends(get_current_active_user)
):
    logger.info(f"RAG请求 | 用户: {user_context['user_id']} | 问题: {request.query}")

    async def event_generator():
        # 1. 记录开始时间
        start_time = time.perf_counter()
        has_sent_sources = False
        
        try:
            # 获取同步生成器
            sync_gen = rag_service.ask_question_stream(request.query, user_context)

            # 使用 iterate_in_threadpool 确保同步迭代不阻塞异步主循环
            async for chunk_data, sources in iterate_in_threadpool(sync_gen):
                
                # --- 修复 1: 严格过滤空内容 ---
                # 如果没有文字内容，且没有来源信息，就不要发这个包
                if not chunk_data and not (not has_sent_sources and sources):
                    continue

                # --- 修复 2: 构造标准 OpenAI 响应格式 ---
                payload = {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk_data},
                        "finish_reason": None
                    }]
                }

                # 仅在第一帧或有来源时注入 sources
                if not has_sent_sources and sources:
                    payload["sources"] = sources
                    has_sent_sources = True
                
                # --- 修复 3: 严格控制换行符 ---
                # data: {JSON}\n\n (注意：中间不要有空格，末尾不多不少两个换行)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                
                # 强制让出控制权，确保数据立刻离开 Docker 容器进入网络
                await asyncio.sleep(0)

            # 2. 发送结束信号
            yield "data: [DONE]\n\n"
            
            logger.success(f"流式推送完成 | 耗时: {time.perf_counter() - start_time:.2f}s")

        except Exception as e:
            logger.error(f"流生成异常: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", 
            "Connection": "keep-alive"
        }
    )


# ==========================================
# 5. 启动配置
# ==========================================
if __name__ == "__main__":
    # 在 1TB 内存的高配环境下，设置多进程 workers 提升并发处理能力
    # host 0.0.0.0 允许内网其他机器访问
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1, reload=False)
