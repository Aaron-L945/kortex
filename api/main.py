"""
FastAPI 主入口

层次：
  POST /api/auth/login     → JWT 登录
  GET  /api/auth/me        → 当前用户信息
  POST /api/chat           → 对话（流式/非流式）
  POST /api/docs/upload    → 上传文档
  GET  /api/docs/list      → 文档列表
  DELETE /api/docs/{id}    → 删除文档
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from api.auth import authenticate_user, create_access_token, get_current_user
from api.routes import chat, docs
from models.schemas import Token, UserInfo

app = FastAPI(
    title="企业知识库 Agent API",
    description="基于 Claude + LlamaIndex + FAISS 的权限感知知识库系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 注册路由 ─────────────────────────────────────────────────────────────────

app.include_router(chat.router, prefix="/api")
app.include_router(docs.router, prefix="/api")


# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/api/auth/token", response_model=Token, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    token = create_access_token({"sub": user["username"]})
    return Token(access_token=token, token_type="bearer")


@app.get("/api/auth/users/me", response_model=UserInfo, tags=["Auth"])
async def me(current_user: UserInfo = Depends(get_current_user)):
    return current_user


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
