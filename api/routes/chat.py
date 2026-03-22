"""
Chat 路由：接收用户问题，调用 Agent 调度层，返回流式或非流式响应。
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import get_current_user
from models.schemas import UserInfo
from agent.scheduler import run_agent, run_agent_stream

router = APIRouter(prefix="/agent/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    stream: bool = False


@router.post("")
async def chat(
    req: ChatRequest,
    user: UserInfo = Depends(get_current_user),
):
    """
    核心对话接口。
    - stream=true：返回 SSE 流式文本
    - stream=false：返回完整 JSON 答案
    """
    if req.stream:
        async def event_generator():
            async for token in run_agent_stream(user, req.message):
                # SSE 格式
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        answer = await run_agent(user, req.message)
        return {"response": answer, "user": user.username}
