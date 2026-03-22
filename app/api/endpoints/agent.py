from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_current_active_user
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from models.schemas import AgentChatRequest, AgentChatResponse, UserInfo

router = APIRouter()

# Initialize RAGService and AgentService (consider using FastAPI's dependency injection for better management)
rag_service = RAGService()
agent_service = AgentService(rag_service)

@router.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(
    request: AgentChatRequest,
    current_user: UserInfo = Depends(get_current_active_user)
):
    response = agent_service.chat(request.message, current_user)
    return AgentChatResponse(response=response)
