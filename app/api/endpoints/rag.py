from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.core.dependencies import get_current_active_user
from app.services.rag_service import RAGService
from models.schemas import DocumentMetadata, QueryRequest, QueryResponse, UserInfo

router = APIRouter()

# Initialize RAGService (consider using FastAPI's dependency injection for better management)
rag_service = RAGService()

@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_documents_endpoint(
    documents: List[DocumentMetadata],
    current_user: UserInfo = Depends(get_current_active_user)
):
    # In a real application, you might want to check if the user has permission to ingest documents
    # For now, any authenticated user can ingest.
    # You might also want to store the documents in a persistent storage before ingesting into FAISS.
    
    # Filter documents based on user's permission level if needed before ingestion
    # For simplicity, we'll ingest all provided documents.
    
    rag_service.ingest_documents(documents)
    return {"message": "Documents ingestion initiated."}

@router.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(
    request: QueryRequest,
    current_user: UserInfo = Depends(get_current_active_user)
):
    retrieved_docs = rag_service.retrieve(request.query, current_user, request.top_k)
    return QueryResponse(results=retrieved_docs)
