import os
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel

from app.core.dependencies import get_current_active_user
from app.services.rag_service import RAGService
from config import settings
from models.schemas import DocumentMetadata, DocumentUploadMeta, UserInfo

router = APIRouter()

# Initialize RAGService (consider using FastAPI's dependency injection for better management)
rag_service = RAGService()

class DocumentUploadResponse(BaseModel):
    message: str
    doc_id: str
    chunks: int = 0

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    meta: DocumentUploadMeta = Depends(),
    current_user: UserInfo = Depends(get_current_active_user)
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .txt files are supported.")

    doc_id = str(uuid4())
    file_path = os.path.join(settings.DOCUMENTS_PATH, f"{doc_id}.txt")
    os.makedirs(settings.DOCUMENTS_PATH, exist_ok=True)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # For simplicity, we'll treat the entire file as one chunk for now.
        # In a real RAG system, you'd chunk the document here.
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            chunk_id=f"{doc_id}_chunk_0", # Simplified chunking
            title=meta.title,
            source=meta.source if meta.source else file.filename,
            content=content.decode("utf-8"),
            user_groups=meta.user_groups,
            departments=meta.departments,
            permission_level=meta.permission_level,
        )
        
        rag_service.ingest_documents([doc_metadata])
        
        return DocumentUploadResponse(
            message="Document uploaded and ingested successfully",
            doc_id=doc_id,
            chunks=1 # Simplified
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload or ingest document: {e}")

@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    current_user: UserInfo = Depends(get_current_active_user)
):
    # This is a placeholder. In a real system, you'd query a metadata store
    # to get a list of documents, filtered by user permissions.
    # For now, we'll return a dummy list or try to infer from FAISS.
    
    # A more robust solution would involve a separate metadata store (e.g., a database)
    # that stores DocumentMetadata for all documents, and then filter that list.
    
    # For demonstration, let's assume we can list all documents that the user has access to
    # by performing a dummy query that retrieves all accessible documents.
    
    # This part needs refinement. LlamaIndex's FAISSVectorStore doesn't directly expose
    # a list of all stored DocumentMetadata.
    # We might need to store metadata separately or iterate through the FAISS index (which is not ideal).
    
    # For now, let's return an empty list or a hardcoded example.
    # To properly implement this, we need a way to retrieve all DocumentMetadata objects
    # that were ingested. This usually means a separate database for metadata.
    
    # Let's assume for now that `rag_service` can provide a list of all documents
    # that the current user has access to. This would require a new method in `RAGService`.
    
    # For the purpose of getting the frontend working, I'll return an empty list.
    # This needs to be improved later.
    
    return DocumentListResponse(documents=[])

@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    current_user: UserInfo = Depends(get_current_active_user)
):
    # In a real application, you'd check if the user has permission to delete this document.
    # You'd also remove the document from the file system and from the FAISS index.
    
    file_path = os.path.join(settings.DOCUMENTS_PATH, f"{doc_id}.txt")
    if os.path.exists(file_path):
        os.remove(file_path)
        # Also remove from FAISS index. This requires LlamaIndex's delete capability.
        # rag_service.delete_document(doc_id) # This method needs to be implemented in RAGService
        return
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
