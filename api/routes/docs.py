"""
文档管理路由：上传、列表、删除文档。

上传流程：
  1. 接收文件 + 权限元数据
  2. 按段落/句子分 chunk
  3. 向量化 → 写入 FAISS 索引
"""

import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from api.auth import get_current_user
from models.schemas import UserInfo, DocumentMetadata
from rag.indexer import FAISSIndexer

router = APIRouter(prefix="/docs", tags=["Documents"])


def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    简单的滑动窗口分块。
    生产环境建议使用 LlamaIndex 的 SentenceSplitter。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    source: str = Form(""),
    user_groups: str = Form("all"),        # 逗号分隔，如 "admin,hr"
    departments: str = Form("all"),         # 逗号分隔，如 "finance,legal"
    permission_level: int = Form(1),
    current_user: UserInfo = Depends(get_current_user),
):
    """上传文档并写入知识库（需要 admin 权限）"""
    if current_user.permission_level < 3:
        raise HTTPException(status_code=403, detail="上传文档需要机密级别以上权限")

    content_bytes = await file.read()
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = content_bytes.decode("gbk", errors="ignore")

    doc_id = str(uuid.uuid4())
    groups = [g.strip() for g in user_groups.split(",")]
    depts = [d.strip() for d in departments.split(",")]

    chunks_text = simple_chunk_text(text)
    chunks = [
        DocumentMetadata(
            doc_id=doc_id,
            chunk_id=f"{doc_id}_{i}",
            title=title,
            source=source or file.filename,
            content=chunk,
            user_groups=groups,
            departments=depts,
            permission_level=permission_level,
        )
        for i, chunk in enumerate(chunks_text)
    ]

    FAISSIndexer.get().add_chunks(chunks)

    return {
        "doc_id": doc_id,
        "title": title,
        "chunks": len(chunks),
        "permission_level": permission_level,
    }


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: UserInfo = Depends(get_current_user),
):
    """删除文档（需要 admin 权限）"""
    if current_user.user_group != "admin":
        raise HTTPException(status_code=403, detail="仅 admin 可删除文档")
    FAISSIndexer.get().remove_by_doc_id(doc_id)
    return {"deleted": doc_id}


@router.get("/list")
async def list_documents(
    current_user: UserInfo = Depends(get_current_user),
):
    """列出当前用户可访问的所有文档"""
    from permissions.filter import PermissionFilter
    indexer = FAISSIndexer.get()
    accessible = PermissionFilter.filter(indexer.metadata, current_user)

    # 按 doc_id 去重，返回文档级别摘要
    seen = {}
    for m in accessible:
        if m.doc_id not in seen:
            seen[m.doc_id] = {
                "doc_id": m.doc_id,
                "title": m.title,
                "source": m.source,
                "permission_level": m.permission_level,
                "chunks": 0,
            }
        seen[m.doc_id]["chunks"] += 1

    return list(seen.values())
