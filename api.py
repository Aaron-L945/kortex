from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json

from config import Config
from embedder import load_vector_store
from llm_service import get_qwen_llm
from reranker import reranker_tool  # <--- 引入重排工具
from langchain.prompts import ChatPromptTemplate

app = FastAPI(title="企业级 RAG 知识库 API (Rerank 增强版)")

class ChatRequest(BaseModel):
    query: str
    include_think: Optional[bool] = None

@app.post("/chat/complete")
def chat_complete(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    think_mode = request.include_think if request.include_think is not None else Config.INCLUDE_THINK

    try:
        # 1. 粗排 (Retrieval)：先捞出 10 条候选
        db = load_vector_store()
        initial_docs = db.similarity_search(request.query, k=10) 
        
        # 2. 精排 (Rerank)：从 10 条中精选出最相关的 3 条
        print(f"--- 正在进行精排: {request.query} ---")
        final_docs = reranker_tool.rerank(request.query, initial_docs, top_n=3)
        
        context = "\n\n".join([d.page_content for d in final_docs])
        sources = list(set([f"{d.metadata.get('source')} (P{d.metadata.get('page')})" for d in final_docs]))

        # 3. 生成回答
        style_instruction = "请先在 <think> 标签内进行推理。" if think_mode else "直接回答，不含 <think>。"
        prompt_template = """你是一个专业的助手。
        {style_instruction}
        背景知识：{context}
        问题：{question}"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = get_qwen_llm(streaming=False) 
        chain = prompt | llm
        
        response = chain.invoke({
            "context": context, 
            "question": request.query, 
            "style_instruction": style_instruction
        })

        return {
            "status": "success",
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)