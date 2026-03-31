import re
import time
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import Config
from embedder import load_vector_store
from core.retriever_without_windows import HybridRetriever 
from llm_service import get_qwen_llm
from reranker import reranker_tool
from langchain.prompts import ChatPromptTemplate

# --- 0. 全局初始化 ---
db = load_vector_store()
retriever = HybridRetriever(db) 

app = FastAPI(title="企业级 RAG 知识库 API (句子窗口补全版)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Pydantic 数据模型 (必须放在路由之前) ---

class ChatRequest(BaseModel):
    query: str
    include_think: Optional[bool] = None

class ChatResponse(BaseModel):
    status: str
    answer: str
    sources: List[str]
    error: Optional[str] = None

# --- 2. 辅助工具函数 ---

def get_expanded_context(selected_docs, all_candidates, window_size=1):
    """
    通过 chunk_id 寻找邻居，补全跨页断层
    """
    expanded_map = {}
    # 建立邻居索引池：key = file_id_chunk_id
    pool = {f"{d.metadata.get('file_id')}_{d.metadata.get('chunk_id')}": d for d in all_candidates}
    
    for doc in selected_docs:
        f_id = doc.metadata.get('file_id')
        c_id = doc.metadata.get('chunk_id')
        
        # 抓取当前片段及其前后 window_size 个邻居
        for i in range(c_id - window_size, c_id + window_size + 1):
            neighbor_key = f"{f_id}_{i}"
            if neighbor_key in pool:
                expanded_map[neighbor_key] = pool[neighbor_key]
    
    # 物理顺序排序
    final_docs = list(expanded_map.values())
    final_docs.sort(key=lambda x: (x.metadata.get('file_id'), x.metadata.get('chunk_id')))
    return final_docs

def extract_thought(full_text: str):
    pattern = r"<(think|thinking)>(.*?)</(think|thinking)>"
    match = re.search(pattern, full_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        thought = match.group(2).strip()
        answer = re.sub(pattern, "", full_text, flags=re.DOTALL | re.IGNORECASE).strip()
        return thought, answer
    return None, full_text

# --- 3. 核心 API 路由 ---

@app.post("/chat/complete", response_model=ChatResponse)
async def chat_complete(request: ChatRequest):
    start_time = time.time()
    think_mode = request.include_think if request.include_think is not None else Config.INCLUDE_THINK

    try:
        # Step 1: 混合检索
        hybrid_results = retriever.search(request.query, top_k=30)
        all_candidates = [res[0] for res in hybrid_results]
        
        # Step 2: Reranker 精排
        valid_docs = reranker_tool.rerank(request.query, all_candidates, top_n=8)
        
        # Step 3: 句子窗口扩展 (解决步骤 12/13 缺失的关键)
        context_docs = get_expanded_context(valid_docs, all_candidates, window_size=1)
        
        # Step 4: 构造上下文
        context_parts = []
        for d in context_docs:
            m = d.metadata
            clean_content = re.sub(r'\n\s*\d+\s*\n', '\n', d.page_content) # 清理孤立页码行
            context_parts.append(f"--- [文件: {m.get('source')} | 第 {m.get('page')} 页] ---\n{clean_content}")
            
        context = "\n\n".join(context_parts)
        sources = sorted(list(set([f"{d.metadata.get('source')} (P{d.metadata.get('page')})" for d in context_docs])))

        # Step 5: Prompt 策略
        prompt = ChatPromptTemplate.from_template("""
        你是一个严谨的技术专家。请根据背景知识回答。
        
        【要求】：
        1. 必须完整梳理所有操作步骤（如 1-13 步）。
        2. 背景知识按物理顺序排列，请特别注意翻页后的后续步骤。
        3. 严禁断言“步骤已结束”，除非背景知识明确显示。
        
        {style_instruction}
        
        背景知识：
        {context}
        
        问题：{question}
        """)

        llm = get_qwen_llm(streaming=False)
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": request.query,
            "style_instruction": "请先在 <think> 中核查 11 步之后是否有 12、13 步。" if think_mode else "",
        })
        
        thought, answer = extract_thought(response.content)
        latency = time.time() - start_time
        logger.info(f"✅ 响应成功 | 耗时: {latency:.2f}s")

        return ChatResponse(status="success", answer=answer, sources=sources)

    except Exception as e:
        logger.exception(f"异常: {str(e)}")
        return ChatResponse(status="error", answer="系统故障", sources=[], error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)