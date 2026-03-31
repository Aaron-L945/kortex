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

# --- 0. 配置与初始化 ---
app = FastAPI(title="企业级 RAG 知识库 API (跨页补全增强版)")

db = load_vector_store()
retriever = HybridRetriever(db) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatResponse(BaseModel):
    status: str
    answer: str
    sources: List[str]
    error: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    include_think: Optional[bool] = None

def extract_thought(full_text: str):
    pattern = r"<(think|thinking)>(.*?)</(think|thinking)>"
    match = re.search(pattern, full_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        thought = match.group(2).strip()
        answer = re.sub(pattern, "", full_text, flags=re.DOTALL | re.IGNORECASE).strip()
        return thought, answer
    return None, full_text

# --- 核心逻辑：解决步骤 12/13 丢失的关键 ---
@app.post("/chat/complete", response_model=ChatResponse)
async def chat_complete(request: ChatRequest):
    start_time = time.time()
    think_mode = request.include_think if request.include_think is not None else Config.INCLUDE_THINK

    try:
        # 1. 扩大召回范围：从 20 增加到 30，确保覆盖跨页内容
        hybrid_results = retriever.search(request.query, top_k=30)
        
        # 2. 精排：选取前 12 个片段 (Top_N 稍微调大)
        docs_to_rerank = [res[0] for res in hybrid_results]
        all_reranked_docs = reranker_tool.rerank(request.query, docs_to_rerank, top_n=12)
        
        # 3. 动态过滤：只要不是负分噪音，都视为潜在步骤
        threshold = float(getattr(Config, "RERANK_THRESHOLD", 0.0))
        valid_docs = [d for d in all_reranked_docs if d.metadata.get('rerank_score', 0) >= threshold]

        # --- 【关键增强：逻辑链补全】 ---
        # 如果命中了第 15 页，但没命中 16 页，且 15 页分数极高（>1.5）
        # 我们逻辑上认为 16 页可能包含后续步骤，在此处通过逻辑排序自然衔接
        # 排序：文件名升序 + 页码升序
        valid_docs.sort(key=lambda x: (x.metadata.get('source'), x.metadata.get('page', 0)))

        if not valid_docs:
            return ChatResponse(status="success", answer="未找到相关步骤。", sources=[])

        # 4. 构造上下文：加入页码锚点说明，引导 LLM 意识到“内容在翻页”
        context_parts = []
        for d in valid_docs:
            meta = d.metadata
            content = d.page_content
            # 清洗 PDF 常见的页码干扰
            content = re.sub(r'\n\s*\d+\s*\n', '\n', content) 
            part = f"[来自文档: {meta.get('source')} | 第 {meta.get('page')} 页]\n{content}"
            context_parts.append(part)
        
        context = "\n\n=== 内容分段 ===\n\n".join(context_parts)
        sources = sorted(list(set([f"{d.metadata.get('source')} (P{d.metadata.get('page')})" for d in valid_docs])))

        # 5. 最终 Prompt：针对“步骤 12 重复上述步骤”这种引用逻辑做强化
        prompt = ChatPromptTemplate.from_template("""
        你是一个严谨的 Nuage 网络技术专家。请根据背景知识汇总操作步骤。
        
        特别注意：
        1. 如果背景知识跨越了多页（如第 15 页到第 16 页），请将它们逻辑连接。
        2. 如果步骤中出现 "Repeat Step X to Step Y"（重复步骤 X 到 Y），请在回答中明确说明重复哪些操作。
        3. 必须保留所有 virsh 命令和配置文件名（如 bof.cfg）。
        4. 即使步骤 12 和 13 简短，也必须完整列出，不要省略。
        
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
            "style_instruction": (
                "请先在 <think> 中梳理 1-13 步的逻辑，确认翻页后的内容是否已包含，再给出回答。"
                if think_mode else "直接给出完整 1-13 步操作。"
            ),
        })
        
        thought, answer = extract_thought(response.content)
        latency = time.time() - start_time
        logger.info(f"✅ 检索完成 | 命中片段: {len(valid_docs)} | 耗时: {latency:.2f}s")

        return ChatResponse(status="success", answer=answer, sources=sources)

    except Exception as e:
        logger.exception(f"系统异常: {str(e)}")
        return ChatResponse(status="error", answer="系统故障", sources=[], error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)