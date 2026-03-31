import time
import re
from langchain.prompts import ChatPromptTemplate
from llm_service import get_qwen_llm
from embedder import load_vector_store
from core.retriever_without_windows import HybridRetriever
from reranker import reranker_tool
from loguru import logger


# --- 核心逻辑：从真实向量库构建上下文 ---
def get_real_context(query: str):
    """
    使用验收通过的最优参数构建上下文
    """
    db = load_vector_store()
    retriever = HybridRetriever(db)

    # 1. 暴力召回
    hybrid_results = retriever.search(query, top_k=100)
    all_candidates = [res[0] for res in hybrid_results]

    # 2. 精排锚点
    valid_docs = reranker_tool.rerank(query, all_candidates, top_n=15)

    # 3. 窗口扩展 (Window Size = 3)
    expanded_map = {}
    pool = {
        f"{d.metadata.get('file_id')}_{d.metadata.get('chunk_id')}": d
        for d in all_candidates
    }

    for doc in valid_docs:
        f_id = doc.metadata.get("file_id")
        c_id = doc.metadata.get("chunk_id")
        if c_id is None:
            continue

        for i in range(c_id - 3, c_id + 4):  # Window Size 3
            key = f"{f_id}_{i}"
            if key in pool:
                expanded_map[key] = pool[key]

    # 4. 物理排序还原逻辑链
    context_docs = list(expanded_map.values())
    context_docs.sort(
        key=lambda x: (x.metadata.get("file_id"), x.metadata.get("chunk_id"))
    )

    # 5. 格式化输出
    context_parts = []
    for d in context_docs:
        m = d.metadata
        context_parts.append(
            f"--- [文件: {m.get('source')} | CID: {m.get('chunk_id')} | P{m.get('page')}] ---\n{d.page_content}"
        )

    return "\n\n".join(context_parts)


def run_llm_validation(query):
    print("\n🚀 " + "=" * 25 + " 真实 RAG 链路 + LLM 还原验证 " + "=" * 25 + "\n")

    # 1. 获取真实数据
    print(f"📡 正在从向量库检索真实上下文 (TopK=100, Window=3)...")
    real_context = get_real_context(query)

    if not real_context:
        print("❌ 未能获取到上下文，请检查向量库数据。")
        return

    # 2. 构造高强度系统 Prompt
    # 建议在 api.py 中使用的通用专家 Prompt
    prompt = ChatPromptTemplate.from_template(
        """
    你是一个严谨的 Nuage 网络技术专家。请根据提供的背景知识回答用户的查询。

    【操作准则】：
    1. **完整性优先**：在回答流程类问题（如升级步骤）时，请务必检索上下文中的所有编号和动作。注意：由于文档切片原因，步骤可能分布在不同的片段中，请通过 CID 和页码逻辑进行拼装。
    2. **忠于原著但灵活处理**：
    - 必须保留原始文档中的关键技术细节（如 linux 命令、virsh 命令、特定内核版本号）。
    - 如果用户要求“完整步骤”，请按顺序排列。若背景知识中某步缺失，请基于上下文推断或诚实标注，严禁编造。
    3. **技术准确性**：
    4. **输出风格**：使用清晰的标题、加粗关键点和代码块（用于命令）。

    背景知识：
    {context}

    用户问题：{question}
    """
    )

    # 3. 初始化 LLM
    llm = get_qwen_llm(streaming=False)
    chain = prompt | llm

    # 4. 执行测试
    print(f"📝 测试 Query: {query}")
    print("⏳ LLM 正在分析真实长上下文并生成答案...\n")

    start_time = time.time()
    response = chain.invoke(
        {
            "context": real_context,
            "question": query,
            "style_instruction": "请先在 <think> 中列出你找到的所有步骤编号及其核心动作",
        }
    )

    # 5. 结果展示
    print("-" * 80)
    print(response.content)
    print("-" * 80)

    # 自动化检查
    content = response.content.lower()
    missing = []
    for i in range(1, 14):
        # 匹配 "step 1" 或 "1."
        if not re.search(rf"(step\s*{i}|{i}\.)", content):
            missing.append(str(i))

    if not missing:
        print(
            f"\n🎉 验证成功！LLM 在真实环境下还原了 1-13 步。耗时: {time.time()-start_time:.2f}s"
        )
    else:
        print(f"\n❌ 验证失败：缺失步骤 {', '.join(missing)}。")
        print("💡 建议：查看 <think> 部分，确认这些步骤是否真的在检索到的 CID 中。")


if __name__ == "__main__":
    # run_llm_validation(query="如何升级 VSD？请给出完整步骤。")
    run_llm_validation(query="python 怎么读取文件？")
