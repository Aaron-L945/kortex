# from langchain.prompts import ChatPromptTemplate
# from embedder import load_vector_store
# from llm_service import get_qwen_llm  # <--- 确保从 service 导入

# # 升级版 Prompt：强制要求模型展示思考过程
# PROMPT_TEMPLATE = """
# 你是一个专业、严谨的企业知识库助手。你拥有深度思考的能力。

# 【回答要求】
# 1. 首先，请在 <think> 标签内详细分析背景知识与用户问题的关联性，列出你的推理逻辑。
# 2. 然后，根据背景知识给出正式回答。
# 3. 如果背景知识中没有相关信息，请在 <think> 后直接回答“抱歉，在现有知识库中未找到相关内容”。
# 4. 必须在回答末尾标注引用的文件来源。

# 【背景知识】
# {context}

# ---
# 【用户问题】
# {question}
# """

# def ask_question(query):
#     # 1. 加载向量库并检索
#     db = load_vector_store()
#     docs = db.similarity_search(query, k=3)
    
#     # 2. 构造上下文并记录来源
#     context_list = []
#     sources = set()
#     for doc in docs:
#         context_list.append(doc.page_content)
#         # 获取文件名和页码元数据
#         src = doc.metadata.get('source', '未知文件')
#         pg = doc.metadata.get('page', '?')
#         sources.add(f"{src} (P{pg})")
    
#     context = "\n\n".join(context_list)
    
#     # 3. 从 Service 获取流式 LLM
#     llm = get_qwen_llm(streaming=True)
    
#     # 4. 构造 Prompt 链
#     prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     chain = prompt | llm
    
#     print("\n" + "="*20 + " Agent 正在思考并回答 " + "="*20 + "\n")
    
#     # 5. 流式输出
#     # 因为开启了 streaming，这里用 stream 方法实时打印
#     full_response = ""
#     for chunk in chain.stream({"context": context, "question": query}):
#         content = chunk.content
#         full_response += content
#         print(content, end="", flush=True)
    
#     print("\n\n" + "-"*30)
#     print("【参考来源】:")
#     for s in sources:
#         print(f"- {s}")
#     print("="*60 + "\n")

# if __name__ == "__main__":
#     import sys
#     # 支持命令行参数直接提问，或者交互式提问
#     if len(sys.argv) > 1:
#         query = " ".join(sys.argv[1:])
#         ask_question(query)
#     else:
#         while True:
#             q = input("请输入您的问题 (输入 'exit' 退出): ")
#             if q.lower() == 'exit':
#                 break
#             if q.strip():
#                 ask_question(q)



import re
from config import Config # 导入配置类
from langchain.prompts import ChatPromptTemplate
from embedder import load_vector_store
from llm_service import get_qwen_llm

PROMPT_TEMPLATE = """
你是一个专业、严谨的企业知识库助手。你拥有深度思考的能力。

【回答要求】
1. 根据背景知识给出正式回答。
2. 如果背景知识中没有相关信息，请直接回答“抱歉，在现有知识库中未找到相关内容”。
3. 必须在回答末尾标注引用的文件来源。

【背景知识】
{context}

---
【用户问题】
{question}
"""

def ask_question(query, include_think=None):
    # 如果调用时没传参数，则使用 .env 中的配置
    if include_think is None:
        include_think = Config.INCLUDE_THINK
        
    # 1. 检索
    db = load_vector_store()
    docs = db.similarity_search(query, k=1)
    context = "\n\n".join([d.page_content for d in docs])
    
    # 2. 动态指令
    if include_think:
        style_instruction = "请先在 <think> 标签内进行逻辑推理，然后再回答。"
    else:
        style_instruction = "请直接给出回答，不要包含任何思考过程或 <think> 标签。"

    # 3. 构造并调用
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_qwen_llm(streaming=True)
    chain = prompt | llm
    
    print(f"\n[系统通知] 当前运行模式: {'思考模式' if include_think else '简洁模式'}\n")
    
    print(f"--- 检索到的上下文长度: {len(context)} ---")
    print(f"--- 正在调用 LLM，请稍候... ---")
    full_response = ""
    for chunk in chain.stream({
        "context": context, 
        "question": query, 
        "style_instruction": style_instruction
    }):
        content = chunk.content
        full_response += content
        print(content, end="", flush=True)

    return full_response

if __name__ == "__main__":
    user_query = input("请输入问题: ")
    ask_question(user_query)