from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from llm_service import get_qwen_llm


# 1. 初始化大模型
llm = get_qwen_llm(streaming=True)


# 2. 初始化窗口记忆
# k=2 表示只保留最近的 2 轮对话（即 2 次提问 + 2 次回答）
memory = ConversationBufferWindowMemory(k=2)

# 3. 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 设置为 True 可以看到后台是如何提取历史记录的
)

# 4. 进行多轮对话测试
print(conversation.predict(input="你好，我的暗号是‘紫蝴蝶778’"))
print(conversation.predict(input="我喜欢吃苹果。"))
print(conversation.predict(input="我想吃火锅"))
print(conversation.predict(input="今天天气不错"))

# 第7轮对话：此时第一轮（名字）应该已经被移出“窗口”了
print(conversation.predict(input="你还记得我的暗号是什么吗？"))