from langchain_openai import ChatOpenAI


def get_qwen_llm(streaming=True):
    # return ChatOpenAI(
    #     model="/models/Qwen3.5-35B-A3B-FP8", # 替换为你本地实际的模型名称
    #     openai_api_key="qwen", # Ollama 不需要 key，随便填
    #     openai_api_base="http://10.66.196.31:20351/v1",
    #     temperature=0.1, # 知识库问答建议低随机性
    #     streaming=streaming
    # )
    return ChatOpenAI(
        model="qwen3-max",  # 替换为你本地实际的模型名称
        openai_api_key="sk-78fe3d06e3ffdbb0f733fa067fe4dacd",  # Ollama 不需要 key，随便填
        openai_api_base="https://apis.iflow.cn/v1",
        temperature=0,  # 知识库问答建议低随机性
        streaming=streaming,
        verbose=True
    )
