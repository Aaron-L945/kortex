# core/tools.py
from langchain.tools import tool


@tool
def get_weather(city: str):
    """查询指定城市的天气。
    Args:
        city: 城市名称，如“北京”、“上海”
    """
    # 这里可以是真实的 API 调用，比如和风天气或高德
    return f"{city}的天气晴朗，气温 25°C，适合户外活动。"


@tool
def search_knowledge_base(query: str):
    """当用户询问专业知识、事实、特定文档内容或人物作品（如《网络独立宣言》）时调用此工具。"""
    # 这里我们稍后在主逻辑中关联你的 HybridRetrieverV3
    return "SEARCH_PENDING"  # 占位符，由主逻辑拦截并执行 RAG


# 工具列表
TOOLS = [get_weather, search_knowledge_base]
