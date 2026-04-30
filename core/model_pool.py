import asyncio
import itertools
import os
import json
from loguru import logger

# 尝试加载 dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_env(key: str, default: str = "") -> str:
    """从环境变量获取值"""
    return os.getenv(key, default)


def _get_int_env(key: str, default: int = 1) -> int:
    """从环境变量获取整数值"""
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def _parse_nodes_from_env() -> list:
    """从环境变量解析 LLM 节点配置"""
    nodes = []
    
    # 检查是否有 JSON 格式的节点配置
    nodes_json = _get_env("LLM_NODES_JSON", "")
    if nodes_json:
        try:
            nodes_config = json.loads(nodes_json)
            for cfg in nodes_config:
                nodes.append(ModelNode(
                    name=cfg.get("name", "Unknown"),
                    url=cfg.get("url", ""),
                    api_key=cfg.get("api_key", ""),
                    model_name=cfg.get("model_name", ""),
                    max_concurrent=cfg.get("max_concurrent", 1),
                ))
            logger.info(f"从 LLM_NODES_JSON 加载了 {len(nodes)} 个节点")
            return nodes
        except json.JSONDecodeError as e:
            logger.warning(f"LLM_NODES_JSON 解析失败: {e}")
    
    # 兼容旧的单节点配置
    llm_url = _get_env("LLM_BASE_URL", "")
    llm_api_key = _get_env("LLM_API_KEY", "")
    llm_model = _get_env("LLM_MODEL_NAME", "gpt-4o-mini")
    
    if llm_url and llm_api_key:
        nodes.append(ModelNode(
            name="Default-LLM",
            url=llm_url,
            api_key=llm_api_key,
            model_name=llm_model,
            max_concurrent=_get_int_env("LLM_MAX_CONCURRENT", 2),
        ))
        logger.info(f"从环境变量加载 LLM 节点: {llm_model}")
    
    return nodes


class ModelNode:
    def __init__(
        self, name: str, url: str, api_key: str, model_name: str, max_concurrent: int = 1
    ):
        self.name = name
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)


class LLMPoolRouter:
    def __init__(self):
        # 优先从环境变量加载节点
        self.nodes = _parse_nodes_from_env()
        
        # 如果没有配置任何节点，使用默认节点
        if not self.nodes:
            logger.warning("未配置 LLM 节点，使用默认配置")
            self.nodes = [
                ModelNode(
                    "Cloud-Backup",
                    "https://xiaoai.plus/v1",
                    "sk-placeholder",
                    "gpt-4o-mini",
                    max_concurrent=1,
                ),
            ]
        
        self.node_cycle = itertools.cycle(range(len(self.nodes)))
        logger.info(f"LLMPoolRouter 初始化完成，共 {len(self.nodes)} 个节点")

    def get_available_node(self):
        start_idx = next(self.node_cycle)
        for i in range(len(self.nodes)):
            idx = (start_idx + i) % len(self.nodes)
            node = self.nodes[idx]
            if not node.semaphore.locked():
                return node
        return self.nodes[start_idx]


router = LLMPoolRouter()
