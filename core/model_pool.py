import asyncio
import itertools
from loguru import logger


class ModelNode:
    def __init__(
        self, name: str, url: str, api_key: str, model_name: str, max_concurrent: int = 1
    ):
        self.name = name
        self.url = url  # 保持 http://10.66.196.31:20351/v1
        self.api_key = api_key
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)


class LLMPoolRouter:
    def __init__(self):
        self.nodes = [
            ModelNode(
                "Local-vLLM",
                "http://10.66.196.31:20351/v1",
                "local-token",
                "/models/Qwen3.5-35B-A3B-FP8",
                max_concurrent=8,
            ),
            ModelNode(
                "Cloud-Backup",
                "https://apis.iflow.cn/v1",
                "sk-b51aac8fea8dcb4fb574275c123f960e",
                "qwen3-max",
                max_concurrent=1,
            ),
        ]
        self.node_cycle = itertools.cycle(range(len(self.nodes)))

    def get_available_node(self):
        start_idx = next(self.node_cycle)
        for i in range(len(self.nodes)):
            idx = (start_idx + i) % len(self.nodes)
            node = self.nodes[idx]
            if not node.semaphore.locked():
                return node
        return self.nodes[start_idx]


router = LLMPoolRouter()
