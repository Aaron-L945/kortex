"""
向量嵌入层：使用 BAAI/bge-large-zh-v1.5 生成中文向量。

BGE 模型建议：
  - 检索 Query 加前缀 "为这个句子生成表示以用于检索相关文章："
  - 文档 Passage 不加前缀
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings


class Embedder:
    _instance: "Embedder | None" = None

    def __init__(self):
        print(f"[Embedder] 加载模型: {settings.EMBED_MODEL}")
        self.model = SentenceTransformer(settings.EMBED_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_query(self, text: str) -> np.ndarray:
        """Query 端：加 BGE instruction 前缀"""
        instruction = "为这个句子生成表示以用于检索相关文章："
        vec = self.model.encode(
            instruction + text,
            normalize_embeddings=True,
        )
        return vec.astype("float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """文档端：批量嵌入，不加前缀"""
        vecs = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vecs.astype("float32")
