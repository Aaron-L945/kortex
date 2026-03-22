"""
FAISS 索引管理层。

存储结构：
  data/faiss_index/
  ├── index.faiss        ← FAISS 向量索引
  └── metadata.json      ← 每条向量对应的 DocumentMetadata（列表，按 index 对齐）

由于 FAISS 不原生支持 metadata 过滤，我们：
  1. 先从 FAISS 取出 top-K * 扩展倍数 的候选结果
  2. 在 retriever 层用 PermissionFilter 过滤
  3. 返回最终 top-K
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from models.schemas import DocumentMetadata
from rag.embedder import Embedder
from config import settings


INDEX_FILE = Path(settings.FAISS_INDEX_PATH) / "index.faiss"
META_FILE = Path(settings.FAISS_INDEX_PATH) / "metadata.json"


class FAISSIndexer:
    _instance: "FAISSIndexer | None" = None

    def __init__(self):
        self.embedder = Embedder.get()
        self.dim = self.embedder.dim
        Path(settings.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        self._load_or_create()

    @classmethod
    def get(cls) -> "FAISSIndexer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─── 持久化 ──────────────────────────────────────────────────────────────

    def _load_or_create(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata: List[DocumentMetadata] = [
                DocumentMetadata(**m) for m in raw
            ]
            print(f"[FAISSIndexer] 加载索引，共 {self.index.ntotal} 条向量")
        else:
            # Inner product（配合 L2 归一化向量 = cosine similarity）
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata: List[DocumentMetadata] = []
            print("[FAISSIndexer] 创建新索引")

    def save(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in self.metadata], f, ensure_ascii=False, indent=2)

    # ─── 写入 ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentMetadata]):
        """将一批文档 chunks 向量化并写入 FAISS"""
        texts = [c.content for c in chunks]
        vecs = self.embedder.embed_passages(texts)             # (N, dim)
        self.index.add(vecs)
        self.metadata.extend(chunks)
        self.save()
        print(f"[FAISSIndexer] 新增 {len(chunks)} 条，总计 {self.index.ntotal} 条")

    def remove_by_doc_id(self, doc_id: str):
        """删除指定文档的所有 chunks（重建索引）"""
        keep_indices = [
            i for i, m in enumerate(self.metadata) if m.doc_id != doc_id
        ]
        if len(keep_indices) == len(self.metadata):
            return  # 没有要删除的

        kept_meta = [self.metadata[i] for i in keep_indices]
        if kept_meta:
            kept_texts = [m.content for m in kept_meta]
            vecs = self.embedder.embed_passages(kept_texts)
            new_index = faiss.IndexFlatIP(self.dim)
            new_index.add(vecs)
        else:
            new_index = faiss.IndexFlatIP(self.dim)

        self.index = new_index
        self.metadata = kept_meta
        self.save()

    # ─── 查询 ─────────────────────────────────────────────────────────────────

    def search(
        self, query_vec: np.ndarray, top_k: int = 20
    ) -> List[Tuple[DocumentMetadata, float]]:
        """
        返回 (metadata, score) 列表，score 越高越相似。
        top_k 设大一些，交给上层 PermissionFilter 过滤后再截断。
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results
