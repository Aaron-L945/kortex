import json
import numpy as np
import jieba
import faiss
import re
import os
import torch
import gc
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from loguru import logger
from opencc import OpenCC

class HybridRetrieverV2:
    def __init__(self, model_path: str, reranker_tool=None):
        """
        :param model_path: Embedding 模型路径
        :param reranker_tool: 精排模型工具对象 (需实现 .rerank 方法)
        """
        # 加载 Embedding 模型
        self.embed_model = SentenceTransformer(model_path)
        self.reranker = reranker_tool
        
        # 核心索引组件
        self.index = None
        self.bm25 = None
        self.corpus_docs = []
        
        # 辅助工具
        self.cc = OpenCC('t2s') # 繁转简
        self.global_phys_map = {} # 物理窗口索引: {file_id: {chunk_id: doc}}
        
        # 持久化路径 (当前路径)
        self.index_path = "faiss_index.bin"
        self.metadata_path = "corpus_metadata.json"

    def _preprocess_text(self, text: str) -> str:
        """统一预处理：繁转简 + 清洗空白"""
        text = self.cc.convert(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_index(self, jsonl_path: str, force_rebuild: bool = False):
        """构建索引：优先加载磁盘缓存，失败则执行 Embedding"""
        
        # 1. 尝试从当前路径加载缓存
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("⚡ 发现本地索引文件，正在直接加载...")
            self._load_from_disk()
            return

        # 2. 解析原始数据
        logger.info(f"🏗️ 正在从 {jsonl_path} 构建新索引...")
        processed_texts = []
        self.corpus_docs = []
        self.global_phys_map = {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 解析 docid 提取物理坐标 (例如 "12345#2")
                f_id, c_id = data["docid"].split("#")
                clean_content = self._preprocess_text(data["text"])
                
                doc_obj = {
                    "content": clean_content,
                    "metadata": {
                        "docid": data["docid"],
                        "title": data.get("title", ""),
                        "file_id": f_id,
                        "chunk_id": int(c_id),
                    },
                }
                self.corpus_docs.append(doc_obj)
                processed_texts.append(clean_content)
                
                # 构建物理映射用于窗口扩展
                if f_id not in self.global_phys_map:
                    self.global_phys_map[f_id] = {}
                self.global_phys_map[f_id][int(c_id)] = doc_obj

        # 3. 构建 BM25 (使用 Jieba 搜索模式)
        logger.info("构建 BM25 关键词索引...")
        tokenized_corpus = [list(jieba.cut_for_search(t)) for t in processed_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 4. 构建 Faiss 向量索引 (显存优化版)
        logger.info(f"开始向量化处理 ({len(processed_texts)} docs)...")
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            embeddings = self.embed_model.encode(
                processed_texts, 
                normalize_embeddings=True, 
                show_progress_bar=True, 
                batch_size=16,  # 调小 Batch 防止 OOM
                convert_to_numpy=True
            )
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        
        # 5. 保存到磁盘
        self._save_to_disk()
        
        # 清理内存
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✅ 索引构建并保存完成")

    def _save_to_disk(self):
        """持久化索引到磁盘"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.corpus_docs, f, ensure_ascii=False)
        logger.info(f"💾 索引已存至: {self.index_path}")

    def _load_from_disk(self):
        """从磁盘加载索引并还原内存映射"""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.corpus_docs = json.load(f)
        
        # 还原 BM25
        texts = [d['content'] for d in self.corpus_docs]
        tokenized_corpus = [list(jieba.cut_for_search(t)) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 还原物理映射
        for d in self.corpus_docs:
            f_id = d["metadata"]["file_id"]
            c_id = d["metadata"]["chunk_id"]
            if f_id not in self.global_phys_map:
                self.global_phys_map[f_id] = {}
            self.global_phys_map[f_id][int(c_id)] = d
        logger.info("✅ 加载成功，已还原内存索引结构")

    def _rrf_score(self, bm25_ranks: List[int], vector_ranks: List[int], k: int = 60) -> List[int]:
        """Reciprocal Rank Fusion 混合排序"""
        rrf_map = {}
        for rank, idx in enumerate(bm25_ranks):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(vector_ranks):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.0 / (k + rank + 1)
        
        sorted_indices = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_indices]

    def get_expanded_context(self, selected_docs: List[Dict], window_size: int = 3):
        """基于物理坐标的上下文扩展"""
        expanded_map = {}
        for doc in selected_docs:
            fid = doc["metadata"]["file_id"]
            center_cid = doc["metadata"]["chunk_id"]
            
            # 扩展左右邻居
            for i in range(center_cid - window_size, center_cid + window_size + 1):
                target_doc = self.global_phys_map.get(fid, {}).get(i)
                if target_doc:
                    expanded_map[target_doc["metadata"]["docid"]] = target_doc

        # 按物理顺序重新排序，确保逻辑连贯
        final_list = list(expanded_map.values())
        final_list.sort(key=lambda x: (x["metadata"]["file_id"], x["metadata"]["chunk_id"]))
        return final_list

    def pipeline(self, query: str, top_k: int = 100, top_n: int = 15, window_size: int = 3):
        """一键检索流水线"""
        query = self._preprocess_text(query)
        
        # 1. 粗排
        # BM25 路
        query_tokens = list(jieba.cut_for_search(query))
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        # Vector 路
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        _, vector_idx = self.index.search(q_emb.astype("float32"), top_k)
        vector_idx = vector_idx[0].tolist()

        # RRF 融合
        combined_indices = self._rrf_score(bm25_idx, vector_idx)[:top_k]
        all_candidates = [self.corpus_docs[i] for i in combined_indices]

        # 2. 精排
        if self.reranker and all_candidates:
            docs_after_rerank = self.reranker.rerank(query, all_candidates, top_n=top_n)
        else:
            docs_after_rerank = all_candidates[:top_n]

        # 3. 窗口扩展
        final_context = self.get_expanded_context(docs_after_rerank, window_size=window_size)

        return final_context