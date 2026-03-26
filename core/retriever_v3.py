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
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger
from opencc import OpenCC

class HybridRetrieverV3:
    def __init__(self, model_path: str, reranker_path: str = None):
        """
        :param model_path: Embedding 模型路径 (Bi-Encoder)
        :param reranker_path: Reranker 模型路径 (Cross-Encoder)
        """
        # 1. 加载 Embedding 模型
        # 如果显存依然紧张，可将 device 改为 'cpu'
        self.embed_model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. 加载 Reranker 模型
        self.reranker = None
        if reranker_path:
            logger.info(f"🚀 正在加载精排模型: {reranker_path}")
            self.reranker = CrossEncoder(
                reranker_path, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                max_length=512  # 限制最大长度以节省显存
            )
        
        self.index = None
        self.bm25 = None
        self.corpus_docs = []
        self.cc = OpenCC('t2s')
        self.global_phys_map = {}
        
        # 持久化路径
        self.index_path = "faiss_index.bin"
        self.metadata_path = "corpus_metadata.json"

    def _preprocess_text(self, text: str) -> str:
        text = self.cc.convert(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_index(self, jsonl_path: str, force_rebuild: bool = False):
        """构建索引逻辑"""
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("⚡ 发现本地缓存，直接加载...")
            self._load_from_disk()
            return

        logger.info(f"🏗️ 正在构建新索引...")
        processed_texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
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
                
                if f_id not in self.global_phys_map:
                    self.global_phys_map[f_id] = {}
                self.global_phys_map[f_id][int(c_id)] = doc_obj

        # BM25 初始化
        tokenized_corpus = [list(jieba.cut_for_search(t)) for t in processed_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Faiss 向量化
        gc.collect()
        torch.cuda.empty_cache()
        embeddings = self.embed_model.encode(
            processed_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        self._save_to_disk()

    def _save_to_disk(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.corpus_docs, f, ensure_ascii=False)

    def _load_from_disk(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.corpus_docs = json.load(f)
        texts = [d['content'] for d in self.corpus_docs]
        self.bm25 = BM25Okapi([list(jieba.cut_for_search(t)) for t in texts])
        for d in self.corpus_docs:
            f_id, c_id = d["metadata"]["file_id"], d["metadata"]["chunk_id"]
            if f_id not in self.global_phys_map: self.global_phys_map[f_id] = {}
            self.global_phys_map[f_id][int(c_id)] = d

    def _rrf_score(self, bm25_ranks: List[int], vector_ranks: List[int], k: int = 60) -> List[int]:
        """混合评分：增加 BM25 权重以应对事实性 Query"""
        rrf_map = {}
        bm25_weight = 1.5 # 增强关键词匹配的权重
        
        for rank, idx in enumerate(bm25_ranks):
            rrf_map[idx] = rrf_map.get(idx, 0) + bm25_weight * (1.0 / (k + rank + 1))
        for rank, idx in enumerate(vector_ranks):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.0 / (k + rank + 1)
            
        return [item[0] for item in sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)]

    def get_expanded_context(self, selected_docs: List[Dict], window_size: int = 3):
        """窗口扩展：补全上下文逻辑"""
        expanded_map = {}
        for doc in selected_docs:
            fid, center_cid = doc["metadata"]["file_id"], doc["metadata"]["chunk_id"]
            # 这里的范围是 [center-w, center+w]
            for i in range(center_cid - window_size, center_cid + window_size + 1):
                target_doc = self.global_phys_map.get(fid, {}).get(i)
                if target_doc:
                    expanded_map[target_doc["metadata"]["docid"]] = target_doc
        
        res = list(expanded_map.values())
        # 按文档顺序排列，保证 LLM 读取逻辑正常
        res.sort(key=lambda x: (x["metadata"]["file_id"], x["metadata"]["chunk_id"]))
        return res

    def pipeline(self, query: str, top_k: int = 60, top_n: int = 5, window_size: int = 3):
        """核心检索流水线"""
        query = self._preprocess_text(query)
        
        # 1. 粗排
        query_tokens = list(jieba.cut_for_search(query))
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        _, vector_idx = self.index.search(q_emb.astype("float32"), top_k)
        vector_idx = vector_idx[0].tolist()

        # RRF 融合候选集
        combined_indices = self._rrf_score(bm25_idx, vector_idx, k=60)[:top_k]
        candidates = [self.corpus_docs[i] for i in combined_indices]

        # 2. 精排 (Reranker)
        if self.reranker and candidates:
            pairs = [[query, doc['content']] for doc in candidates]
            
            # 精排前清理显存，防止 OOM
            gc.collect()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                # 显存安全：batch_size=4
                scores = self.reranker.predict(pairs, batch_size=4, show_progress_bar=False)
            
            # 将精排分数写回并排序
            for i, score in enumerate(scores):
                candidates[i]['temp_score'] = float(score)
            
            # 严格按精排分数从高到低排列 (reverse=True)
            docs_after_rerank = sorted(candidates, key=lambda x: x['temp_score'], reverse=True)[:top_n]
        else:
            docs_after_rerank = candidates[:top_n]

        # 3. 上下文扩展
        return self.get_expanded_context(docs_after_rerank, window_size=window_size)