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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"⚙️ 初始化检索器，使用设备: {self.device}")
        
        self.embed_model = SentenceTransformer(model_path, device=self.device)
        self.reranker = None
        if reranker_path:
            logger.info(f"🚀 加载精排模型: {reranker_path}")
            self.reranker = CrossEncoder(reranker_path, device=self.device, max_length=512)

        self.index = None
        self.bm25 = None
        self.corpus_docs = []
        self.cc = OpenCC("t2s")
        
        self.index_path = "faiss_index.bin"
        self.metadata_path = "corpus_metadata.json"
        # 🚀 修复点：定义审计日志与失败日志路径
        self.audit_path = "retrieval_audit.jsonl"
        self.failure_log_path = "failure_analysis.jsonl"
        
        # 初始化时清空上一次的失败日志
        if os.path.exists(self.failure_log_path):
            os.remove(self.failure_log_path)

    def _preprocess_text(self, text: str) -> str:
        text = self.cc.convert(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_index(self, jsonl_path: str, force_rebuild: bool = False):
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("⚡ 加载本地索引...")
            self._load_from_disk()
            return

        logger.info("🏗️ 构建新索引...")
        processed_texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                clean_content = self._preprocess_text(data["text"])
                doc_obj = {
                    "content": clean_content,
                    "metadata": {
                        "docid": data["docid"],
                        "title": data.get("title", ""),
                        "corpus_idx": i
                    },
                }
                self.corpus_docs.append(doc_obj)
                processed_texts.append(clean_content)

        self.bm25 = BM25Okapi([list(jieba.cut_for_search(t)) for t in processed_texts])
        embeddings = self.embed_model.encode(processed_texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        self._save_to_disk()

    def _save_to_disk(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_docs, f, ensure_ascii=False)

    def _load_from_disk(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.corpus_docs = json.load(f)
        texts = [d["content"] for d in self.corpus_docs]
        self.bm25 = BM25Okapi([list(jieba.cut_for_search(t)) for t in texts])

    def _get_context_window(self, corpus_idx: int, window_size: int = 1) -> str:
        start = max(0, corpus_idx - window_size)
        end = min(len(self.corpus_docs), corpus_idx + window_size + 1)
        return " ".join([self.corpus_docs[i]["content"] for i in range(start, end)])
    
    def _expand_query(self, query):
        expansions = [query]

        # 简单规则增强（可自行扩展）
        if "比例" in query:
            expansions.append(query + " 占比 百分比")
        if "时间" in query:
            expansions.append(query + " 日期 年份")
        if "有哪些" in query:
            expansions.append(query + " 包括 什么")

        return list(set(expansions))

    def _rrf_fusion(self, bm25_idx, vector_idx, k=60):
        scores = {}

        for rank, idx in enumerate(bm25_idx):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        for rank, idx in enumerate(vector_idx):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    def pipeline(self, query: str, top_k: int = 300, rerank_k: int = 100, top_n: int = 5, target_id: str = None):

        query_clean = query.replace("·", " ").replace("•", " ").strip()
        t_id_str = str(target_id).strip().lower() if target_id else ""

        # ===============================
        # 阶段 A: 多 Query + 多路召回
        # ===============================
        queries = self._expand_query(query_clean)

        bm25_all = []
        vector_all = []

        for q in queries:
            tokens = list(jieba.cut_for_search(q))
            scores = self.bm25.get_scores(tokens)
            bm25_idx = np.argsort(scores)[::-1][:top_k]
            bm25_all.extend(bm25_idx)

            q_emb = self.embed_model.encode([f"query: {q}"], normalize_embeddings=True)
            _, v_idx = self.index.search(q_emb.astype("float32"), top_k)
            vector_all.extend(v_idx[0])

        # RRF融合
        fused_idx = self._rrf_fusion(bm25_all, vector_all)

        candidates_idx = fused_idx[:top_k]
        candidates = [self.corpus_docs[i] for i in candidates_idx]

        if not candidates:
            return []

        # ===============================
        # 阶段 B: Rerank（两阶段）
        # ===============================
        scored_items = []

        if self.reranker:

            # 👉 只 rerank 前 rerank_k
            rerank_candidates = candidates[:rerank_k]

            pairs = [
                [
                    query_clean,
                    f"{c['metadata'].get('title','')}\n{c['content']}"
                ]
                for c in rerank_candidates
            ]

            gc.collect()
            torch.cuda.empty_cache()

            with torch.inference_mode():
                scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)

            for i, score in enumerate(scores):
                f_score = float(score)

                # 标题增强（更强）
                title = str(rerank_candidates[i]["metadata"].get("title", ""))
                q_words = [w for w in jieba.lcut(query_clean) if len(w) > 1]

                match_count = sum(1 for w in q_words if w in title)
                f_score += 0.1 * match_count

                scored_items.append({
                    "score": f_score,
                    "doc_obj": rerank_candidates[i],
                    "full_id": str(rerank_candidates[i]["metadata"]["docid"]).strip()
                })

            # 未 rerank 的补零分
            for c in candidates[rerank_k:]:
                scored_items.append({
                    "score": 0,
                    "doc_obj": c,
                    "full_id": str(c["metadata"]["docid"]).strip()
                })

            scored_items.sort(key=lambda x: x["score"], reverse=True)

        else:
            scored_items = [
                {"score": 0, "doc_obj": c, "full_id": str(c["metadata"]["docid"])}
                for c in candidates
            ]

        # ===============================
        # 阶段 C: 去重优化（延迟策略）
        # ===============================
        final_results = []
        seen_main = {}

        def normalize(s):
            return str(s).split('#')[0].strip().lower()

        target_rank = -1
        raw_pos = -1

        # raw rank
        for idx, item in enumerate(scored_items):
            if t_id_str and t_id_str in item["full_id"].lower():
                raw_pos = idx + 1
                break

        for item in scored_items:
            f_id = item["full_id"].lower()
            m_id = normalize(f_id)

            is_target = (t_id_str != "" and t_id_str in f_id)

            count = seen_main.get(m_id, 0)

            # 👉 前10不去重
            if len(final_results) < 10:
                keep = True
            else:
                keep = (count < 3)

            if is_target or keep:
                final_results.append(item["doc_obj"])
                seen_main[m_id] = count + 1

                if is_target and target_rank == -1:
                    target_rank = len(final_results)

            if len(final_results) >= 50:
                break

        # ===============================
        # 阶段 D: 日志
        # ===============================
        # if t_id_str and (target_rank > 5 or target_rank == -1):
        if t_id_str:
            with open(self.failure_log_path, "a", encoding="utf-8") as f:
                analysis = {
                    "query": query,
                    "target_id": target_id,
                    "raw_rank": raw_pos,
                    "final_rank": target_rank,
                    "top_1_id": scored_items[0]["full_id"] if scored_items else None,
                    "score_gap": (
                        scored_items[0]["score"] - scored_items[raw_pos - 1]["score"]
                        if raw_pos > 0 else None
                    )
                }
                f.write(json.dumps(analysis, ensure_ascii=False) + "\n")

        return final_results[:top_n]