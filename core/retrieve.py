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
        self.audit_path = "retrieval_audit.jsonl"

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
                        "corpus_idx": i  # 保存原始索引用于窗口扩展
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
        """获取当前片段前后的上下文进行拼接"""
        start = max(0, corpus_idx - window_size)
        end = min(len(self.corpus_docs), corpus_idx + window_size + 1)
        # 拼接上下文，增强精排阶段的语义识别
        context_text = " ".join([self.corpus_docs[i]["content"] for i in range(start, end)])
        return context_text

    def _rrf_score(self, bm25_ranks, vector_ranks, k=60):
        rrf_map = {}
        for rank, idx in enumerate(bm25_ranks):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.5 * (1.0 / (k + rank + 1))
        for rank, idx in enumerate(vector_ranks):
            rrf_map[idx] = rrf_map.get(idx, 0) + 1.0 / (k + rank + 1)
        return [item[0] for item in sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)]
    

    def pipeline(self, query: str, top_k: int = 1000, top_n: int = 5, score_threshold: float = 0.3):
        query_clean = self._preprocess_text(query)
        
        # 1. 扩大粗排池
        query_tokens = list(jieba.cut_for_search(query_clean))
        bm25_idx = np.argsort(self.bm25.get_scores(query_tokens))[::-1][:top_k].tolist()
        
        inst_query = f"为这个句子生成表示以用于检索相关文章：{query_clean}"
        q_emb = self.embed_model.encode([inst_query], normalize_embeddings=True)
        _, vector_idx = self.index.search(q_emb.astype("float32"), top_k)
        
        combined_indices = self._rrf_score(bm25_idx, vector_idx[0].tolist(), k=60)[:top_k]
        candidates = [self.corpus_docs[i] for i in combined_indices]

        # 2. 精排 + 窗口扩展
        scored_list = []
        if self.reranker and candidates:
            pairs = [[query_clean, self._get_context_window(c["metadata"].get("corpus_idx", 0), 1)] for c in candidates]
            with torch.inference_mode():
                scores = self.reranker.predict(pairs, batch_size=64)
            for i, score in enumerate(scores):
                scored_list.append({"score": float(score), "doc_obj": candidates[i]})
            scored_list.sort(key=lambda x: x["score"], reverse=True)

        # 3. 【核心优化】DocID 聚合去重 (提升 Precision 的关键)
        # 如果前几个结果都属于同一个 Doc，只保留分最高的一个，空位留给后面的 Doc
        unique_final = []
        seen_docids = set()
        
        for item in scored_list:
            # 假设你的 docid 格式是 "12345#1"，我们需要提取前面的 "12345"
            raw_id = str(item["doc_obj"]["metadata"]["docid"])
            main_id = raw_id.split('#')[0] 
            
            if main_id not in seen_docids:
                unique_final.append(item)
                seen_docids.add(main_id)
            if len(unique_final) >= top_n:
                break

        # 4. 动态过滤策略
        if not unique_final:
            return [candidates[0]] if candidates else []

        top1_score = unique_final[0]["score"]
        # 如果第一名遥遥领先，执行“精准打击”，只返回一个
        if len(unique_final) > 1 and top1_score > 0.9 and (top1_score - unique_final[1]["score"] > 0.5):
            return [unique_final[0]["doc_obj"]]

        # 否则，返回通过动态阈值过滤后的结果
        dynamic_thresh = min(score_threshold, top1_score * 0.4)
        return [x["doc_obj"] for x in unique_final if x["score"] >= dynamic_thresh]

    # def pipeline(self, query: str, top_k: int = 200, top_n: int = 5, score_threshold: float = 0.35):
        query_clean = self._preprocess_text(query)
        
        # 1. 粗排
        query_tokens = list(jieba.cut_for_search(query_clean))
        bm25_idx = np.argsort(self.bm25.get_scores(query_tokens))[::-1][:top_k].tolist()
        
        inst_query = f"为这个句子生成表示以用于检索相关文章：{query_clean}"
        q_emb = self.embed_model.encode([inst_query], normalize_embeddings=True)
        _, vector_idx = self.index.search(q_emb.astype("float32"), top_k)
        
        combined_indices = self._rrf_score(bm25_idx, vector_idx[0].tolist(), k=60)[:top_k]
        candidates = [self.corpus_docs[i] for i in combined_indices]

        # 2. 精排 (带窗口扩展)
        final_results = []
        if self.reranker and candidates:
            # 窗口扩展：拼接上下文后再送入 Reranker
            pairs = []
            for cand in candidates:
                context_text = self._get_context_window(cand["metadata"]["corpus_idx"], window_size=1)
                pairs.append([query_clean, context_text])

            with torch.inference_mode():
                scores = self.reranker.predict(pairs, batch_size=32)
            
            scored_list = []
            for i, score in enumerate(scores):
                scored_list.append({"score": float(score), "doc_obj": candidates[i]})

            scored_list.sort(key=lambda x: x["score"], reverse=True)

            print(f"\n--- [DEBUG] Query: {query[:30]} ---")
            seen_ids = {}
            for rank, item in enumerate(scored_list[:10]): # 观察前10个
                d_id = item["doc_obj"]["metadata"]["docid"]
                score = item["score"]
                seen_ids[d_id] = seen_ids.get(d_id, 0) + 1
                
                # 打印前10名的得分分布
                print(f"Rank {rank+1} | DocID: {d_id} | Score: {score:.4f}")

            # 计算重复率事实
            duplicate_count = sum([v for v in seen_ids.values() if v > 1])
            print(f"统计：Top 10 中重复出现的 Doc 数量: {duplicate_count}")
            print(f"统计：去重后的唯一 Doc 数量: {len(seen_ids)}")

            # 3. 动态截断策略 (F1 优化核心)
            # 优先选择超过阈值的高分结果
            high_score_results = [x["doc_obj"] for x in scored_list[:top_n] if x["score"] >= score_threshold]
            
            # 特殊逻辑：如果第一名具有压倒性优势，只返回第一名（极大提升 Precision）
            if len(scored_list) > 1:
                margin = scored_list[0]["score"] - scored_list[1]["score"]
                if scored_list[0]["score"] > 0.8 and margin > 0.45:
                    final_results = [scored_list[0]["doc_obj"]]
                else:
                    final_results = high_score_results
            else:
                final_results = high_score_results

            # 4. 补齐逻辑：如果结果不满 5 个，用排名前列的补齐 (确保多答案 Recall 不掉)
            if 0 < len(final_results) < 5:
                existing_ids = {d["metadata"]["docid"] for d in final_results}
                for item in scored_list:
                    if item["doc_obj"]["metadata"]["docid"] not in existing_ids:
                        final_results.append(item["doc_obj"])
                        existing_ids.add(item["doc_obj"]["metadata"]["docid"])
                    if len(final_results) == 5: break

        # 保底：如果全部被过滤，返回最相关的 1 个
        if not final_results and candidates:
            final_results = [candidates[0]]

        print(f"{top_n=}")
        return final_results[:top_n]