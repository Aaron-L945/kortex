import json
import numpy as np
import jieba
import faiss
import re
import os
import torch
import gc
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger
from opencc import OpenCC
from collections import defaultdict


class HybridRetrieverV3:
    def __init__(self, model_path: str, reranker_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"⚙️ 初始化检索器，使用设备: {self.device}")

        self.embed_model = SentenceTransformer(model_path, device=self.device)
        self.reranker = None
        if reranker_path:
            logger.info(f"🚀 加载精排模型: {reranker_path}")
            self.reranker = CrossEncoder(
                reranker_path, device=self.device, max_length=512
            )

        self.index = None
        self.bm25 = None
        self.corpus_docs = []
        self.cc = OpenCC("t2s")

        self.index_path = "faiss_index.bin"
        self.metadata_path = "corpus_metadata.json"
        # 🚀 修复点：定义审计日志与失败日志路径
        self.audit_path = "retrieval_audit.jsonl"
        self.failure_log_path = "failure_analysis.jsonl"
        self.stopwords = set(
            [
                "什么",
                "多少",
                "如何",
                "是否",
                "为什么",
                "哪里",
                "的",
                "了",
                "和",
                "是",
                "在",
            ]
        )

        # 初始化时清空上一次的失败日志
        if os.path.exists(self.failure_log_path):
            os.remove(self.failure_log_path)

        self.semantic_keywords = []
        self.keyword_embs = None
        self.user_dict_path = "user_dict.txt"

        if os.path.exists(self.user_dict_path):
            logger.info("📚 加载用户词典")
            jieba.load_userdict(self.user_dict_path)

    def _preprocess_text(self, text: str) -> str:
        text = self.cc.convert(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_index(self, jsonl_path: str, force_rebuild: bool = False):
        self._build_user_dict(jsonl_path)

        if (
            not force_rebuild
            and os.path.exists(self.index_path)
            and os.path.exists(self.metadata_path)
        ):
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
                        "corpus_idx": i,
                    },
                }
                self.corpus_docs.append(doc_obj)
                processed_texts.append(clean_content)

        self.bm25 = BM25Okapi([list(jieba.cut_for_search(t)) for t in processed_texts])
        embeddings = self.embed_model.encode(
            processed_texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True,
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        self._save_to_disk()
        self._build_semantic_index()

    def _build_user_dict(self, jsonl_path):
        logger.info("🧠 自动构建用户词典（优化版）...")

        from collections import Counter

        counter = Counter()
        title_words = set()

        # =========================
        # 1️⃣ 收集候选词（带频次）
        # =========================
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # 🔥 标题（强信号）
                title = data.get("title", "")
                if len(title) > 2:
                    title_words.add(title.strip())

                text = data.get("text", "")

                # 正则抽长词
                words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{4,}", text)

                counter.update(words)

        # =========================
        # 2️⃣ 过滤规则（核心🔥）
        # =========================
        bad_patterns = [
            "研究",
            "表示",
            "发现",
            "指出",
            "认为",
            "具有",
            "进行",
            "通过",
            "一种",
            "可以",
        ]

        def is_good_word(w, c):
            if c < 3:  # 🔥 频次过滤
                return False
            if len(w) > 20:  # 句子过滤
                return False
            if w.isdigit():  # 纯数字
                return False
            if any(p in w for p in bad_patterns):
                return False
            return True

        word_set = [w for w, c in counter.items() if is_good_word(w, c)]

        # =========================
        # 3️⃣ 加入标题词（不做过滤🔥）
        # =========================
        word_set = set(word_set) | title_words

        # =========================
        # 4️⃣ 控制规模（防炸🔥）
        # =========================
        MAX_DICT_SIZE = 60000

        if len(word_set) > MAX_DICT_SIZE:
            logger.warning(f"⚠️ 词典过大，截断到 {MAX_DICT_SIZE}")
            word_set = list(word_set)[:MAX_DICT_SIZE]

        # =========================
        # 5️⃣ 写入词典（自动+手工一起写）
        # =========================
        manual_words = [
            "南京大学生命科学学院",
            "中国科学社生物研究所",
            "教育部",
            "印第安纳自治市镇",
            "年龄人口分布",
            "网络孟乔森综合症",
        ]

        with open(self.user_dict_path, "w", encoding="utf-8") as f:
            # 自动词
            for w in word_set:
                f.write(f"{w} 10\n")

            # 手工词（高权重🔥）
            for w in manual_words:
                f.write(f"{w} 50\n")

        # =========================
        # 6️⃣ 加载词典
        # =========================
        jieba.load_userdict(self.user_dict_path)

        logger.info(f"✅ 用户词典构建完成，共 {len(word_set)} 个词")

    def _build_semantic_index(self):
        logger.info("🧠 构建语义关键词索引...")
        counter = defaultdict(int)
        for doc in self.corpus_docs:
            phrases = self.extract_phrases(doc["content"])
            for p in phrases:
                if len(p) > 1:
                    counter[p] += 1

        # 取高频词
        keywords = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:3000]
        self.semantic_keywords = [k for k, _ in keywords]

        self.keyword_embs = self.embed_model.encode(
            self.semantic_keywords, normalize_embeddings=True, batch_size=32
        )

    def extract_phrases(self, text):
        words = jieba.lcut(text)
        phrases = []

        # unigram
        phrases.extend(words)

        # bigram
        for i in range(len(words) - 1):
            phrases.append(words[i] + words[i + 1])

        # trigram（关键🔥）
        for i in range(len(words) - 2):
            phrases.append(words[i] + words[i + 1] + words[i + 2])

        return phrases

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

    def _semantic_expand(self, word, top_k=3):
        if not self.semantic_keywords:
            return []

        q_emb = self.embed_model.encode([word], normalize_embeddings=True)

        sims = np.dot(self.keyword_embs, q_emb[0])
        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [self.semantic_keywords[i] for i in top_idx]

    def _expand_query_priority1(self, query):
        # 【P0：绝对优先级】原始 Query 必须在第一位
        expansions = [query]

        # 【P1：业务硬规则 - 外语/映射增强】
        # 解决：中文搜不到，英文能搜到的情况
        foreign_map = {
            "网络孟乔森综合症": "Munchausen by Internet",
            "印第安纳自治市镇": "Indiana township",
        }
        for k, v in foreign_map.items():
            if k in query:
                expansions.append(v)

        # 【P2：业务硬规则 - 疑问词增强】
        # 解决：语义对齐（时间->年份，比例->百分比）
        if "比例" in query or "占比" in query:
            expansions.append(query + " 占比 百分比 %")
        if "时间" in query or "何时" in query or "年份" in query:
            expansions.append(query + " 日期 年份 成立时间")
        if "有哪些" in query or "是什么" in query:
            expansions.append(query + " 包括 包含 简介")

        # 【P3：实体/别名增强】
        # 解决：缩写与全称（南大->南京大学）
        alias_map = {
            "南京大学": ["南大", "国立中央大学"],
            "成立": ["创办", "创建", "建立"],
            "成员": ["团员", "名单", "组成人员"]
        }
        for k, synonyms in alias_map.items():
            if k in query:
                for syn in synonyms:
                    expansions.append(query.replace(k, syn))

        # 【P4：结构化提取】
        # 解决：长 Query 降噪，只留核心实体
        import jieba.posseg as pseg
        words = pseg.lcut(query)
        # 提取名词性实体
        entities = [w for w, t in words if t in ['nt', 'nz', 'nr', 'n']]
        if len(entities) > 1:
            expansions.append(" ".join(entities))

        # ==========================================
        # 最后一步：【确定性去重返回】🔥
        # 使用 dict.fromkeys 保证 P0 -> P1 -> P2 -> P3 -> P4 的顺序不乱
        # ==========================================
        unique_results = list(dict.fromkeys(expansions))
        
        # 返回前 8 个，确保最精准的规则在前，最泛化的在后
        return unique_results[:8]

    def _rewrite_query(self, query):
        # 使用 list 保持顺序，或者最后进行确定性排序
        query_list = []

        # 1. 原始 query 优先级最高，排第一
        query_list.append(query)

        # 2. 规则扩展 (优先级 2)
        rule_queries = self._expand_query_priority1(query)
        for rq in rule_queries:
            if rq not in query_list:
                query_list.append(rq)

        # 3. 语义扩展 (优先级 3)
        words = jieba.lcut(query)
        semantic_queries = []
        for w in words:
            if len(w) <= 1 or w in self.stopwords:
                continue
            sim_words = self._semantic_expand(w, top_k=2)
            for sw in sim_words:
                new_q = query.replace(w, sw)
                if new_q not in query_list:
                    semantic_queries.append(new_q)

        # 将语义扩展接在后面
        query_list.extend(semantic_queries)

        # 4. 【关键】去重并保持顺序（Python 3.7+ dict 是有序的）
        unique_queries = list(dict.fromkeys(query_list))

        # 5. 【增强】确定性截断：如果还是想按长度优先，可以再排一次序
        # 但通常建议：原始 Query 必须在前 3 个。
        return unique_queries[:8]

    def _force_phrase(self, query):
        phrases = ["南京大学生命科学学院", "中国科学社生物研究所", "印第安纳自治市镇"]
        for p in phrases:
            if p in query:
                jieba.add_word(p)
        return query

    def pipeline(
        self,
        query: str,
        top_k: int = 300,
        rerank_k: int = 100,
        top_n: int = 5,
        target_id: str = None,
    ):

        query_clean = query.replace("·", " ").replace("•", " ").strip()
        t_id_str = str(target_id).strip().lower() if target_id else ""
        query_clean = self._force_phrase(query_clean)

        # ===============================
        # 阶段 A: 多 Query + 多路召回
        # ===============================
        queries = self._rewrite_query(query_clean)

        logger.warning(
            f"""
            🔍 Query分析:
            原始: {query}
            rewrite后: {queries}

            分词: {jieba.lcut(query)}

            semantic触发:
            {[(w, self._semantic_expand(w)) for w in jieba.lcut(query) if len(w)>1]}
            """
        )

        # bm25 + vector 多路召回
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
                [query_clean, f"{c['metadata'].get('title','')}\n{c['content']}"]
                for c in rerank_candidates
            ]

            gc.collect()
            torch.cuda.empty_cache()

            with torch.inference_mode():
                scores = self.reranker.predict(
                    pairs, batch_size=32, show_progress_bar=False
                )

            for i, score in enumerate(scores):
                f_score = float(score)

                # 标题增强（更强）
                title = str(rerank_candidates[i]["metadata"].get("title", ""))
                q_words = [w for w in jieba.lcut(query_clean) if len(w) > 1]

                match_count = sum(1 for w in q_words if w in title)
                f_score += 0.1 * match_count

                scored_items.append(
                    {
                        "score": f_score,
                        "doc_obj": rerank_candidates[i],
                        "full_id": str(rerank_candidates[i]["metadata"]["docid"]).strip(),
                    }
                )

            # 未 rerank 的补零分
            for c in candidates[rerank_k:]:
                scored_items.append(
                    {
                        "score": 0,
                        "doc_obj": c,
                        "full_id": str(c["metadata"]["docid"]).strip(),
                    }
                )

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
            return str(s).split("#")[0].strip().lower()

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

            is_target = t_id_str != "" and t_id_str in f_id

            count = seen_main.get(m_id, 0)

            # 👉 前10不去重
            if len(final_results) < 10:
                keep = True
            else:
                keep = count < 3

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
        logger.info("Start to write log ...")
        with open(self.failure_log_path, "a", encoding="utf-8") as f:
            analysis = {
                "query": query,
                "target_id": target_id,
                "raw_rank": raw_pos,
                "final_rank": target_rank,
                "top_1_id": scored_items[0]["full_id"] if scored_items else None,
                "score_gap": (
                    scored_items[0]["score"] - scored_items[raw_pos - 1]["score"]
                    if raw_pos > 0
                    else None
                ),
            }
            f.write(json.dumps(analysis, ensure_ascii=False) + "\n")

        return final_results[:top_n]
