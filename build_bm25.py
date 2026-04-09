import json
import pickle
import jieba
from rank_bm25 import BM25Okapi

def build_bm25(metadata_path, bm25_path="config/bm25.pkl"):
    corpus_docs  = None
    with open(metadata_path, "r", encoding="utf-8") as f:
        corpus_docs = json.load(f)

    texts = [d["content"] for d in corpus_docs]

    # 分词（只做一次！）
    tokenized = [list(jieba.cut_for_search(t)) for t in texts]

    bm25 = BM25Okapi(tokenized)

    # ✅ 持久化 BM25
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # （可选）保存 token，方便 debug / 重建
    with open("tokens.pkl", "wb") as f:
        pickle.dump(tokenized, f)

    print("✅ BM25 构建完成并已持久化")


build_bm25(metadata_path="corpus_metadata.json")