import json
from collections import Counter
import jieba.posseg as pseg

STOPWORDS = set(["进行", "相关", "表示", "可以", "其中", "一种", "这个", "以及"])
VALID_FLAGS = ("n", "nr", "ns", "nz", "vn")  # 名词、人名、地名、专有名词、动名词

def extract_keywords_from_mixed_json(file_path, text_field="text_snippet", top_k=200):
    """
    万能 JSON 解析 + jieba 分词 + 关键词统计
    兼容：
    - JSONL（每行 JSON）
    - 连续 JSON 对象
    - 换行容错
    """
    counter = Counter()
    buffer = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buffer += line
            try:
                doc = json.loads(buffer)
                buffer = ""  # 解析成功后清空 buffer
            except json.JSONDecodeError:
                # 如果解析失败，可能是当前行不完整 JSON，等下一行继续拼接
                continue

            text = doc.get(text_field, "")
            for word, flag in pseg.cut(text):
                if len(word) > 1 and word not in STOPWORDS and flag.startswith(VALID_FLAGS):
                    counter[word] += 1

    return counter.most_common(top_k)


file_path = "/root/tests/aaron/kortex/corpus.jsonl"
top_keywords = extract_keywords_from_mixed_json(file_path, text_field="text_snippet", top_k=100)
print(top_keywords)

