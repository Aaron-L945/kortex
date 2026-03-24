import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import Config

class Reranker:
    def __init__(self):
        # 使用你提供的路径
        self.model_path = "/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3"
        print(f"正在加载重排模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def rerank(self, query, passages, top_n=3):
        if not passages:
            return []
        
        # 构造输入对
        pairs = [[query, p.page_content] for p in passages]
        
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 计算相关性分值
            scores = self.model(**inputs).logits.view(-1,).float()
            
        # 绑定分值并排序
        for i, score in enumerate(scores):
            passages[i].metadata['rerank_score'] = score.item()
            
        sorted_passages = sorted(passages, key=lambda x: x.metadata['rerank_score'], reverse=True)
        return sorted_passages[:top_n]

# 实例化单例
reranker_tool = Reranker()