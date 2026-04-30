import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from loguru import logger
from tqdm import tqdm

def deduplicate(input_path, output_path, model_path, threshold=0.95):
    # 1. 加载模型
    model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 读取原始数据
    raw_docs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_docs.append(json.loads(line))
    
    logger.info(f"原始数据量: {len(raw_docs)}")
    
    # 3. 计算所有文档的向量
    texts = [doc['text'] for doc in raw_docs]
    logger.info("正在计算向量以进行比对...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
    
    # 4. 贪心去重
    keep_indices = []
    dropped_count = 0
    # 已保存文档的向量库
    processed_embeddings = None 
    
    logger.info("正在进行语义查重...")
    for i in tqdm(range(len(embeddings))):
        current_emb = embeddings[i].unsqueeze(0)
        
        if processed_embeddings is None:
            keep_indices.append(i)
            processed_embeddings = current_emb
        else:
            # 计算当前文档与已保存文档的最大相似度
            cos_sim = util.cos_sim(current_emb, processed_embeddings)[0]
            max_sim = torch.max(cos_sim).item()
            
            if max_sim < threshold:
                keep_indices.append(i)
                # 将新向量加入库中用于后续比对
                processed_embeddings = torch.cat([processed_embeddings, current_emb], dim=0)
            else:
                dropped_count += 1
                
    # 5. 保存去重后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in keep_indices:
            f.write(json.dumps(raw_docs[idx], ensure_ascii=False) + '\n')
            
    logger.success(f"去重完成！")
    logger.info(f"保留: {len(keep_indices)} 条")
    logger.info(f"剔除重复: {dropped_count} 条")
    logger.info(f"去重后文件已保存至: {output_path}")

if __name__ == "__main__":
    from config import Config
    deduplicate(
        input_path="corpus.jsonl",
        output_path="corpus_dedup.jsonl",
        model_path=Config.LOCAL_MODEL_PATH,
        threshold=0.98  # 阈值建议 0.95~0.98，针对你这种几乎原样重复的，0.98 比较保险
    )
