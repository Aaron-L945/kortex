import os
import re
from loguru import logger
# 假设你的项目结构如下，请确保导入路径正确
from embedder import load_vector_store
from core.retriever_without_windows import HybridRetriever
from reranker import reranker_tool

def get_expanded_context_final(selected_docs, all_candidates, window_size=3):
    """
    最优扩展逻辑：
    1. 增加 window_size 到 3（前后各捞 3 个，覆盖约 1.5 页内容）
    2. 严格按物理顺序 (file_id + chunk_id) 重组
    """
    expanded_map = {}
    # 邻居池
    pool = {f"{d.metadata.get('file_id')}_{d.metadata.get('chunk_id')}": d for d in all_candidates}
    
    for doc in selected_docs:
        f_id = doc.metadata.get('file_id')
        c_id = doc.metadata.get('chunk_id')
        
        if c_id is None: continue
            
        # 向上向下各找 3 个邻居
        for i in range(c_id - window_size, c_id + window_size + 1):
            key = f"{f_id}_{i}"
            if key in pool:
                is_original = (i == c_id)
                # 如果已存在，保留其“原创”状态
                if key in expanded_map:
                    if is_original: expanded_map[key] = (pool[key], True)
                else:
                    expanded_map[key] = (pool[key], is_original)
    
    final_list = list(expanded_map.values())
    final_list.sort(key=lambda x: (x[0].metadata.get('file_id'), x[0].metadata.get('chunk_id')))
    return final_list

def run_acceptance_test():
    # --- 硬编码配置区 ---
    TOP_K_RETRIEVAL = 100    # 粗排必须给够，防止 12/13 掉出前 50
    TOP_N_RERANK = 15        # 精排多留一点锚点
    WINDOW_SIZE = 3          # 强力窗口，跨页必杀
    TARGET_FILE_KEYWORD = "Nuage-VSP-20.10.R14.1" # 重点关注的目标文件
    # ------------------

    print(f"\n🧪 [实验模式] 检索参数: TopK={TOP_K_RETRIEVAL}, TopN={TOP_N_RERANK}, Window={WINDOW_SIZE}\n")

    try:
        db = load_vector_store()
        retriever = HybridRetriever(db)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    query = "如何升级VSC？"
    print(f"🔎 Query: {query}")

    # 1. 粗排
    hybrid_results = retriever.search(query, top_k=TOP_K_RETRIEVAL)
    all_candidates = [res[0] for res in hybrid_results]
    
    # 2. 精排
    docs_after_rerank = reranker_tool.rerank(query, all_candidates, top_n=TOP_N_RERANK)

    # 3. 窗口扩展
    final_context = get_expanded_context_final(docs_after_rerank, all_candidates, window_size=WINDOW_SIZE)

    print("\n" + "="*90)
    print(f"{'来源':<40} | {'CID':<6} | {'类型':<10} | {'内容摘要'}")
    print("-" * 90)

    found_bgp = False
    found_repeat = False

    for doc, is_original in final_context:
        meta = doc.metadata
        source = meta.get('source', 'Unknown')[:38]
        cid = meta.get('chunk_id', 'N/A')
        tag = "🎯 HIT" if is_original else "🪟 WIN"
        
        content = doc.page_content.replace('\n', ' ').strip()
        
        # 实时检测关键步骤
        highlight = ""
        if "BGP" in content or "neighbor" in content:
            highlight = " 🔥 [Step 11 候选]"
            found_bgp = True
        if "Repeat" in content or "Step 1" in content and cid > 60:
            highlight = " ⚡ [Step 12 候选]"
            found_repeat = True

        # 重点只打印目标文件的内容，过滤噪音
        if TARGET_FILE_KEYWORD in source:
            print(f"{source:<40} | {cid:<6} | {tag:<10} | {content[:60]}...{highlight}")

    print("="*90)

    if found_bgp and found_repeat:
        print("\n🎉 SUCCESS: 11-13 步已进入上下文！这套参数可用。")
    else:
        print("\n❌ FAILED: 仍未发现 11-13 步。")
        print("💡 建议：如果 CID 63-70 都在，但没内容，说明是 split_documents 时把这部分漏了。")

if __name__ == "__main__":
    run_acceptance_test()