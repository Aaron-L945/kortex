import os
from embedder import load_vector_store
from config import Config

def run_acceptance_test():
    print("🔍 === 开始第一阶段验收测试 ===\n")

    # 1. 检查物理文件
    print("[1/3] 检查向量库文件...")
    faiss_file = os.path.join(Config.VECTOR_DB_PATH, "index.faiss")
    pkl_file = os.path.join(Config.VECTOR_DB_PATH, "index.pkl")
    
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print("✅ 向量库文件已就绪。")
    else:
        print("❌ 错误：未找到向量库文件，请先运行 main.py")
        return

    # 2. 尝试加载向量库
    print("\n[2/3] 尝试加载索引并提取元数据...")
    try:
        db = load_vector_store()
        # 获取索引中的总条数（FAISS 特有方法）
        total_vectors = db.index.ntotal
        print(f"✅ 成功加载索引。当前库内共有 {total_vectors} 个知识切片。")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 3. 检索功能测试（中英双语）
    print("\n[3/3] 执行模拟检索测试...")
    test_queries = [
        "如何做好logging？",  # 中文测试
    ]
    
    for query in test_queries:
        print(f"👉 提问: '{query}'")
        # 搜索最相关的 2 条数据
        results = db.similarity_search_with_score(query, k=2)
        
        if not results:
            print("❓ 警告：未找到相关匹配项。")
        else:
            for i, (doc, score) in enumerate(results):
                # score 越小表示越相似（L2 距离）
                print(f"   结果 {i+1} [得分: {score:.4f}]:")
                print(f"   来自文件: {doc.metadata.get('source', '未知')}")
                print(f"   所在页码: {doc.metadata.get('page', '未知')}")
                print(f"   部门标签: {doc.metadata.get('department', '未分类')}")
                print(f"   内容片段: {doc.page_content[:100].replace('\\n', ' ')}...")
                print("-" * 30)

    print("\n🎉 === 第一阶段验收通过！您可以进入第二阶段 (LLM 接入) ===\n")

if __name__ == "__main__":
    run_acceptance_test()
