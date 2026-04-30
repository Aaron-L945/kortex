import os
from embedder import load_vector_store
from config import Config

def inspect_knowledge_base(target_file_keyword=None, search_test_query=None):
    """
    检查向量库状态并进行检索测试
    :param target_file_keyword: 想要查找的具体文件名关键词
    :param search_test_query: 想要测试的语义搜索词
    """
    if not os.path.exists(Config.VECTOR_DB_PATH):
        print(f"❌ 错误：在 {Config.VECTOR_DB_PATH} 未找到向量库文件。")
        return

    db = load_vector_store()
    
    # --- 1. 基础统计 ---
    total_chunks = db.index.ntotal
    print("="*40)
    print(f"📊 知识库概览")
    print(f"总片段数 (Chunks): {total_chunks}")
    print("="*40)

    # --- 2. 遍历所有来源文件 (从 docstore 提取) ---
    all_sources = set()
    # FAISS 的 docstore 存储了所有的原始文档对象
    if hasattr(db, 'docstore'):
        for doc_id in db.docstore._dict:
            doc = db.docstore.search(doc_id)
            source = doc.metadata.get('source', 'Unknown')
            all_sources.add(source)
    
    print(f"已入库文件清单 ({len(all_sources)} 个):")
    found_target = False
    for s in sorted(all_sources):
        status = "✅"
        if target_file_keyword and target_file_keyword.lower() in s.lower():
            status = "⭐ [匹配目标]"
            found_target = True
        print(f"  {status} {s}")

    if target_file_keyword and not found_target:
        print(f"\n❌ 警告：未在库中找到包含 '{target_file_keyword}' 的文件！")

    # --- 3. 语义检索回测 (可选) ---
    if search_test_query:
        print("\n" + "="*40)
        print(f"🔍 检索测试: '{search_test_query}'")
        print("="*40)
        # 模拟一次 Top-3 检索
        docs = db.similarity_search(search_test_query, k=3)
        for i, doc in enumerate(docs):
            print(f"\n结果 {i+1} | 来源: {doc.metadata.get('source')} | 页码: {doc.metadata.get('page')}")
            # 打印前 100 个字符预览
            content_preview = doc.page_content.replace('\n', ' ')[:100]
            print(f"内容预览: {content_preview}...")

if __name__ == "__main__":
    # 示例 1: 只看整体情况
    # inspect_knowledge_base()

    # 示例 2: 检查特定文件是否在库里，并顺便测试一下检索效果
    # 请根据你的实际文件名修改 target_file_keyword
    inspect_knowledge_base(
        target_file_keyword="微软的软件测试之道", 
        search_test_query="AI是什么？"
    )
