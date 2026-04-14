import os
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.loader import smart_load_pdf
from embedder import create_vector_store
from config import Config
from loguru import logger

def generate_file_id(filename: str) -> str:
    """生成文件的唯一标识，防止同名文件冲突"""
    return hashlib.md5(filename.encode()).hexdigest()[:8]

def main():
    # 确保基础目录存在
    if not os.path.exists("data"):
        os.makedirs("data")

    print("\n🚀 " + "=" * 20 + " 知识库全量构建 (分层+窗口增强版) " + "=" * 20 + "\n")

    # 1. 初始化切片器
    # 建议：分层架构下，chunk_size 可以设得稍小（如 500-600），提高检索精度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,  
        chunk_overlap=Config.CHUNK_OVERLAP, 
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        add_start_index=True,
    )

    all_final_chunks = []
    all_files = [f for f in os.listdir("data") if f.endswith(".pdf")]

    if not all_files:
        print("❌ 错误: data/ 目录下没有找到任何 PDF 文件。")
        return

    for file_name in all_files:
        full_path = os.path.join("data", file_name)
        file_id = generate_file_id(file_name)
        print(f"📄 正在解析: {file_name} (ID: {file_id}) ...")

        # A. 加载原始页面
        raw_pages = smart_load_pdf(file_path=full_path)

        if raw_pages:
            # B. 执行切片
            file_chunks = text_splitter.split_documents(raw_pages)
            
            # --- 【关键：注入层级元数据】 ---
            for i, chunk in enumerate(file_chunks):
                # 记录该片段在当前文件中的绝对顺序，用于后续“窗口补全”
                chunk.metadata["chunk_id"] = i  
                chunk.metadata["file_id"] = file_id
                # 显式保留原始文件名，方便追溯
                chunk.metadata["source"] = file_name 
            
            all_final_chunks.extend(file_chunks)
            print(f"   ✅ 提取完成: {len(raw_pages)} 页 -> {len(file_chunks)} 个语义片段")
        else:
            logger.warning(f"⚠️ 文件 {file_name} 未能提取到内容。")

    # 2. 统一构建向量库
    if all_final_chunks:
        print(f"\n📊 [统计] 待入库总片段数: {len(all_final_chunks)}")
        print("🧠 正在生成 Embedding 并存入 FAISS ...")

        try:
            # 存储切好块且带 chunk_id 的数据
            create_vector_store(all_final_chunks)
            print("\n✨ " + "=" * 20 + " 向量库构建成功 (支持邻居检索) " + "=" * 20 + "\n")
        except Exception as e:
            logger.error(f"❌ 向量库构建失败: {e}")
    else:
        print("\n终止: 无有效数据。")

if __name__ == "__main__":
    main()