import os
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 新增：导入切片工具
from core.loader import smart_load_pdf
from embedder import create_vector_store
from config import Config
from loguru import logger


def main():
    # 确保基础目录存在
    if not os.path.exists("data"):
        os.makedirs("data")

    print("\n🚀 " + "=" * 20 + " 知识库全量构建开始 " + "=" * 20 + "\n")

    # 1. 初始化切片器：这是解决“步骤丢失”的核心
    # 从 Config 或 .env 读取参数，保证灵活性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,  # 建议 800-1000
        chunk_overlap=Config.CHUNK_OVERLAP,  # 建议 200，让跨页步骤在两个块中都有保留
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 优先级切分符
        add_start_index=True,  # 在元数据中保留起始位置
    )

    all_final_chunks = []  # 用于存储最终切分后的所有片段
    all_files = [f for f in os.listdir("data") if f.endswith(".pdf")]

    if not all_files:
        print("❌ 错误: data/ 目录下没有找到任何 PDF 文件。")
        return

    for file in all_files:
        full_path = os.path.join("data", file)
        print(f"📄 正在加载文件: {file} ...")

        # A. 调用 smart_load_pdf 获取“页”级别的 Document 列表
        # 注意：这里返回的是 raw_pages，不是真正的 chunks
        raw_pages = smart_load_pdf(file_path=full_path)

        if raw_pages:
            # B. 执行语义切片：将“页”切成符合长度要求的“块”
            # split_documents 会自动继承每页的 metadata (source, page 等)
            file_chunks = text_splitter.split_documents(raw_pages)

            all_final_chunks.extend(file_chunks)
            print(
                f"   ✅ 提取完成: {len(raw_pages)} 页 -> 转化成 {len(file_chunks)} 个片段"
            )
        else:
            logger.warning(f"⚠️ 文件 {file} 未能提取到任何文字，请检查是否为加密文档。")

    # 2. 统一构建向量库
    if all_final_chunks:
        print(f"\n📊 [统计] 最终待入库片段总数: {len(all_final_chunks)}")
        print("🧠 正在生成向量并存储到本地 (FAISS) ...")

        try:
            # 这里的 create_vector_store 应该接收最终切好块的 list
            create_vector_store(all_final_chunks)
            print("\n✨ " + "=" * 20 + " 向量库构建成功 " + "=" * 20 + "\n")
        except Exception as e:
            logger.error(f"❌ 向量库构建失败: {e}")
    else:
        print("\n终止: 没有有效片段，无法构建索引。")


if __name__ == "__main__":
    main()
