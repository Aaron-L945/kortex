import os
from loader import smart_load_pdf
from embedder import create_vector_store
from config import Config


def main():
    if not os.path.exists("data"):
        os.makedirs("data")

    print("--- 知识库构建开始 ---")

    all_chunks = []  # 1. 初始化一个总列表
    all_files = [f for f in os.listdir("data") if f.endswith(".pdf")]

    for file in all_files:
        full_path = os.path.join("data", file)
        print(f"正在处理: {full_path}...")

        # 获取单本书的切片
        chunks = smart_load_pdf(file_path=full_path)

        if chunks:
            all_chunks.extend(chunks)  # 2. 将当前书的切片追加到总列表中
            print(f"✅ 已提取 {len(chunks)} 个片段")

    # 3. 循环结束后，统一存入向量库
    if all_chunks:
        print(f"--- 正在构建向量库，总计 {len(all_chunks)} 个片段 ---")
        create_vector_store(all_chunks)
        print("--- 构建完成 ---")
    else:
        print("错误: 未能从 data/ 目录下的 PDF 中提取到任何有效文字。")


if __name__ == "__main__":
    main()
