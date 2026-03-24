#!/bin/bash

# 1. 创建目录结构
echo "正在创建项目目录..."
mkdir -p kb_agent/data kb_agent/vector_store
cd kb_agent

# 2. 写入 config.py
cat <<EOF > config.py
import os

class Config:
    # 推荐使用 BGE-M3，中英兼顾
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    DOC_DIR = "data/"
    VECTOR_DB_PATH = "vector_store/faiss_index"
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 60
EOF

# 3. 写入 loader.py
cat <<EOF > loader.py
import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

def clean_text(text):
    text = re.sub(r'\x0c', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text

def load_and_split():
    all_chunks = []
    if not os.listdir(Config.DOC_DIR):
        return []
        
    for file in os.listdir(Config.DOC_DIR):
        if file.endswith(".pdf"):
            print(f"正在解析: {file}")
            loader = PyMuPDFLoader(os.path.join(Config.DOC_DIR, file))
            documents = loader.load()
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
            )
            all_chunks.extend(text_splitter.split_documents(documents))
    return all_chunks
EOF

# 4. 写入 embedder.py
cat <<EOF > embedder.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config

def create_vector_store(chunks):
    print(f"正在对 {len(chunks)} 个切片进行向量化...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(Config.VECTOR_DB_PATH)
    print(f"向量库已保存至: {Config.VECTOR_DB_PATH}")
    return vector_db
EOF

# 5. 写入 main.py
cat <<EOF > main.py
import os
from loader import load_and_split
from embedder import create_vector_store

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("--- 知识库构建开始 ---")
    chunks = load_and_split()
    if chunks:
        create_vector_store(chunks)
        print("--- 构建完成 ---")
    else:
        print("错误: data/ 目录下没有找到 PDF 文件。")

if __name__ == "__main__":
    main()
EOF

# 6. 写入 requirements.txt
cat <<EOF > requirements.txt
langchain
langchain-community
langchain-huggingface
faiss-cpu
pymupdf
sentence-transformers
EOF

echo "项目构建完成！"
echo "请执行以下命令开始运行："
echo "1. pip install -r requirements.txt"
echo "2. 将 PDF 文件放入 kb_agent/data/"
echo "3. python main.py"
