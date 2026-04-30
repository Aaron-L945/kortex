import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config

def get_embedding_model():
    print(f"正在从本地加载模型: {Config.LOCAL_MODEL_PATH}")
    
    # 检查路径是否存在，防止报错
    if not os.path.exists(Config.LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"未找到模型路径，请检查: {Config.LOCAL_MODEL_PATH}")
        
    return HuggingFaceEmbeddings(
        model_name=Config.LOCAL_MODEL_PATH,
        model_kwargs={'device': 'cpu'} # 如果有显卡可改为 'cuda'
    )

def create_vector_store(chunks):
    embeddings = get_embedding_model()
    
    print(f"开始本地向量化计算 (共 {len(chunks)} 个切片)...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(Config.VECTOR_DB_PATH), exist_ok=True)
    
    vector_db.save_local(Config.VECTOR_DB_PATH)
    print(f"✅ 向量库已保存至: {Config.VECTOR_DB_PATH}")
    return vector_db

def load_vector_store():
    embeddings = get_embedding_model()
    return FAISS.load_local(
        Config.VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )