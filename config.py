import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 基础目录配置
    TOP_K_RERANK = 6
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 精准指向 ModelScope 下载的本地模型路径
    # 请确保文件夹名称中的下划线和版本号完全匹配
    LOCAL_MODEL_PATH = os.path.expanduser(os.getenv("EMBEDDING_MODEL_NAME"))
    RERANKER_MODEL_PATH = os.path.expanduser(os.getenv("RERANKER_MODEL_PATH"))
    
    DOC_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store/faiss_index")
    
    # 检索相关
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 20))
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 8))
    RRF_K = int(os.getenv("RRF_K", 60))
    
    # 切片相关
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # 其他
    INCLUDE_THINK = os.getenv("INCLUDE_THINK", "True").lower() == "true"
    RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.3))