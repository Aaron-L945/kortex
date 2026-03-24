import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 基础目录配置
    TOP_K_RERANK = 6
    INCLUDE_THINK = os.getenv("AGENT_INCLUDE_THINK", "True").lower() == "true"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 精准指向 ModelScope 下载的本地模型路径
    # 请确保文件夹名称中的下划线和版本号完全匹配
    LOCAL_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-large-zh-v1___5")
    
    DOC_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store/faiss_index")
    
    # 切片配置
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 60
