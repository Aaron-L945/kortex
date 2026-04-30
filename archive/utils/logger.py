from loguru import logger
import os

def init_logger(log_path="logs/rag.log"):
    if not os.path.exists("logs"): os.makedirs("logs")
    logger.add(log_path, rotation="10 MB", retention="10 days", encoding="utf-8")
    return logger