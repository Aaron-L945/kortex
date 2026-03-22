from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    FAISS_INDEX_PATH: str = "./data/faiss_index"
    DOCUMENTS_PATH: str = "./data/documents"
    EMBED_MODEL: str = "BAAI/bge-large-zh-v1.5"

    class Config:
        env_file = ".env"


settings = Settings()
