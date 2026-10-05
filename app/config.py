from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "faces"
    face_match_threshold: float = 0.40
    face_search_min_score: float = 0.40
    face_search_default_limit: int = 5
    face_search_max_limit: int = 20
    model_dir: str = "./models"
    max_image_side: int = 640
    max_upload_bytes: int = 5_000_000
    embedding_dim: int = 512

def get_settings() -> Settings:
    return Settings()
