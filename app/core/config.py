from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "all-MiniLM-L6-v2" # SBERT model
    UPLOAD_DIR: str = "data/uploads"
    KB_DIR: str = "data/knowledge_base"
    # SPACY_MODEL: str = "en_core_web_sm"
    # Add other settings as needed


settings = Settings()
