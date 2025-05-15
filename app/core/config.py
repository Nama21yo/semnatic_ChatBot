from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # model-name - all-mpnet-base-v2
    MODEL_NAME: str = "all-MiniLM-L6-v2" # SBERT model

    EMBEDDING_DM: int = 384 # 768
    UPLOAD_DIR: str = "data/uploads"
    SPACY_MODEL: str = "en_core_web_sm"

    CHAT_HISTORY_DIR: str = "data/chat_histories" # For FileChatMessageHistory

   # --- ChromaDB Specific Settings ---
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    # Collection names in Chroma will be derived from session_id
    NORMALIZE_EMBEDDINGS: bool = True 

    GOOGLE_API_KEY: Optional[str] = None # Loaded from .env
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
    
    # Chunking parameters
    CHUNK_SIZE: int = 500 # Target character size for chunks
    CHUNK_OVERLAP: int = 50 # Character overlap between chunks
    # Add other settings as needed
     # For Custom Retriever
    SIMILARITY_THRESHOLD_REDUNDANT_FILTER: float = 0.95 # For EmbeddingsRedundantFilter
    INITIAL_RETRIEVAL_K: int = 10 # How many docs to fetch before filtering
    FINAL_RETRIEVAL_K_FOR_QA: int = 3 # How many docs to pass to LLM after filtering


    class Config:
        env_file = ".env"               # Tells Pydantic-Settings to look for a .env file
        env_file_encoding = 'utf-8'     # Specifies encoding for the .env file
        extra = "ignore"                # Tells Pydantic-Settings to ignore extra variables in the .env file not defined in the Settings class

settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True) # Create ChromaDB persist directory
Path(settings.CHAT_HISTORY_DIR).mkdir(parents=True, exist_ok=True) # Create chat history directory
