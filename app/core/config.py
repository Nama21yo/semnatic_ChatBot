from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    MODEL_NAME: str = "all-MiniLM-L6-v2" # SBERT model
    EMBEDDING_DM: int = 384
    UPLOAD_DIR: str = "data/uploads"
    SPACY_MODEL: str = "en_core_web_sm"

    FAISS_INDEX_DIR: str = "data/faiss_indexes"
    FAISS_METADATA_DIR: str = "data/faiss_metadata"
    FAISS_NLIST: int = 4  # Number of Voronoi cells (centroids) for IndexIVFFlat
                           # Adjust based on expected number of vectors.
                           # Good starting point: sqrt(total_vectors) to 4*sqrt(total_vectors)
    FAISS_NPROBE: int = 1   # Number of cells to search. Higher is more accurate but slower.
    FAISS_TRAINING_THRESHOLD: int = 1000 # Min vectors to collect before initial training for IndexIVFFlat
                                       # Or train if index is new and has some vectors.
    # If True, normalizes embeddings before adding to FAISS IndexFlatIP for true cosine.
    # SBERT `encode` usually returns normalized embeddings if `normalize_embeddings=True` is passed to SentenceTransformer model init or encode call.
    # For `all-MiniLM-L6-v2`, it's generally good to normalize.
    NORMALIZE_EMBEDDINGS: bool = True 
    
    # Chunking parameters
    CHUNK_SIZE: int = 500 # Target character size for chunks
    CHUNK_OVERLAP: int = 50 # Character overlap between chunks
    # Add other settings as needed

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore" # Ignore extra fields from .env


settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.FAISS_METADATA_DIR).mkdir(parents=True, exist_ok=True)

