from typing import Optional
from fastapi import FastAPI, HTTPException
from app.api.v1.endpoints import documents, search
from app.core.config import settings
from app.core.vector_store import ChromaDBManager
from app.core.dependency import set_chromadb_manager_instance
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Context-Aware Semantic Search API with ChromaDB")

@app.on_event("startup")
async def startup_event():
    global chromadb_manager_instance
    logger.info("FastAPI application startup...")

    chromadb_manager_instance = ChromaDBManager()  # Initialize Chroma manager
    set_chromadb_manager_instance(chromadb_manager_instance) 
    if chromadb_manager_instance.client is None:
        logger.error("CRITICAL: ChromaDBManager failed to initialize client. Vector store functionality will be impaired.")
    else:
        logger.info("ChromaDBManager initialized successfully.")

    try:
        import spacy
        spacy.load(settings.SPACY_MODEL)
        logger.info(f"Ensured SpaCy model '{settings.SPACY_MODEL}' is available.")
    except Exception as e:
        logger.error(f"Failed to load/ensure spaCy model '{settings.SPACY_MODEL}': {e}")



app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API..."}
