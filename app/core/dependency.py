# app/core/dependency.py

from fastapi import HTTPException
from app.core.vector_store import ChromaDBManager
import logging

logger = logging.getLogger(__name__)

# This instance will be initialized by main.py on startup
chromadb_manager_instance: ChromaDBManager = None

def set_chromadb_manager_instance(instance: ChromaDBManager):
    global chromadb_manager_instance
    chromadb_manager_instance = instance

def get_chromadb_manager() -> ChromaDBManager:
    if chromadb_manager_instance is None:
        logger.critical("ChromaDBManager is not initialized! This indicates a startup problem.")
        raise HTTPException(status_code=503, detail="Vector store service is unavailable.")
    return chromadb_manager_instance
