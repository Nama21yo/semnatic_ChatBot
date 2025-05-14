import spacy
from typing import List, Tuple, Dict, Any
from app.core.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import logging
logger = logging.getLogger(__name__)

# Load spaCy model (consider doing this at app startup if used frequently)
try:
    nlp = spacy.load(settings.SPACY_MODEL)
    logger.info(f"SpaCy model '{settings.SPACY_MODEL}' loaded for NER.")
except OSError:
    logger.error(f"SpaCy model '{settings.SPACY_MODEL}' not found. Please download it: python -m spacy download {settings.SPACY_MODEL}")
    nlp = None # Fallback

def chunk_documents(langchain_docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Chunks Langchain Documents using RecursiveCharacterTextSplitter."""
    if not langchain_docs:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Adds start index of chunk in original document
    )
    
    split_docs = text_splitter.split_documents(langchain_docs)
    
    # The splitter preserves metadata from original docs and adds "start_index".
    # We want to prepare these for Pinecone (id, text, metadata)
    # The SearchService will format them for Pinecone upsert.
    return split_docs


def ner_tag_query(query: str) -> List[Tuple[str, str]]:
    if nlp is None:
        logger.warning("SpaCy model not loaded. NER tagging is unavailable.")
        return []
    doc = nlp(query)
    return [(ent.text, ent.label_) for ent in doc.ents]