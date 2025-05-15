# app/core/vector_store_chroma.py
import chromadb
from app.core.config import settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

# Global embedding instance (good)
LANGCHAIN_HF_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None
try:
    LANGCHAIN_HF_EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
        model_name=settings.MODEL_NAME,
        model_kwargs={'device': 'cuda' if settings.NORMALIZE_EMBEDDINGS else 'cpu'},
        encode_kwargs={'normalize_embeddings': settings.NORMALIZE_EMBEDDINGS}
    )
    logger.info(f"Global Langchain HuggingFaceEmbeddings loaded with {settings.MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load global HuggingFaceEmbeddings: {e}", exc_info=True)


class ChromaDBManager:
    _client: Optional[chromadb.PersistentClient] = None # Singleton client

    def __init__(self):
        if ChromaDBManager._client is None:
            try:
                Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
                ChromaDBManager._client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
                logger.info(f"ChromaDB PersistentClient initialized. Root data directory: {settings.CHROMA_PERSIST_DIR}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB PersistentClient: {e}", exc_info=True)
                raise RuntimeError(f"ChromaDB PersistentClient could not be initialized at {settings.CHROMA_PERSIST_DIR}") from e
        
        self.client = ChromaDBManager._client # Assign instance client

        if LANGCHAIN_HF_EMBEDDINGS_INSTANCE is None:
            raise RuntimeError("Global HuggingFaceEmbeddings instance is not available.")
        self.embedding_function_for_langchain = LANGCHAIN_HF_EMBEDDINGS_INSTANCE

    def _get_collection_name(self, session_id: str) -> str:
        return f"session_{session_id.replace('-', '_')}"

    def _ensure_collection_exists_with_cosine(self, collection_name: str) -> Optional[chromadb.Collection]:
        """Ensures a collection exists, configured for cosine similarity."""
        if not self.client: return None
        try:
            # Try to get it first. If it exists, check its metadata if possible (Chroma API for this is limited).
            # For simplicity, get_or_create_collection with desired metadata is often used.
            # If metadata differs, Chroma might raise an error or update, depending on version.
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} # Specify cosine distance
            )
            # logger.info(f"ChromaDB collection '{collection_name}' (cosine) ensured.")
            return collection
        except Exception as e:
            logger.error(f"Failed to ensure/create ChromaDB collection '{collection_name}' with cosine metric: {e}", exc_info=True)
            return None

    def upsert_langchain_documents(self, session_id: str, lc_documents: List[LangchainDocument]):
        if not self.client or not self.embedding_function_for_langchain:
            logger.error("ChromaDB client or embedding function not initialized for upsert.")
            return False
        if not lc_documents: return True

        collection_name = self._get_collection_name(session_id)
        
        # Ensure the collection exists and is configured for cosine similarity
        native_collection = self._ensure_collection_exists_with_cosine(collection_name)
        if not native_collection:
            logger.error(f"Could not prepare collection '{collection_name}' for upsert.")
            return False

        try:
            ids = [doc.metadata.get("chroma_id", f"{doc.metadata.get('source_document', 'doc')}_{i}_{session_id[:8]}") 
                   for i, doc in enumerate(lc_documents)] # Generate more robust default IDs
            
            # Texts and metadatas for batch add
            texts_to_add = [doc.page_content for doc in lc_documents]
            metadatas_to_add = [doc.metadata for doc in lc_documents] # Langchain doc.metadata
            
            # Add documents directly to the native collection using its add method
            # This gives more control and uses the collection's pre-configured embedding function implicitly if set,
            # or we pass embeddings explicitly. Since we want Langchain's HF embeddings:
            embeddings_to_add = self.embedding_function_for_langchain.embed_documents(texts_to_add)

            native_collection.upsert( # Use upsert for add-or-update
                ids=ids,
                embeddings=embeddings_to_add,
                documents=texts_to_add, # Store the text content
                metadatas=metadatas_to_add # Store Langchain metadata
            )
            logger.info(f"Upserted {len(lc_documents)} documents to ChromaDB collection '{collection_name}'. Count: {native_collection.count()}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert documents to ChromaDB collection '{collection_name}': {e}", exc_info=True)
            return False

    def _get_langchain_chroma_store_instance(self, session_id: str) -> Optional[Chroma]:
        """
        Initializes and returns a Langchain Chroma vector store instance
        connected to the correct persistent client and collection.
        """
        if not self.client or not self.embedding_function_for_langchain:
            logger.error("ChromaDB client or embedding function not available for Langchain store.")
            return None
        
        collection_name = self._get_collection_name(session_id)
        
        try:
            # Crucially, we initialize Langchain's Chroma by passing our EXISTING client
            # and the collection name. Langchain will then use this client to interact
            # with the collection.
            # Also ensure the collection actually exists before Langchain tries to use it.
            self.client.get_collection(name=collection_name) # Will raise error if not found

            langchain_chroma_store = Chroma(
                client=self.client, # Use the shared PersistentClient
                collection_name=collection_name,
                embedding_function=self.embedding_function_for_langchain # Use the shared embedding instance
            )
            # logger.debug(f"Langchain Chroma store instance created for collection '{collection_name}'.")
            return langchain_chroma_store
        except chromadb.errors.CollectionNotDefinedError:
             logger.warning(f"Langchain Chroma store: Collection '{collection_name}' does not exist for session {session_id}.")
             return None
        except Exception as e:
            logger.error(f"Error creating Langchain Chroma store instance for '{collection_name}': {e}", exc_info=True)
            return None

    def query_documents_with_langchain_chroma(self, session_id: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        langchain_chroma_store = self._get_langchain_chroma_store_instance(session_id)
        if not langchain_chroma_store:
            logger.warning(f"Query: Langchain Chroma store not available for session {session_id}. Returning no results.")
            return []

        try:
            results_with_scores: List[Tuple[LangchainDocument, float]] = \
                langchain_chroma_store.similarity_search_with_relevance_scores(query=query_text, k=top_k)
            
            logger.info(f"Query '{query_text[:30]}...' returned {len(results_with_scores)} results from collection '{langchain_chroma_store._collection.name}'. Scores: {[s for _,s in results_with_scores]}")

            processed_results = []
            for doc, score in results_with_scores:
                processed_results.append({
                    "chunk": {
                        "text": doc.page_content,
                        "page_number": doc.metadata.get("page_number"),
                        "source_document": doc.metadata.get("source_document", "Unknown")
                    },
                    "score": score, # Higher is better
                    "id": doc.metadata.get("chroma_id", doc.metadata.get("id", str(uuid.uuid4()))) # Use chroma_id from metadata
                })
            return processed_results # Already sorted by relevance by Chroma
        except Exception as e:
            logger.error(f"Error querying ChromaDB collection via Langchain store: {e}", exc_info=True)
            return []

    def get_langchain_chroma_retriever(self, session_id: str, search_type="similarity", search_kwargs={"k": 3}):
        langchain_chroma_store = self._get_langchain_chroma_store_instance(session_id)
        if not langchain_chroma_store:
            return None
        return langchain_chroma_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def delete_session_collection(self, session_id: str):
        if not self.client: return
        collection_name = self._get_collection_name(session_id)
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' deleted for session {session_id}.")
        except chromadb.errors.CollectionNotDefinedError:
            logger.info(f"ChromaDB collection '{collection_name}' not found for deletion.")
        except Exception as e:
            logger.error(f"Error deleting ChromaDB collection '{collection_name}': {e}", exc_info=True)