from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np
from typing import List
import faiss 

# consider lazy loading if memory is a concern and it is not always used
try:
    model = SentenceTransformer(settings.MODEL_NAME)
    print(f"SBERT model '{settings.MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading SBERT model: {e}")
    model = None

def generate_embeddings(texts: List[str]): # returns np.ndarray
    if model is None:
        raise RuntimeError("SBERT model is not loaded. Cannot generate embeddings.")
    if not texts:
        return np.array([])
    
    # Some SBERT models output normalized embeddings by default.
    # If using normalize_embeddings=True with SentenceTransformer, this explicit step might be redundant.
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    if settings.NORMALIZE_EMBEDDINGS and embeddings.ndim > 0 and embeddings.shape[0] > 0:
        # FAISS normalization is efficient
        # It is used to change the magnitude of the Vector into Unit Vector
        # Then cosine similarity will be just dot product
        faiss.normalize_L2(embeddings)

    return embeddings.astype('float32')