import faiss
import numpy as np
import json
from typing import List, Dict, Tuple
from app.core.embedding_generator import model # SBERT model for query embedding
from app.core.config import settings
import os

def index_embeddings(embeddings: np.ndarray, chunks_metadata: List[Dict], index_path: str, metadata_path: str):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean)
    # For cosine similarity with IndexFlatL2, normalize vectors first then L2 is equivalent to maximizing cosine.
    # Or use IndexFlatIP (Inner Product)
    # faiss.normalize_L2(embeddings) # Normalize for IndexFlatL2 to act like cosine
    # index = faiss.IndexFlatIP(dimension) # Use Inner Product for cosine directly
    
    index.add(embeddings)
    faiss.write_index(index, index_path)
    
    with open(metadata_path, "w") as f:
        json.dump(chunks_metadata, f)
    print(f"Indexed {len(chunks_metadata)} chunks. Index: {index_path}, Metadata: {metadata_path}")


def search_indexed_embeddings(query_text: str, session_id: str, top_k: int = 3) -> List[Dict]:
    session_kb_dir = os.path.join(settings.KB_DIR, session_id)
    index_path = os.path.join(session_kb_dir, "faiss_index.idx")
    metadata_path = os.path.join(session_kb_dir, "chunks_metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return [] # No knowledge base for this session

    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        chunks_metadata = json.load(f)

    query_embedding = model.encode([query_text]).astype('float32')
    # faiss.normalize_L2(query_embedding) # if using IndexFlatL2 after normalizing indexed embeddings

    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 : continue # FAISS can return -1 if not enough results
        results.append({
            "chunk": chunks_metadata[idx], # Contains text, page_number, source_document
            "score": 1 - distances[0][i] if index.metric_type == faiss.METRIC_L2 else distances[0][i] # Convert L2 dist to similarity-like score
            # For IndexFlatIP, distances are inner products (higher is better)
            # For IndexFlatL2, distances are L2 (lower is better). 
            # If normalized, L2^2 = 2 - 2*cos_sim. So, smaller L2 means higher cos_sim.
            # A simple proxy for similarity score with L2: 1 / (1 + distance) or exp(-distance)
            # Or, if vectors are normalized, score = (2 - D^2) / 2 for cosine similarity
        })
    return results
