import faiss
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from app.core.embedding_generator import generate_embeddings
import logging

logger = logging.getLogger(__name__)

class FaissManager:
    def __init__(self):
        self.index_dir = Path(settings.FAISS_INDEX_DIR)
        self.metadata_dir = Path(settings.FAISS_METADATA_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_paths(self, session_id: str) -> Tuple[Path, Path]:
        session_index_file = self.index_dir / f"{session_id}.index"
        session_metadata_file = self.metadata_dir / f"{session_id}_meta.json"
        return session_index_file, session_metadata_file

    def _load_metadata(self, session_id: str) -> List[Dict[str, Any]]:
        _, meta_file = self._get_session_paths(session_id)
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON metadata for session {session_id}. Returning empty list.")
                return [] # Corrupted file
        return []

    def _save_metadata(self, session_id: str, metadata: List[Dict[str, Any]]):
        _, meta_file = self._get_session_paths(session_id)
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_or_create_index(self, session_id: str, dimension: int) -> Optional[faiss.Index]:
        index_file, _ = self._get_session_paths(session_id)
        if index_file.exists():
            try:
                logger.info(f"Loading existing FAISS index for session {session_id} from {index_file}")
                return faiss.read_index(str(index_file))
            except Exception as e:
                logger.error(f"Failed to load FAISS index {index_file}: {e}. Will try to create a new one.")
                # Potentially delete corrupted index file here
                if index_file.exists(): os.remove(index_file) 
                # And corresponding metadata might be out of sync, so clear it too
                self.delete_session_metadata_file_only(session_id)


        logger.info(f"Creating new FAISS IndexIVFFlat for session {session_id}")
        # Using IndexFlatIP as quantizer for cosine similarity (requires normalized vectors)
        quantizer = faiss.IndexFlatIP(dimension)
        # nlist is the number of clusters/centroids
        index = faiss.IndexIVFFlat(quantizer, dimension, settings.FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
        # METRIC_INNER_PRODUCT with normalized vectors gives cosine similarity.
        # If using METRIC_L2, ensure embeddings are normalized, then smaller L2 dist is better.
        return index
    
    def delete_session_metadata_file_only(self, session_id: str):
        _, meta_file = self._get_session_paths(session_id)
        if meta_file.exists():
            try:
                os.remove(meta_file)
                logger.info(f"Deleted metadata file {meta_file} for session {session_id}.")
            except OSError as e:
                logger.error(f"Error deleting metadata file {meta_file}: {e}")


    def upsert_chunks(self, session_id: str, new_chunks_data: List[Dict[str, Any]]):
        """
        Adds chunks to the FAISS index for a given session.
        Each item in `new_chunks_data` is a dict from `process_and_index_file`:
        { "id": "logical_chunk_id", "text": "chunk text", "metadata": {"source_document": ..., "page_number": ...} }
        """
        if not new_chunks_data:
            logger.info("No new chunks to upsert.")
            return

        texts_to_embed = [chunk['text'] for chunk in new_chunks_data]
        embeddings = generate_embeddings(texts_to_embed)

        if embeddings.shape[0] == 0:
            logger.warning("Generated zero embeddings. Skipping upsert.")
            return
        
        dimension = embeddings.shape[1]
        index_file, _ = self._get_session_paths(session_id)
        
        index = self._load_or_create_index(session_id, dimension)
        if index is None: # Should not happen if _load_or_create_index handles errors by creating new
            logger.error(f"Failed to load or create index for session {session_id}. Aborting upsert.")
            return

        existing_metadata = self._load_metadata(session_id)

        # Training for IndexIVFFlat:
        # Train if the index is new (not trained) and we have enough vectors.
        # For simplicity, if index.ntotal is 0 (empty index) and it's an IndexIVF, train it.
        if isinstance(index, faiss.IndexIVF) and not index.is_trained and embeddings.shape[0] > 0:
            # Use current embeddings for training if it's the first batch.
            # A more robust strategy might collect more data or use dedicated training set.
            logger.info(f"Training FAISS IndexIVFFlat for session {session_id} with {embeddings.shape[0]} vectors.")
            # We need at least FAISS_NLIST * 39 vectors by default for training k-means in IVF, can be less with `MinMaxPointsPerCentroid`
            # or if k-means parameters are adjusted. For small number of vectors, training might fail or be suboptimal.
            # Let's check against nlist.
            if embeddings.shape[0] < index.nlist: # faiss.IndexIVF.nlist
                 logger.warning(f"Not enough vectors ({embeddings.shape[0]}) to train IndexIVFFlat with nlist={index.nlist}. Need at least {index.nlist}. Consider reducing nlist or adding more data first. Will attempt with current data.")
            try:
                index.train(embeddings)
                logger.info(f"Training complete for session {session_id}.")
            except Exception as train_e:
                logger.error(f"FAISS training failed for session {session_id}: {train_e}. Index might not be usable.")
                # Potentially fall back to a simpler index like IndexFlatIP if IVF training fails consistently for small data.
                return # Abort if training fails for now
        
        if isinstance(index, faiss.IndexIVF) and not index.is_trained:
            logger.warning(f"IndexIVF for session {session_id} is not trained. Adding vectors might fail or lead to poor performance. Ensure training step is successful.")
            # If training failed above, or wasn't triggered, we might not want to proceed.
            return

        # Add embeddings to FAISS index
        # FAISS `add` assigns sequential IDs. These IDs are 0-based indices into the conceptual "flat" list of all vectors in the index.
        index.add(embeddings)
        logger.info(f"Added {embeddings.shape[0]} new embeddings to FAISS index for session {session_id}. Total vectors: {index.ntotal}")

        # Prepare new metadata entries. The order must match the order of `embeddings` added.
        # The FAISS ID will be `len(existing_metadata) + i` for the i-th new chunk.
        start_faiss_id = len(existing_metadata)
        for i, chunk_meta_info in enumerate(new_chunks_data):
            # Store the full chunk metadata provided by the parsing/chunking stage
            # This includes source_document, page_number, and the original text.
            full_chunk_details = {
                "faiss_id": start_faiss_id + i, # This is the index in the metadata list
                "logical_id": chunk_meta_info["id"], # e.g., filename_chunk_0
                "text": chunk_meta_info["text"],
                **chunk_meta_info["metadata"] # source_document, page_number
            }
            existing_metadata.append(full_chunk_details)
        
        # Save updated index and metadata
        try:
            faiss.write_index(index, str(index_file))
            self._save_metadata(session_id, existing_metadata)
            logger.info(f"FAISS index and metadata saved for session {session_id}.")
        except Exception as e:
            logger.error(f"Error saving FAISS index or metadata for session {session_id}: {e}")


    def query_vectors(self, session_id: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        index_file, _ = self._get_session_paths(session_id)
        if not index_file.exists():
            logger.warning(f"No FAISS index found for session {session_id} at {index_file}.")
            return []

        try:
            index = faiss.read_index(str(index_file))
            all_metadata = self._load_metadata(session_id)
        except Exception as e:
            logger.error(f"Error loading FAISS index or metadata for query (session {session_id}): {e}")
            return []

        if index.ntotal == 0:
            logger.info(f"FAISS index for session {session_id} is empty.")
            return []

        query_embedding = generate_embeddings([query_text])
        if query_embedding.shape[0] == 0:
            return []
        query_embedding = query_embedding.astype('float32') # Ensure correct type

        # For IndexIVFFlat, set nprobe (number of Voronoi cells to search)
        if isinstance(index, faiss.IndexIVF):
            index.nprobe = settings.FAISS_NPROBE
            logger.debug(f"Set nprobe to {index.nprobe} for session {session_id} query.")
            if not index.is_trained:
                logger.error(f"Querying an untrained IndexIVF for session {session_id}. Results will be incorrect or fail.")
                # This should ideally not happen if upsert handles training.
                return []

        # Search the index
        # distances are inner products (higher is better for IndexFlatIP / cosine)
        # faiss_ids are the 0-based indices of the vectors in the index
        distances, faiss_ids = index.search(query_embedding, top_k) 
        
        results = []
        for i in range(len(faiss_ids[0])):
            faiss_id = faiss_ids[0][i]
            if faiss_id == -1: # FAISS returns -1 if fewer than top_k results are found
                continue
            
            score = float(distances[0][i])
            
            # Retrieve corresponding metadata
            if 0 <= faiss_id < len(all_metadata):
                chunk_meta = all_metadata[faiss_id]
                results.append({
                    "chunk": {
                        "text": chunk_meta["text"],
                        "page_number": chunk_meta.get("page_number"),
                        "source_document": chunk_meta.get("source_document", "Unknown")
                    },
                    "score": score, # Higher is better for Inner Product
                    "id": int(faiss_id) # The FAISS sequential ID
                })
            else:
                logger.warning(f"FAISS ID {faiss_id} out of bounds for metadata list (len {len(all_metadata)}) for session {session_id}.")
        
        # Sort by score descending if not already (FAISS IP search should return in order)
        # results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_all_data_for_session_viz(self, session_id: str, limit: int = 200) -> Dict[str, Any]:
        index_file, _ = self._get_session_paths(session_id)
        all_metadata = self._load_metadata(session_id)

        if not all_metadata:
            return {"embeddings": [], "chunks_metadata": []}
        
        # For visualization, we need embeddings.
        # Option 1: Reconstruct from FAISS index (can be slow for many, but fine for `limit`)
        # Option 2: Store raw embeddings separately (adds storage but faster retrieval)
        # Let's go with reconstruction for now.
        
        embeddings_list = []
        chunks_metadata_for_viz = []

        if index_file.exists():
            try:
                index = faiss.read_index(str(index_file))
                if index.ntotal > 0:
                    # Sample IDs to reconstruct, or take the first `limit`
                    num_to_reconstruct = min(limit, len(all_metadata), index.ntotal)
                    
                    for i in range(num_to_reconstruct):
                        # The FAISS ID `i` corresponds to the i-th item in `all_metadata`
                        # if the metadata list is perfectly aligned with FAISS internal order.
                        # And faiss_id in metadata should be `i`.
                        meta_item = all_metadata[i]
                        if meta_item.get("faiss_id") == i: # Sanity check
                            try:
                                reconstructed_vector = index.reconstruct(i).tolist() # Reconstruct by FAISS ID
                                embeddings_list.append(reconstructed_vector)
                                chunks_metadata_for_viz.append(meta_item) # Already contains text, page, source
                            except RuntimeError as recon_err:
                                # reconstruction may not be supported by all index types or if not direct_map
                                logger.warning(f"Could not reconstruct vector {i} for session {session_id}: {recon_err}. Trying to use stored text only.")
                                # Fallback: just use metadata without embedding if reconstruction fails
                                # This part needs careful handling depending on chosen FAISS index.
                                # For IndexIVFFlat, reconstruction should work.
                                # If direct_map is not set for IVF, reconstruction can be tricky.
                                # For simplicity, if reconstruction fails, we might skip that point for viz.
                        else:
                             logger.warning(f"Metadata mismatch for faiss_id {i} in session {session_id}")


                # If index doesn't exist or reconstruction fails for all, we can still return metadata text
                if not embeddings_list and chunks_metadata_for_viz: # Only metadata, no embeddings
                    logger.warning(f"Visualization for session {session_id} will only use metadata text as embeddings could not be retrieved/reconstructed.")

            except Exception as e:
                logger.error(f"Error processing FAISS index for visualization (session {session_id}): {e}")
        else: # Index file doesn't exist, but metadata might
            logger.warning(f"No FAISS index file for session {session_id}, visualization might only use metadata text.")
            chunks_metadata_for_viz = all_metadata[:limit]


        # If embeddings could not be reconstructed, send empty list for embeddings
        # and the frontend can decide how to handle it (e.g., show text only, no plot).
        if not embeddings_list and chunks_metadata_for_viz:
            # Create dummy embeddings if only metadata is available and plotting is desired (e.g., random projection of text embeddings)
            # For now, just return metadata. Frontend t-SNE will fail if embeddings_list is empty.
            # Better: if embeddings_list is empty, don't attempt t-SNE in frontend for those.
             return {"embeddings": [], "chunks_metadata": chunks_metadata_for_viz}


        return {
            "embeddings": embeddings_list,
            "chunks_metadata": chunks_metadata_for_viz # This is list of dicts
        }

    def delete_session_data(self, session_id: str):
        index_file, meta_file = self._get_session_paths(session_id)
        deleted_something = False
        if index_file.exists():
            try:
                os.remove(index_file)
                logger.info(f"Deleted FAISS index file: {index_file}")
                deleted_something = True
            except OSError as e:
                logger.error(f"Error deleting FAISS index file {index_file}: {e}")
        
        if meta_file.exists():
            try:
                os.remove(meta_file)
                logger.info(f"Deleted metadata file: {meta_file}")
                deleted_something = True
            except OSError as e:
                logger.error(f"Error deleting metadata file {meta_file}: {e}")
        
        if not deleted_something:
            logger.info(f"No data found to delete for session {session_id}.")