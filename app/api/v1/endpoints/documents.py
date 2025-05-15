from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from app.api.v1.schemas import DocumentUploadResponse, EmbeddingsDataResponse
from app.core.file_parser import parse_file
from app.core.nlp_utils import chunk_documents
from app.core.vector_store import ChromaDBManager
from app.core.config import settings
from app.core.dependency import get_chromadb_manager
import shutil
import os
import logging
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)


def save_uploaded_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


async def process_and_index_file(
    session_id: str,
    file_path: Path,
    filename: str,
    chroma_manager: ChromaDBManager
):
    logger.info(f"Background processing file: {filename} for session: {session_id} with ChromaDB")
    try:
        raw_docs = parse_file(str(file_path), filename)
        if not raw_docs:
            logger.warning(f"No content parsed from {filename}.")
            return

        chunked_lc_docs = chunk_documents(raw_docs)
        if not chunked_lc_docs:
            logger.warning(f"No chunks generated from {filename}.")
            return

        for i, doc in enumerate(chunked_lc_docs):
            doc.metadata["chroma_id"] = doc.metadata.get("chroma_id", f"{filename}_chunk_{i}_{session_id}")
            doc.metadata["source_document"] = filename

        success = chroma_manager.upsert_langchain_documents(session_id, chunked_lc_docs)
        if success:
            logger.info(f"Successfully processed and indexed {filename} to ChromaDB for session {session_id}.")
        else:
            logger.error(f"Failed to index {filename} to ChromaDB for session {session_id}.")

    except Exception as e:
        logger.error(f"Error processing file {filename} for ChromaDB (session {session_id}): {e}", exc_info=True)
    finally:
        try:
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"Cleaned up uploaded file: {file_path}")
        except OSError as e_os:
            logger.error(f"Error cleaning up file {file_path}: {e_os}")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chroma_manager: ChromaDBManager = Depends(get_chromadb_manager)
):
    if chroma_manager.client is None:
        raise HTTPException(status_code=503, detail="Vector store (ChromaDB) service is unavailable.")

    upload_dir_for_session = Path(settings.UPLOAD_DIR) / session_id
    upload_dir_for_session.mkdir(parents=True, exist_ok=True)
    file_location = upload_dir_for_session / file.filename
    save_uploaded_file(file, file_location)

    background_tasks.add_task(
        process_and_index_file,
        session_id,
        file_location,
        file.filename,
        chroma_manager
    )

    logger.info(f"File {file.filename} received for session {session_id}. Processing in background.")
    return DocumentUploadResponse(
        filename=file.filename,
        message="File received and scheduled for processing. Indexing may take time.",
        doc_id=file.filename,
        session_id=session_id
    )


@router.get("/knowledge_base_data", response_model=EmbeddingsDataResponse)
async def get_knowledge_base_data_for_viz(
    session_id: str,
    limit: int = 200,
    chroma_manager: ChromaDBManager = Depends(get_chromadb_manager)
):
    if not chroma_manager.client:
        raise HTTPException(status_code=503, detail="ChromaDB client not available.")

    collection_name = chroma_manager._get_collection_name(session_id)
    try:
        collection = chroma_manager.client.get_collection(name=collection_name)
        data = collection.get(include=["embeddings", "metadatas"], limit=limit)

        embeddings = data.get("embeddings", [])
        chunks_metadata = data.get("metadatas", [])

        return EmbeddingsDataResponse(
            embeddings=embeddings,
            chunks_metadata=chunks_metadata
        )
    except Exception as e:
        logger.error(f"Error retrieving ChromaDB data for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")


@router.delete("/session_data/{session_id}", status_code=204)
async def delete_session_knowledge_base(
    session_id: str,
    chroma_manager: ChromaDBManager = Depends(get_chromadb_manager)
):
    try:
        chroma_manager.delete_session_data(session_id)
        logger.info(f"Successfully deleted all ChromaDB data for session: {session_id}")
    except Exception as e:
        logger.error(f"Error deleting ChromaDB data for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete session data: {str(e)}")
