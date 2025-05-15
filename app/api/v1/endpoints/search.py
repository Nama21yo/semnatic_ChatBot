from fastapi import APIRouter, HTTPException, Body, Depends
from app.api.v1.schemas import QueryRequest, SearchResponse, QAQueryRequest, QAResponse, FeedbackRequest, FeedbackResponse
from app.services.search_service import SearchService
# from app.core.vector_store import PineconeManager # Remove
from app.core.vector_store import ChromaDBManager # Add
from app.core.dependency import get_chromadb_manager # Update dependency import
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency to get SearchService instance, now with FaissManager
def get_search_service(chroma_manager: ChromaDBManager = Depends(get_chromadb_manager)) -> SearchService:
    # The get_faiss_manager dependency already handles if faiss_manager_instance is None
    return SearchService(chroma_manager)

# Endpoints (@router.post("/"), @router.post("/qa"), @router.post("/feedback"))
# remain largely the same in structure, as they depend on `get_search_service`
# which now correctly instantiates `SearchService` with `FaissManager`.
@router.post("/semantic_search", response_model=SearchResponse) # New endpoint name for clarity
async def semantic_search_documents(
    request: QueryRequest = Body(...), # QueryRequest has query, session_id, top_k
    search_service: SearchService = Depends(get_search_service)
):
    try:
        response = await search_service.perform_semantic_search(request)
        return response
    except Exception as e:
        logger.error(f"Semantic Search service error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during semantic search: {str(e)}")

@router.post("/conversational_qa", response_model=QAResponse) # New endpoint name
async def conversational_question_answering(
    request: QAQueryRequest = Body(...), # QAQueryRequest has question, session_id
    search_service: SearchService = Depends(get_search_service)
):
    try:
        response = await search_service.perform_conversational_qa(request)
        return response
    except Exception as e:
        logger.error(f"Conversational QA service error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during conversational QA: {str(e)}")