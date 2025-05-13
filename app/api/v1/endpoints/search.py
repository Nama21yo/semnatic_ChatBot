from fastapi import APIRouter, HTTPException, Body
from app.api.v1.schemas import QueryRequest, SearchResponse # Add QA schemas if doing bonus
from app.services.search_service import SearchService
from app.core.config import settings
import os

router = APIRouter()
search_service_instance = SearchService() # Instantiate the service

@router.post("/", response_model=SearchResponse)
async def query_documents(request: QueryRequest = Body(...)):
    # Check if knowledge base for session_id exists
    session_kb_dir = os.path.join(settings.KB_DIR, request.session_id)
    if not os.path.exists(session_kb_dir):
        raise HTTPException(status_code=404, detail=f"Knowledge base for session '{request.session_id}' not found. Please upload documents first.")
    
    try:
        response = search_service_instance.perform_search(request)
        return response
    except Exception as e:
        # Log the exception e
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Error during search process.")

# Bonus: QA Endpoint
@router.post("/qa")
async def question_answering(request: QueryRequest = Body(...)): # Reusing QueryRequest for simplicity
    session_kb_dir = os.path.join(settings.KB_DIR, request.session_id)
    if not os.path.exists(session_kb_dir):
        raise HTTPException(status_code=404, detail=f"Knowledge base for session '{request.session_id}' not found.")

    answer = search_service_instance.perform_qa(request.session_id, request.query)
    if answer:
        return {"question": request.query, "answer": answer, "session_id": request.session_id}
    else:
        return {"question": request.query, "answer": "Could not find an answer.", "session_id": request.session_id}
