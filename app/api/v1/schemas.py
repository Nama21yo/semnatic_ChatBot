from pydantic import BaseModel
from typing import List, Optional

class DocumentUploadResponse(BaseModel):
    filename : str
    message : str
    doc_id : str # unique id for uploading document

class Chunk(BaseModel):
    text : str
    page_number : Optional[int] = None
    source_document : str # filename 

class SearchResultItem(BaseModel):
    chunk : Chunk
    score : float

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    query_id : str # to track context

class Queryrequest(BaseModel):
    query : str
    session_id : str
    previous_query_id : Optional[str] = None

