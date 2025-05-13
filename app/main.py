from fastapi import FastAPI
from app.api.v1.endpoints import documents, search
import os
from app.core.config import settings 


app = FastAPI(title="Semantic Search ChatBot")

# create directories if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.KB_DIR, exist_ok=True)

app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(search.router, prefix="api/v1/search/", tags=["search"])


@app.get("/")
async def root():
    return {"message" : "Welcome to Semantic Search"}
