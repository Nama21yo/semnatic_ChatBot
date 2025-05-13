#! resolve this error
from sentence_transformers import SentenceTrnasformer
from app.core.config import settings
import numpy as np
from typing import List

model = SentenceTrnasformer(settings.MODEL_NAME)

def generate_embeddings_for_chunks(chunks_text : List[str]):
    embeddings = model.encode(chunks_text, convert_to_tensor=False)

    return embeddings.astype("float32")

