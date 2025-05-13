from app.core.search_engine import search_indexed_embeddings
from app.core.nlp_utils import ner_tag_query
from app.core.context_manager import get_conversation_history, add_to_conversation_history
from app.api.v1.schemas import QueryRequest, SearchResponse, SearchResultItem, Chunk
from typing import List, Optional
import uuid

class SearchService:
    def __init__(self):
        # Potentially load models or clients here if not global
        pass

    def perform_search(self, request: QueryRequest) -> SearchResponse:
        session_id = request.session_id
        current_query = request.query
        
        # Add current user query to history
        add_to_conversation_history(session_id, "user", current_query)
        
        # --- Advanced Context Handling (Illustrative) ---
        # This is where true multi-turn logic would go.
        # Example: If the query is "How old is he?"
        # 1. Identify "he" as a pronoun.
        # 2. Look at `get_conversation_history(session_id)`.
        # 3. Find the most recent male entity (e.g., "Joe Biden") from previous turns.
        # 4. Reformulate query: "How old is Joe Biden?"
        # This often requires a more sophisticated NLP model or an LLM.
        
        # For this project, let's use NER on the current query.
        # The "context" is primarily searching within the user's uploaded docs (via session_id).
        # The "multi-turn" could be: if a query is very short or pronoun-heavy,
        # concatenate it with the *text of the previously retrieved top chunk*.
        # This is a common retrieval augmentation technique.

        history = get_conversation_history(session_id)
        effective_query = current_query

        # Simple conversational enhancement:
        # If current query is short and seems like a follow-up
        if len(current_query.split()) < 4 and any(pronoun in current_query.lower() for pronoun in ["he", "she", "it", "they", "his", "her", "its", "their", "him"]):
            # Find the last user query or assistant response that might provide context
            for i in range(len(history) - 2, -1, -1): # Iterate backwards, skipping current user query
                if history[i]["role"] == "user" or (history[i]["role"] == "assistant" and "retrieved context" in history[i]):
                    # A simple strategy: prepend previous relevant text
                    # A better strategy would be to use the *retrieved chunk* from the last turn.
                    # For now, let's just prepend the last user query for simplicity if it's a follow-up.
                    # This is highly heuristic.
                    if history[i]["role"] == "user":
                        effective_query = history[i]["content"] + ". " + current_query
                        print(f"Contextualized query: {effective_query}")
                        break
        
        # NER tagging (can be used for filtering or boosting, not fully implemented here)
        entities = ner_tag_query(effective_query)
        print(f"Query: '{effective_query}', Entities: {entities}")

        # Perform search
        search_results_raw = search_indexed_embeddings(
            query_text=effective_query,
            session_id=session_id,
            top_k=3 # As per requirement
        )

        # Format results
        results_items = []
        for res in search_results_raw:
            # res['chunk'] is already {'text': ..., 'page_number': ..., 'source_document': ...}
            chunk_obj = Chunk(
                text=res["chunk"]["text"],
                page_number=res["chunk"].get("page_number"),
                source_document=res["chunk"]["source_document"]
            )
            results_items.append(SearchResultItem(chunk=chunk_obj, score=res["score"]))
        
        query_id = str(uuid.uuid4()) # Unique ID for this specific query-response interaction

        # Add system response summary to history (e.g., top result text)
        # This helps for future context.
        if results_items:
            summary = f"Found results for '{current_query}'. Top result from '{results_items[0].chunk.source_document}' pg {results_items[0].chunk.page_number}: {results_items[0].chunk.text[:100]}..."
            add_to_conversation_history(session_id, "assistant", summary) # Store "retrieved context" potentially
        else:
            add_to_conversation_history(session_id, "assistant", f"No results found for '{current_query}'.")


        return SearchResponse(results=results_items, query_id=query_id)

    # (Optional) QA method for bonus
    def perform_qa(self, session_id: str, question: str) -> Optional[str]:
        # 1. Retrieve relevant chunks (reuse perform_search or search_indexed_embeddings)
        relevant_chunks_data = search_indexed_embeddings(
            query_text=question,
            session_id=session_id,
            top_k=1 # Get the most relevant chunk for QA
        )
        if not relevant_chunks_data:
            return None

        context_text = relevant_chunks_data[0]["chunk"]["text"]

        # 2. Use a QA model
        # Placeholder for Hugging Face QA pipeline
        try:
            from transformers import pipeline
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")
            qa_result = qa_pipeline(question=question, context=context_text)
            return qa_result["answer"]
        except Exception as e:
            print(f"QA model error: {e}")
            return f"Could not extract answer. Most relevant text: {context_text[:200]}..."
