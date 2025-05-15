from app.core.vector_store import ChromaDBManager
from app.core.nlp_utils import ner_tag_query
from app.core.config import settings
from app.api.v1.schemas import (
    QueryRequest, SearchResponse, SearchResultItem, Chunk,
    QAQueryRequest, QAResponse
)
from app.core.context_manager import SessionConversationManager, get_chat_history_manager

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import logging
import uuid
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager
        self.langchain_embeddings = self.chroma_manager.embedding_function_for_langchain

        self.gemini_llm: Optional[ChatGoogleGenerativeAI] = None
        if settings.GOOGLE_API_KEY:
            try:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL_NAME,
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=0.1
                )
                logger.info(f"Initialized Gemini LLM: {settings.GEMINI_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Gemini init failed: {e}", exc_info=True)
        else:
            logger.warning("GOOGLE_API_KEY not found. Gemini will not work.")

    async def _rephrase_query_with_history(self, query: str, chat_history_messages) -> str:
        if not self.gemini_llm: return query
        if not chat_history_messages: return query

        history_str = "\n".join(
            [f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" for msg in chat_history_messages]
        )
        REPHRASE_TEMPLATE = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that incorporates relevant context from the history.
If the follow-up question is already standalone, return it as is.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question:"""

        prompt = PromptTemplate(
            template=REPHRASE_TEMPLATE,
            input_variables=["chat_history", "question"]
        )
        rephrase_chain = LLMChain(llm=self.gemini_llm, prompt=prompt, verbose=True)
        try:
            response = await rephrase_chain.acall({"chat_history": history_str, "question": query})
            rephrased = response.get("text", query).strip()
            if not rephrased:
                logger.warning(f"Query rephrasing returned empty for '{query}'. Using original.")
                return query
            return rephrased
        except Exception as e:
            logger.error(f"Query rephrasing error: {e}", exc_info=True)
            return query

    async def perform_semantic_search(self, request: QueryRequest) -> SearchResponse:
        session_id = request.session_id
        original_query = request.query

        semantic_search_history_mgr = get_chat_history_manager(session_id, context_type="semantic_search")
        rephrased_query = await self._rephrase_query_with_history(original_query, semantic_search_history_mgr.messages)
        semantic_search_history_mgr.add_user_message(original_query)

        logger.info(f"Semantic Search - Session '{session_id}': Original='{original_query}', Rephrased='{rephrased_query}'")

        # Use ChromaDBManager's query method (uses persistent client, collection, and global embeddings)
        search_results_raw = self.chroma_manager.query_documents_with_langchain_chroma(
            session_id=session_id,
            query_text=rephrased_query,
            top_k=request.top_k if request.top_k is not None else 3
        )

        results_items: List[SearchResultItem] = []
        assistant_response_summary = "No relevant chunks found."
        if search_results_raw:
            for res_dict in search_results_raw:
                chunk_obj = Chunk(**res_dict["chunk"])
                results_items.append(SearchResultItem(chunk=chunk_obj, score=res_dict["score"], id=str(res_dict.get("id"))))
            if results_items:
                top_result_summary = (
                    f"Top result for '{original_query}' from '{results_items[0].chunk.source_document}'.")
                assistant_response_summary = f"Found {len(results_items)} chunks. {top_result_summary}"
        else:
            logger.info(f"Semantic search for '{rephrased_query}' (original: '{original_query}') returned no results from ChromaDBManager for session {session_id}.")

        semantic_search_history_mgr.add_ai_message(assistant_response_summary)
        return SearchResponse(results=results_items, query_id=str(uuid.uuid4()), session_id=session_id)

    def _get_custom_chroma_retriever_with_filter(self, session_id: str):
        if not self.langchain_embeddings:
            logger.error("Langchain embeddings (from ChromaManager) not initialized for custom retriever.")
            return None

        # Get the base retriever from ChromaDBManager
        base_retriever = self.chroma_manager.get_langchain_chroma_retriever(
            session_id=session_id,
            search_type="similarity",
            search_kwargs={"k": getattr(settings, "INITIAL_RETRIEVAL_K", 5)}
        )
        if not base_retriever:
            logger.warning(f"Base Chroma retriever could not be created for session {session_id} via ChromaDBManager.")
            return None

        from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        redundant_filter = EmbeddingsFilter(
            embeddings=self.langchain_embeddings,
            similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD_REDUNDANT_FILTER", 0.8)
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=redundant_filter,
            base_retriever=base_retriever
        )
        logger.info(f"Custom Chroma retriever with filter created for session {session_id}.")
        return compression_retriever

    async def perform_conversational_qa(self, request: QAQueryRequest) -> QAResponse:
        session_id = request.session_id
        question = request.question

        if not self.gemini_llm:
            return QAResponse(question=question, answer="Gemini LLM is not configured.", source_chunks=[], session_id=session_id)

        qa_history_mgr = get_chat_history_manager(session_id, context_type="qa")
        chat_history = [
            (msg.content, "") for msg in qa_history_mgr.messages if msg.type == "human"
        ]

        custom_retriever = self._get_custom_chroma_retriever_with_filter(session_id)
        if not custom_retriever:
            return QAResponse(question=question, answer="Knowledge base retriever could not be initialized.", source_chunks=[], session_id=session_id)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.gemini_llm,
            retriever=custom_retriever,
            return_source_documents=True,
            chain_type="stuff",
        )
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: qa_chain({"question": question, "chat_history": chat_history})
            )
            final_answer = result.get("answer", "No answer from LLM.")
            source_documents_from_chain = result.get("source_documents", [])
            source_chunks_for_response = [
                Chunk(
                    text=doc.page_content,
                    page_number=doc.metadata.get("page_number"),
                    source_document=doc.metadata.get("source_document", "Unknown")
                )
                for doc in source_documents_from_chain
            ]
        except Exception as e:
            logger.error(f"Error in conversational QA chain: {e}", exc_info=True)
            final_answer = "Error during QA processing."
            source_chunks_for_response = []

        qa_history_mgr.add_user_message(question)
        qa_history_mgr.add_ai_message(final_answer)

        return QAResponse(
            question=question,
            answer=final_answer,
            source_chunks=source_chunks_for_response,
            session_id=session_id
        )