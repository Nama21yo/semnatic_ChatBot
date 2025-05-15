# app/core/context_manager.py
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory # Updated import
from langchain.schema import BaseMessage # For type hinting
from app.core.config import settings
from pathlib import Path
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# This global dict will store FileChatMessageHistory instances, not ConversationBufferMemory directly
# The key will be session_id + context_type (e.g., "semantic_search", "gemini_qa")
PERSISTENT_CHAT_HISTORIES: Dict[str, FileChatMessageHistory] = {}

def get_chat_history_manager(session_id: str, context_type: str = "default") -> FileChatMessageHistory:
    """
    Gets or creates a FileChatMessageHistory instance for a given session and context type.
    context_type can be 'semantic_search' or 'gemini_qa' etc.
    """
    history_key = f"{session_id}_{context_type}"
    if history_key not in PERSISTENT_CHAT_HISTORIES:
        history_file_path = Path(settings.CHAT_HISTORY_DIR) / f"{history_key}_messages.json"
        PERSISTENT_CHAT_HISTORIES[history_key] = FileChatMessageHistory(str(history_file_path))
        logger.info(f"Initialized FileChatMessageHistory for '{history_key}' at {history_file_path}")
    return PERSISTENT_CHAT_HISTORIES[history_key]

class SessionConversationManager:
    """
    Manages a specific conversation within a session, using FileChatMessageHistory for persistence
    and wrapping it with ConversationBufferMemory for Langchain chain compatibility.
    """
    def __init__(self, session_id: str, memory_context_type: str = "gemini_qa"):
        self.session_id = session_id
        self.memory_context_type = memory_context_type # e.g., "gemini_qa"
        
        # Get the persistent file-based history
        self.file_chat_history: FileChatMessageHistory = get_chat_history_manager(session_id, memory_context_type)
        
        # Wrap it with ConversationBufferMemory for use in chains
        # This memory will load from and save to the file_chat_history
        self.memory: ConversationBufferMemory = ConversationBufferMemory(
            chat_memory=self.file_chat_history,
            memory_key="chat_history", # Standard key for ConversationalRetrievalChain
            input_key="question",     # For ConversationalRetrievalChain
            output_key="answer",      # For ConversationalRetrievalChain
            return_messages=True      # Crucial for ChatPromptTemplate and MessagesPlaceholder
        )
        logger.debug(f"SessionConversationManager for '{memory_context_type}' in session '{session_id}' initialized.")

    def get_langchain_memory(self) -> ConversationBufferMemory:
        return self.memory

    def add_user_message(self, message: str):
        self.file_chat_history.add_user_message(message) # Also saves to file

    def add_ai_message(self, message: str):
        self.file_chat_history.add_ai_message(message) # Also saves to file
    
    def get_messages(self) -> List[BaseMessage]:
        return self.file_chat_history.messages

    def clear(self):
        self.file_chat_history.clear() # Clears messages and the file content
        logger.info(f"Cleared chat history for session '{self.session_id}', context '{self.memory_context_type}'.")


# Helper to clear all history files for a session when session data is deleted
def clear_all_histories_for_session(session_id: str):
    # Remove from in-memory cache
    keys_to_remove = [k for k in PERSISTENT_CHAT_HISTORIES if k.startswith(session_id)]
    for key in keys_to_remove:
        del PERSISTENT_CHAT_HISTORIES[key]

    # Delete physical files
    history_dir = Path(settings.CHAT_HISTORY_DIR)
    for f_path in history_dir.glob(f"{session_id}_*_messages.json"):
        try:
            f_path.unlink()
            logger.info(f"Deleted chat history file: {f_path}")
        except OSError as e:
            logger.error(f"Error deleting chat history file {f_path}: {e}")