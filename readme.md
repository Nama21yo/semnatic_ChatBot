# Context-Aware Semantic Search & QA Platform

This project implements a full-stack semantic search engine and a conversational question-answering (QA) system. Users can upload their documents, perform semantic searches to find relevant chunks, and engage in conversational QA powered by Google's Gemini Flash 2.0 LLM, all with an awareness of the ongoing conversation context.

### Video DEMO LInk - https://www.awesomescreenshot.com/video/39855808?key=95db5f8c828a6aa91a409293fe0fce6c

**Core Features:**

*   **Document Upload:** Supports PDF, DOCX, and TXT file uploads.
*   **Semantic Search:**
    *   Utilizes Sentence Transformers (`all-mpnet-base-v2`) for generating text embeddings.
    *   Stores and retrieves embeddings using ChromaDB for efficient similarity search.
    *   Returns top 3 contextually relevant document chunks with source and page number references.
    *   **Context-Aware Follow-ups:** Employs Gemini to rephrase follow-up queries in the semantic search block based on its dedicated conversation history, leading to more relevant search results.
*   **Conversational QA (with Gemini):**
    *   A dedicated interface for asking questions about the uploaded documents.
    *   Powered by Google's Gemini Pro (`gemini-pro`) LLM.
    *   Uses Langchain's `ConversationalRetrievalChain` with `ConversationBufferMemory` (backed by `FileChatMessageHistory`) to maintain multi-turn conversational context.
    *   Retrieves relevant document chunks from ChromaDB as context for Gemini.
    *   Includes an `EmbeddingsRedundantFilter` to de-duplicate and refine retrieved context before passing it to the LLM.
*   **Advanced NLP:**
    *   Named Entity Recognition (NER) tagging of queries (using spaCy).
*   **Persistent Storage:**
    *   ChromaDB for vector embeddings and metadata.
    *   File-based chat history (`FileChatMessageHistory`) for both semantic search and Gemini QA interactions, allowing context to persist across sessions (for the same `session_id`).
*   **Full-Stack Implementation:**
    *   **Backend:** FastAPI (Python)
    *   **Frontend:** Streamlit (Python)
*   **Visualization:** t-SNE visualization of document chunk embeddings.

---

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **Frontend:** Streamlit
*   **Vector Embeddings:** Sentence Transformers (`sentence-transformers/all-mpnet-base-v2`)
*   **Vector Database:** ChromaDB
*   **LLM for QA & Rephrasing:** Google Gemini Pro (via `langchain-google-genai`)
*   **Orchestration & Context Management:** Langchain
*   **NLP Utilities:** spaCy (for NER)
*   **File Parsing:** PyPDF2, python-docx

---


## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository:**

```bash
git clone <your-repository-url>
cd semantic_search_project
```

2. Create and Activate a Virtual Environment:

It's highly recommended to use a virtual environment. We'll name it .semantic_project as requested.

```bash
python3 -m venv .semantic_project
source .semantic_project/bin/activate
```

(On Windows, use: python -m venv .semantic_project followed by .semantic_project\Scripts\activate)

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

4. Download spaCy NLP Model:

The project uses spaCy for Named Entity Recognition.
```bash
python -m spacy download en_core_web_sm
```

5. Set Up Environment Variables (.env file):

You need to provide API keys and potentially other configurations. Create a file named .env in the root of the semantic_search_project directory.

Copy the contents of .env.example (if provided) or create .env with the following content:

# Google Gemini API Key
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"

Obtaining GOOGLE_API_KEY:

<ol>
<li>Go to Google AI Studio.</li>
<li>Sign in with your Google account.</li>
<li>Click on "Get API key" and create a new API key.</li>
</ol>

Important: Ensure your project associated with the API key has the "Generative Language API" enabled and billing set up if you exceed free tier limits.

6. Data Directories:

The application will automatically create the following directories inside the data/ folder if they don't exist: uploads/, chroma_db/, chat_histories/. Ensure your .gitignore file is set up to ignore data/chroma_db/ and data/chat_histories/ to avoid committing user data and large database files.

Running the Application

You need to run the FastAPI backend and the Streamlit frontend separately, typically in two different terminal windows. Make sure your virtual environment (.semantic_project) is activated in both terminals.

1. Start the FastAPI Backend:

Navigate to the project root directory (semantic_search_project/) in your terminal:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output indicating the Uvicorn server is running.

2. Start the Streamlit Frontend:

Open a new terminal window, navigate to the project root directory, and activate the virtual environment:
```bash
source .semantic_project/bin/activate # If not already active in this terminal
streamlit run frontend/app.py
```
Streamlit will typically open the application automatically in your default web browser (usually at http://localhost:8501). If not, the terminal output will provide the URL.

Usage Instructions

Once both the backend and frontend are running:

Open the Streamlit UI: Navigate to http://localhost:8501 (or the URL provided by Streamlit).

![Image](https://github.com/user-attachments/assets/67fd8a63-f8a7-4e13-8852-a94df0332401)
Description: The main interface showing the document upload sidebar, and the two main interaction blocks: "Semantic Search" and "Conversational QA (with Gemini)".

Upload Documents:

Use the sidebar on the left to upload your knowledge base.

Supported formats: PDF, DOCX, TXT. You can upload multiple files.

Click the "Process Uploaded Files" button.

The backend will process these files in the background (parsing, chunking, embedding, and indexing into ChromaDB). This may take some time depending on the size and number of files.

A list of processed files will appear in the sidebar.

Semantic Search Block:

Enter your query into the "Enter your search query:" input field in the "Semantic Search" block.

Click "Search Documents".

The system will display the top 3 most relevant chunks from your uploaded documents. Each result shows the source document, page number (if available), the text chunk, and a similarity score.

Follow-up Questions: If you ask a follow-up question related to your previous search query in this block, the system will use Gemini to rephrase your follow-up into a standalone question based on the semantic search block's specific conversation history before searching.

The interaction history for this block is displayed above the input field.

![Image](https://github.com/user-attachments/assets/06e632ab-bf25-4276-ae6c-01a4b3e4d998)
Description: Shows the semantic search input, a sample query, and the displayed top 3 results with expanders for chunk text.

Conversational QA (Gemini) Block:

Enter your question about the uploaded documents into the "Ask Gemini a question about your documents:" input field in the "💬 Conversational QA (with Gemini)" block.

Click "Ask Gemini".

Gemini, using the context retrieved from your documents (after redundancy filtering) and the conversation history within this QA block, will generate a natural language answer.

Follow-up Questions: This block is designed for multi-turn conversations. Gemini will remember previous questions and answers within this specific QA chat to understand context for follow-ups like "Tell me more about that." or "What was its impact?".

The conversation history for this block is displayed above its input field.

![Image](https://github.com/user-attachments/assets/e544d255-3adc-4c6c-ae12-a5c7a81ecb79)
Description: Shows the Gemini QA input, a sample question, and a conversational answer from Gemini, potentially with source document snippets if implemented.


Session Management (Sidebar):

Current Knowledge Base: Lists files successfully processed for the current session.

Clear All Data for This Session: This button will delete the ChromaDB vector store and all chat histories associated with the current session_id. Use this if you want to start fresh with a new knowledge base for the session.

Embedding Visualization (Sidebar):

After processing files, click "Show Embedding Space (t-SNE)" in the sidebar to generate and view a 2D t-SNE visualization of your document chunks. This helps understand the semantic relationships between different parts of your knowledge base.

📄 API Documentation

The FastAPI backend provides automatic API documentation:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

These interfaces allow you to explore and interact with the available API endpoints (e.g., for document upload, search, QA).
