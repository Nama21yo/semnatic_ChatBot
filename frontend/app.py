import streamlit as st
import requests # To communicate with FastAPI backend
import uuid
import json
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import os
# import time # time module was imported but not used, can be removed if still unused.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Context-Aware Semantic Search")

# --- Initialize session state variables ---
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    # For Semantic Search
    if "semantic_search_results" not in st.session_state:
        st.session_state.semantic_search_results = []
    if "semantic_search_query_value" not in st.session_state:
        st.session_state.semantic_search_query_value = ""
    if "semantic_search_history_log" not in st.session_state: # UI log for semantic search
        st.session_state.semantic_search_history_log = []
    
    # For Conversational QA (Gemini)
    if "gemini_qa_response_answer" not in st.session_state: # Store only the answer string
        st.session_state.gemini_qa_response_answer = ""
    if "gemini_qa_query_value" not in st.session_state:
        st.session_state.gemini_qa_query_value = ""
    if "gemini_qa_history_log" not in st.session_state: # UI log for Gemini QA
        st.session_state.gemini_qa_history_log = []

    # Common
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []
    # Remove feedback related session state if not used
    # if "feedback" not in st.session_state: st.session_state.feedback = {}
    # if "current_query_id_for_feedback" not in st.session_state: st.session_state.current_query_id_for_feedback = None



init_session_state()

st.title("📚 Context-Aware Semantic Search Platform")
st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.markdown("---")

# --- Sidebar: File Upload and Session Management ---
with st.sidebar:
    st.header("📄 Document Management")
    
    # File uploader uses its 'key' to manage its own state in st.session_state
    uploaded_files_from_widget = st.file_uploader(
        "Upload PDF, DOCX, or TXT files to your Knowledge Base",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader_widget" 
    )

    if uploaded_files_from_widget:
        if st.button("⚙️ Process Uploaded Files"):
            newly_uploaded_filenames = []
            with st.spinner("Processing files... This may take a moment per file."):
                for uploaded_file_obj in uploaded_files_from_widget:
                    files_payload = {"file": (uploaded_file_obj.name, uploaded_file_obj.getvalue(), uploaded_file_obj.type)}
                    try:
                        upload_url = f"{FASTAPI_URL}/api/v1/documents/upload?session_id={st.session_state.session_id}"
                        response = requests.post(upload_url, files=files_payload)
                        response.raise_for_status()
                        response_data = response.json()
                        if not any(info['filename'] == response_data["filename"] for info in st.session_state.uploaded_files_info):
                            st.session_state.uploaded_files_info.append({
                                "filename": response_data["filename"],
                                "doc_id": response_data["doc_id"]
                            })
                        newly_uploaded_filenames.append(response_data["filename"])
                        st.success(f"✅ '{uploaded_file_obj.name}' received for processing.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error uploading {uploaded_file_obj.name}: {e}")
                        if e.response is not None: st.error(f"Backend Error: {e.response.text}")
                    except Exception as e_gen:
                        st.error(f"Unexpected error with {uploaded_file_obj.name}: {e_gen}")
            if newly_uploaded_filenames:
                st.info(f"Files {', '.join(newly_uploaded_filenames)} sent for background processing. Indexing may take time.")
            # After processing, the button click itself will cause a rerun.
            # The file_uploader_widget will be re-rendered, and uploaded_files_from_widget will be
            # fresh for that new run (empty unless the user re-selects files).

    st.markdown("---")
    st.subheader("Current Knowledge Base:")
    if st.session_state.uploaded_files_info:
        for info in st.session_state.uploaded_files_info:
            st.markdown(f"- `{info['filename']}`")
    else:
        st.caption("No files processed for this session yet.")

    st.markdown("---")
    st.header("🔧 Session Tools")
    if st.button("🗑️ Clear All Data for This Session"):
        if st.session_state.session_id:
            try:
                delete_url = f"{FASTAPI_URL}/api/v1/documents/session_data/{st.session_state.session_id}"
                response = requests.delete(delete_url)
                response.raise_for_status()
                st.success(f"All data for session '{st.session_state.session_id}' cleared!")
                # Reset relevant session state variables
                st.session_state.uploaded_files_info = []
                st.session_state.semantic_search_results = []
                st.session_state.semantic_search_history_log = []
                st.session_state.gemini_qa_history_log = []
                st.session_state.gemini_qa_response_answer = ""
                st.session_state.semantic_search_query_value = ""
                st.session_state.gemini_qa_query_value = ""
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Error clearing session data: {e}")
                if e.response is not None: st.error(f"Backend Error: {e.response.text}")
            except Exception as e_gen:
                st.error(f"An unexpected error clearing session: {e_gen}")
        else:
            st.warning("No active session ID to clear.")

# --- Main Area ---
col_search, col_qa = st.columns(2) # Two main columns for the two blocks

# === Block 1: Semantic Search with Simple Follow-up Context ===
with col_search:
    st.header("🔍 Semantic Search")
    st.caption("Get top document chunks. Ask follow-up questions based on these results.")

    # Display Semantic Search History
    if st.session_state.semantic_search_history_log:
        st.write("Search History:")
        for entry in st.session_state.semantic_search_history_log:
            if entry["role"] == "user":
                st.markdown(f"🧑‍💻 **You (Search):** {entry['content']}")
            else:
                st.markdown(f"🤖 **System (Search Results):** {entry['content']}")
        st.markdown("---")

    semantic_query_input = st.text_input(
        "Enter your search query:",
        key="semantic_search_input_widget",
        value=st.session_state.semantic_search_query_value
    )

    if st.button("Search Documents", key="semantic_search_button"):
        if semantic_query_input:
            st.session_state.semantic_search_query_value = "" # Clear input for next run
            st.session_state.semantic_search_history_log.append({"role": "user", "content": semantic_query_input})
            
            payload = {
                "query": semantic_query_input,
                "session_id": st.session_state.session_id,
                "top_k": 3
            }
            with st.spinner("Searching documents..."):
                try:
                    response = requests.post(f"{FASTAPI_URL}/api/v1/search/semantic_search", json=payload) # Updated endpoint
                    response.raise_for_status()
                    search_data = response.json()
                    st.session_state.semantic_search_results = search_data.get("results", [])
                    
                    if st.session_state.semantic_search_results:
                        assistant_msg = (f"Found {len(st.session_state.semantic_search_results)} relevant chunks. "
                                         f"Top result from '{st.session_state.semantic_search_results[0]['chunk']['source_document']}'.")
                    else:
                        assistant_msg = "No relevant chunks found."
                    st.session_state.semantic_search_history_log.append({"role": "assistant", "content": assistant_msg})
                except Exception as e:
                    st.error(f"Semantic Search Error: {e}")
                    st.session_state.semantic_search_history_log.append({"role": "assistant", "content": f"Search failed: {e}"})
            st.rerun() # Rerun to display results and cleared input
        else:
            st.warning("Please enter a search query.")

    # Display Semantic Search Results (No relevance feedback)
    if st.session_state.semantic_search_results:
        st.subheader(f"Top {len(st.session_state.semantic_search_results)} Search Results:")
        for i, result_item in enumerate(st.session_state.semantic_search_results):
            chunk = result_item["chunk"]
            score = result_item["score"]
            with st.expander(f"**{i+1}. {chunk['source_document']}** (Page: {chunk.get('page_number', 'N/A') or 'N/A'}) - Score: {score:.4f}"):
                st.markdown(f"```text\n{chunk['text']}\n```")


# === Block 2: Conversational QA with Gemini ===
with col_qa:
    st.header("💬 Conversational QA (with Gemini)")
    st.caption("Ask questions, including follow-ups. Powered by Gemini and document context.")

    # Display Gemini QA History
    if st.session_state.gemini_qa_history_log:
        st.write("QA Conversation:")
        for entry in st.session_state.gemini_qa_history_log:
            if entry["role"] == "user":
                st.markdown(f"🧑‍💻 **You (QA):** {entry['content']}")
            else:
                st.markdown(f"🤖 **Gemini:** {entry['content']}")
        st.markdown("---")
    
    gemini_query_input = st.text_input(
        "Ask Gemini a question about your documents:",
        key="gemini_qa_input_widget",
        value=st.session_state.gemini_qa_query_value
    )

    if st.button("Ask Gemini", key="gemini_qa_button"):
        if gemini_query_input:
            st.session_state.gemini_qa_query_value = "" # Clear input for next run
            st.session_state.gemini_qa_history_log.append({"role": "user", "content": gemini_query_input})

            payload = {
                "question": gemini_query_input,
                "session_id": st.session_state.session_id
            }
            with st.spinner("Gemini is thinking..."):
                try:
                    response = requests.post(f"{FASTAPI_URL}/api/v1/search/conversational_qa", json=payload) # Updated endpoint
                    response.raise_for_status()
                    qa_data = response.json() # QAResponse model
                    answer = qa_data.get("answer", "No answer from Gemini.")
                    st.session_state.gemini_qa_response_answer = answer # Store for display if needed outside log
                    st.session_state.gemini_qa_history_log.append({"role": "assistant", "content": answer})
                    
                    # Optionally display source chunks if returned and desired
                    # source_chunks = qa_data.get("source_chunks", [])
                    # if source_chunks:
                    #     with st.expander("Sources considered by Gemini"):
                    #         for sc_idx, sc in enumerate(source_chunks):
                    #             st.caption(f"Source {sc_idx+1}: {sc['source_document']} (Pg: {sc.get('page_number','N/A')}) - {sc['text'][:100]}...")
                except Exception as e:
                    st.error(f"Gemini QA Error: {e}")
                    error_msg = f"QA failed: {e}"
                    if hasattr(e, 'response') and e.response is not None: error_msg += f" - Backend: {e.response.text}"
                    st.session_state.gemini_qa_history_log.append({"role": "assistant", "content": error_msg})
            st.rerun() # Rerun to display new history and cleared input
        else:
            st.warning("Please enter a question for Gemini.")

# --- Visualization Section (t-SNE/UMAP) ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Visualize Embeddings")
if st.sidebar.button("Show Embedding Space (t-SNE)"):
    if not st.session_state.uploaded_files_info:
        st.sidebar.warning("Upload and process documents first to visualize embeddings.")
    else:
        with st.spinner("Fetching and preparing data for visualization..."):
            try:
                viz_data_url = f"{FASTAPI_URL}/api/v1/documents/knowledge_base_data?session_id={st.session_state.session_id}&limit=500"
                response = requests.get(viz_data_url)
                response.raise_for_status()
                data = response.json()
                
                embeddings_list = data.get("embeddings", [])
                chunks_meta = data.get("chunks_metadata", [])

                if embeddings_list and chunks_meta and len(embeddings_list) == len(chunks_meta):
                    st.session_state.embeddings_for_viz = pd.DataFrame(embeddings_list)
                    hover_texts = [f"Doc: {cm.get('source_document', 'N/A')}\nPage: {cm.get('page_number', 'N/A')}\nText: {cm.get('text', '')[:100]}..." for cm in chunks_meta]
                    st.session_state.chunks_text_for_viz = hover_texts
                    st.sidebar.success(f"Loaded {len(embeddings_list)} chunk embeddings for visualization.")
                else:
                    st.sidebar.error("Could not load sufficient embedding data for visualization, or data mismatch.")
                    st.session_state.embeddings_for_viz = None
                    st.session_state.chunks_text_for_viz = None
            except requests.exceptions.RequestException as e_viz_req:
                st.sidebar.error(f"Error fetching data for viz: {e_viz_req}")
                if e_viz_req.response is not None: st.sidebar.error(f"Backend: {e_viz_req.response.text}")
                st.session_state.embeddings_for_viz = None
            except Exception as e_viz:
                st.sidebar.error(f"An unexpected error fetching data for viz: {e_viz}")
                st.session_state.embeddings_for_viz = None

if "embeddings_for_viz" in st.session_state and st.session_state.embeddings_for_viz is not None \
   and "chunks_text_for_viz" in st.session_state and st.session_state.chunks_text_for_viz is not None:
    embeddings_df = st.session_state.embeddings_for_viz
    if len(embeddings_df) > 1:
        perplexity_val = min(30, len(embeddings_df) - 1)
        if perplexity_val > 0 :
            with st.spinner("Running t-SNE... (can be slow for many points)"):
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300)
                try:
                    embeddings_2d = tsne.fit_transform(embeddings_df.values)
                    df_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
                    df_2d['hover_text'] = st.session_state.chunks_text_for_viz
                    fig = px.scatter(df_2d, x='x', y='y', hover_name='hover_text', title="t-SNE Visualization of Document Chunks")
                    fig.update_traces(marker=dict(size=8))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e_tsne:
                    st.error(f"Error during t-SNE computation or plotting: {e_tsne}")
        else:
            st.sidebar.warning("Not enough unique data points for t-SNE perplexity calculation (need > perplexity).")
    else:
        st.sidebar.warning("Not enough data points for t-SNE visualization (need at least 2).")