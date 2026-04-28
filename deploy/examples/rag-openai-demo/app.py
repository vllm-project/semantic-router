#!/usr/bin/env python3
"""
Streamlit Web App for OpenAI RAG Demo

Interactive web interface for demonstrating OpenAI RAG functionality.

Usage:
    streamlit run app.py
"""

import os
import time
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="OpenAI RAG Demo", page_icon="🔍", layout="wide")


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        st.error("OPENAI_API_KEY not set in environment variables!")
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def main():
    st.title("🔍 OpenAI RAG Demo")
    st.markdown(
        "Demonstrate Retrieval-Augmented Generation with OpenAI File Store and Vector Store APIs"
    )

    client = get_openai_client()
    if not client:
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        semantic_router_url = st.text_input(
            "Semantic Router URL",
            value=os.getenv("SEMANTIC_ROUTER_URL", "http://localhost:8080"),
        )
        workflow_mode = st.selectbox(
            "Workflow Mode",
            ["direct_search", "tool_based"],
            help="direct_search: Synchronous retrieval\n\ntool_based: LLM-controlled retrieval",
        )

        # Optional: Explicit decision selection (for testing)
        # In production, router automatically selects decisions based on signals
        use_explicit_decision = st.checkbox(
            "Use explicit decision header",
            value=False,
            help="If enabled, uses X-VSR-Selected-Decision header. "
            "Otherwise, router automatically matches decisions based on signals.",
        )
        explicit_decision_name = st.text_input(
            "Decision Name",
            value="rag-openai-decision",
            disabled=not use_explicit_decision,
        )

        st.divider()
        st.header("Vector Store")

        # List existing vector stores
        if st.button("Refresh Vector Stores"):
            st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 File Management", "🗂️ Vector Store", "💬 Chat with RAG", "📊 Comparison"]
    )

    # Tab 1: File Management
    with tab1:
        st.header("File Store Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload File")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "txt", "md"],
                help="Upload a file to OpenAI File Store",
            )

            if uploaded_file and st.button("Upload"):
                with st.spinner("Uploading file..."):
                    try:
                        file_obj = client.files.create(
                            file=uploaded_file, purpose="assistants"
                        )
                        st.success(f"File uploaded: {file_obj.id}")
                        st.json(
                            {
                                "id": file_obj.id,
                                "filename": file_obj.filename,
                                "bytes": file_obj.bytes,
                                "purpose": file_obj.purpose,
                            }
                        )
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        with col2:
            st.subheader("List Files")
            if st.button("Refresh Files"):
                try:
                    files = client.files.list(purpose="assistants")
                    if files.data:
                        st.dataframe(
                            [
                                {
                                    "ID": f.id,
                                    "Filename": f.filename,
                                    "Bytes": f.bytes,
                                    "Created": time.strftime(
                                        "%Y-%m-%d %H:%M:%S",
                                        time.localtime(f.created_at),
                                    ),
                                }
                                for f in files.data
                            ]
                        )
                    else:
                        st.info("No files found")
                except Exception as e:
                    st.error(f"Failed to list files: {e}")

    # Tab 2: Vector Store
    with tab2:
        st.header("Vector Store Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Create Vector Store")
            vector_store_name = st.text_input(
                "Vector Store Name", value="Demo Vector Store"
            )

            # File selection
            try:
                files = client.files.list(purpose="assistants")
                file_options = {f"{f.filename} ({f.id})": f.id for f in files.data}
                selected_files = st.multiselect(
                    "Select files to attach",
                    options=list(file_options.keys()),
                    help="Select files to attach to the vector store",
                )
                file_ids = [file_options[f] for f in selected_files]
            except:
                file_ids = []
                st.warning("Could not load files")

            if st.button("Create Vector Store"):
                with st.spinner("Creating vector store..."):
                    try:
                        vs = client.beta.vector_stores.create(
                            name=vector_store_name, file_ids=file_ids
                        )
                        st.success(f"Vector store created: {vs.id}")
                        st.session_state["vector_store_id"] = vs.id
                        st.json(
                            {
                                "id": vs.id,
                                "name": vs.name,
                                "file_counts": {
                                    "total": vs.file_counts.total,
                                    "completed": vs.file_counts.completed,
                                    "in_progress": vs.file_counts.in_progress,
                                },
                            }
                        )
                    except Exception as e:
                        st.error(f"Failed to create vector store: {e}")

        with col2:
            st.subheader("Vector Store Status")
            if "vector_store_id" in st.session_state:
                vs_id = st.session_state["vector_store_id"]
                try:
                    vs = client.beta.vector_stores.retrieve(vs_id)
                    st.json(
                        {
                            "id": vs.id,
                            "name": vs.name,
                            "file_counts": {
                                "total": vs.file_counts.total,
                                "completed": vs.file_counts.completed,
                                "in_progress": vs.file_counts.in_progress,
                                "failed": vs.file_counts.failed,
                            },
                        }
                    )
                except Exception as e:
                    st.error(f"Failed to retrieve vector store: {e}")
            else:
                st.info("No vector store selected. Create one in the left column.")

    # Tab 3: Chat with RAG
    with tab3:
        st.header("Chat with RAG")

        if "vector_store_id" not in st.session_state:
            st.warning("Please create a vector store first in the 'Vector Store' tab.")
        else:
            vs_id = st.session_state["vector_store_id"]

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            import requests

                            # Use Responses API with file_search tool
                            # Router automatically selects decision based on signals (keywords, etc.)
                            # No need for X-VSR-Selected-Decision header in production
                            vs_id = st.session_state["vector_store_id"]
                            request_body = {
                                "model": "gpt-4o-mini",
                                "input": prompt,
                                "tools": [
                                    {
                                        "type": "file_search",
                                        "vector_store_ids": [vs_id],
                                    }
                                ],
                            }

                            # Build headers - only include decision header if explicitly requested
                            headers = {"Content-Type": "application/json"}
                            if use_explicit_decision and explicit_decision_name:
                                headers["X-VSR-Selected-Decision"] = explicit_decision_name

                            response = requests.post(
                                f"{semantic_router_url}/v1/responses",
                                headers=headers,
                                json=request_body,
                                timeout=60,
                            )
                            response.raise_for_status()
                            result = response.json()

                            # Extract response content from Responses API format
                            # Response API returns output array with content items
                            if "output" in result and len(result["output"]) > 0:
                                output_item = result["output"][0]
                                if "content" in output_item:
                                    content = output_item["content"]
                                    if isinstance(content, list) and len(content) > 0:
                                        answer = content[0].get("text", "")
                                    else:
                                        answer = str(content)
                                else:
                                    answer = str(output_item)
                            else:
                                answer = "No response content found"

                            st.markdown(answer)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer}
                            )
                        except Exception as e:
                            error_msg = f"Error: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_msg}
                            )

    # Tab 4: Comparison
    with tab4:
        st.header("Response Comparison")
        st.markdown("Compare responses with and without RAG context")

        comparison_query = st.text_input(
            "Enter a query to compare", value="What is the Semantic Router?"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Without RAG (Baseline)")
            if st.button("Get Baseline Response"):
                with st.spinner("Getting baseline response..."):
                    try:
                        import requests

                        response = requests.post(
                            f"{semantic_router_url}/v1/chat/completions",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": "gpt-4o-mini",
                                "messages": [
                                    {"role": "user", "content": comparison_query}
                                ],
                            },
                            timeout=60,
                        )
                        response.raise_for_status()
                        result = response.json()
                        st.markdown(result["choices"][0]["message"]["content"])
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            st.subheader("With RAG (Responses API)")
            if st.button("Get RAG Response") and "vector_store_id" in st.session_state:
                with st.spinner("Getting RAG response..."):
                    try:
                        import requests

                        vs_id = st.session_state["vector_store_id"]
                        # Use Responses API with file_search tool
                        # Router automatically selects decision based on signals
                        headers = {"Content-Type": "application/json"}
                        if use_explicit_decision and explicit_decision_name:
                            headers["X-VSR-Selected-Decision"] = explicit_decision_name

                        response = requests.post(
                            f"{semantic_router_url}/v1/responses",
                            headers=headers,
                            json={
                                "model": "gpt-4o-mini",
                                "input": comparison_query,
                                "tools": [
                                    {
                                        "type": "file_search",
                                        "vector_store_ids": [vs_id],
                                    }
                                ],
                            },
                            timeout=60,
                        )
                        response.raise_for_status()
                        result = response.json()

                        # Extract response content from Responses API format
                        if "output" in result and len(result["output"]) > 0:
                            output_item = result["output"][0]
                            if "content" in output_item:
                                content = output_item["content"]
                                if isinstance(content, list) and len(content) > 0:
                                    answer = content[0].get("text", "")
                                else:
                                    answer = str(content)
                            else:
                                answer = str(output_item)
                        else:
                            answer = "No response content found"

                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error: {e}")
            elif "vector_store_id" not in st.session_state:
                st.warning("Create a vector store first")


if __name__ == "__main__":
    main()
