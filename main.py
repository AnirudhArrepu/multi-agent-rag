import streamlit as st
import uuid
import os

from utils.pdfhelper import DocHelper
from utils.text_processing import SplitEmbedDB
from dotenv import load_dotenv
from mistralai import Mistral
import requests

from utils.agents import (  # assume you moved agent functions here
    rag_qa_pipeline,
    summarise_agent,
    kpi_agent,
    report_agent,
    search_web,
    query_mistral_with_prompt
)

# Load environment variables
load_dotenv()
spemdb = SplitEmbedDB()

st.set_page_config(page_title="Multi-Agent Chat with PDF", page_icon="🤖", layout="wide")

# Initialize chats in session state
if "chats" not in st.session_state:
    st.session_state.chats = {}  # {chat_id: {name, messages, pdf, embeddings_processed}}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Sidebar: Create new chat
st.sidebar.title("🤖 Your Chats")

if st.sidebar.button("➕ New Chat"):
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = {
        "name": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "pdf": None,
        "embeddings_processed": False  # Flag to track if embeddings have been processed
    }
    st.session_state.current_chat_id = new_chat_id

# Sidebar: List existing chats with editable names
if st.session_state.chats:
    for chat_id, chat in st.session_state.chats.items():
        col1, col2 = st.sidebar.columns([4, 1])
        if col1.text_input("Chat name", value=chat["name"], key=f"name_{chat_id}"):
            st.session_state.chats[chat_id]["name"] = st.session_state[f"name_{chat_id}"]
        if col2.button("🗨️", key=f"select_{chat_id}"):
            st.session_state.current_chat_id = chat_id

# Main area: Show current chat
if st.session_state.current_chat_id:
    chat_id = st.session_state.current_chat_id
    chat_data = st.session_state.chats[chat_id]

    st.header(f"{chat_data['name']}")
    st.caption(f"Chat ID: `{chat_id}`")

    # Upload PDF
    uploaded_file = st.file_uploader("📄 Upload a PDF for this chat", type=["pdf"], key=f"pdf_{chat_id}")
    if uploaded_file is not None:
        chat_data["pdf"] = uploaded_file
        st.success("PDF uploaded!")

        # Only process embeddings if they haven't been processed already
        if not chat_data["embeddings_processed"]:
            dh = DocHelper(uploaded_file)
            text = dh.extract_text_from_doc()
            spemdb.store_embeddings(text, chat_id)
            chat_data["embeddings_processed"] = True
            st.success("PDF processed and embeddings stored!")
        else:
            st.info("Embeddings have already been processed for this document.")

    # Show previous messages
    for message in chat_data["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        chat_data["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Agents are thinking..."):

            #asking mistral which agent to choose (langchain would be an overkill in this case as the agents are independent of each other)
            mistral_prompt = f"""
The following are different functions available to respond to user queries:

1. **RAG (Retrieve and Generate) Answer:**  
   This function searches through the PDF content to find relevant information based on the query. It uses retrieval-augmented generation (RAG) to provide answers from the document. This function is useful when the user asks specific questions related to the content of the uploaded PDF.

2. **Summarize:**  
   This function generates a summary of the PDF content. It is used when the user wants a brief overview of the entire document, extracting key points and summarizing them.

3. **KPI (Key Performance Indicators) Extraction:**  
   This function extracts and identifies key performance indicators (KPIs) from the PDF. It is ideal when the user is looking for specific data points, metrics, or performance-related information from the document.

4. **Report Generation:**  
   This function generates a report based on the content of the PDF. It can be used when the user needs a detailed summary or a structured report derived from the document's data.

5. **Web Search:**  
   This function allows searching the web for answers, useful when the information needed is not found in the uploaded PDF. It performs a web search and retrieves relevant information from the internet.

Now, based on the user's input below, decide which function is the most appropriate to call:

User Query: {prompt}

Please choose one of the following agents based on the query:  
- "rag_qa" for the RAG Answer function  
- "summarize" for the Summarize function  
- "kpi" for the KPI Extraction function  
- "report" for the Report Generation function  
- "web" for the Web Search function

Output only the agent name (e.g., "rag_qa", "summarize", "kpi", "report", "web"). Do not include any other information or explanations.
"""

            mistral_response = query_mistral_with_prompt(mistral_prompt)

            print(mistral_response, flush=True)

            if "rag_qa" in mistral_response.lower():
                agent_response = rag_qa_pipeline(prompt, chat_id, spemdb)
            elif "summarize" in mistral_response.lower():
                agent_response = summarise_agent(prompt, chat_id, spemdb)
            elif "kpi" in mistral_response.lower():
                agent_response = kpi_agent(prompt, chat_id, spemdb)
            elif "report" in mistral_response.lower():
                agent_response = report_agent(prompt, chat_id, spemdb)
            elif "web" in mistral_response.lower():
                agent_response = search_web(prompt)
            else:
                agent_response = "I'm not sure which agent to use. Could you clarify?"

            chat_data["messages"].append({"role": "assistant", "content": agent_response})

        with st.chat_message("assistant"):
            st.markdown(agent_response)

else:
    st.info("Create or select a chat to get started.")
