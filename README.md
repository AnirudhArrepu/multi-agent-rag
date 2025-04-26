# multi-agent-rag

### Objective:
Building a lightweight AI-powered assistant that can:
1. Answer questions using uploaded documents (via RAG).
2. Act autonomously using tools based on the user’s intent (Agentic AI).
3. Should suport multiple chat interfaces each with they own PDF/txt documents.

### Agents
- "rag_qa" for the RAG Answer function  
- "summarize" for the Summarize function  
- "kpi" for the KPI Extraction function  
- "report" for the Report Generation function  
- "web" for the Web Search function

#### Multi Agent workflow
`agent.py` contains an implementation of crewAI for multi agent workflow. However, in `main.py` a Mistral model is given a detailed description of the agents and the query, which then outputs the agent to use. 

### RAG Pipeline
DocHelper class is used to extract the text from documents
SplitEmbedDB class is used to store/ retrieve embeddings (which are generated from the text using a sentence transformer), each of the document stored in chromaDB is also asociated with a chat_id value which corresponds to the chat instance having the related PDFs
rag_qa agent generates a prompt with the context retrieved from chromaDB based on chat_id and query (similarity) and the query. this prompt is sent to a mistral LLM for answer.

  ### Technical Details
  Streamlit (UI)
  Mistral (LLM)
  crewAI for multi agent workflow
  tavilly api for web search results
  PyPDF2 for text extraction from pdf
  chromaDB for vector embeddings DB
  all-MiniLM-L6-v2 for generating vector embeddigns from text
  


  ### Local Run
  ```bash
  python -m venv env
  pip install -r requirements.txt
  streamlit run main.py
  ```
