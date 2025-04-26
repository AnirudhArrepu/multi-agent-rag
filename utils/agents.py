from .pdfhelper import DocHelper
from .text_processing import SplitEmbedDB
import os
import requests
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
tavilly_api_key = os.environ["TAVILLY_API_KEY"]
mistral_api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=mistral_api_key)

# dh = DocHelper('./Assignment.pdf')
# text = dh.extract_text_from_doc()

# spemdb = SplitEmbedDB()
# spemdb.store_embeddings(text, 'test_project')


def relevant_docs(query: str, project_id, spemdb: SplitEmbedDB):
    context = spemdb.query_project(project_id, query)

    con = ""
    con = con.join(context[0])
    return con

def query_mistral_with_prompt(prompt: str):
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content

def rag_qa_pipeline(query: str, project_id, spemdb: SplitEmbedDB):
    context = relevant_docs(query, project_id, spemdb)
    print("context", context)
    prompt = f""" Based on the context given, answer the question. If the answer is not found in context, say "I son't know".
    Context: {context}
    Question: {query}
    Answer:
    """
    return query_mistral_with_prompt(prompt)


def summarise_agent(query: str, project_id, spemdb: SplitEmbedDB):
    context = relevant_docs(query, project_id, spemdb)
    prompt = f"""Given the context, summarise the text in a few lines. Do not include any other information.
    Context: {context}
    Answer:
    """
    return query_mistral_with_prompt(prompt)

def kpi_agent(query: str, project_id, spemdb: SplitEmbedDB):
    context = relevant_docs(query, project_id, spemdb)
    prompt = f"""Given the context, extract the KPIs from the text. Do not include any other information.
    If no KPIs are found, say "No KPIs found".
    Context: {context}
    Answer:
    """
    return query_mistral_with_prompt(prompt)

def report_agent(query: str, project_id, spemdb: SplitEmbedDB):
    context = relevant_docs(query, project_id, spemdb)
    prompt = f"""Based on the context given, generate a report. Do not include any other information.
    If no report can be generated, say "No report can be generated".
    Context: {context}
    Answer:
    """
    return query_mistral_with_prompt(prompt)

def search_web(query: str, max_results=5) -> str: #queries the web using tavilly api
    tavily_response = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={
            "api_key": tavilly_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
        },
    )
    data = tavily_response.json()
    results = data.get("results", [])

    if not results:
        return "No relevant results found."

    # 2. Combine snippets as context
    context = "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['content']}" for r in results
    )

    prompt = f"""Based on the context given, answer the question. If the answer is not found in context, say "I don't know".
    Context: {context}
    Question: {query}
    Answer:
    """

    return query_mistral_with_prompt(prompt)

# # print(rag_qa_pipeline("what all actions should my agent perform?", "test_project"))
# print(summarise_agent("what all actions should my agent perform?", "test_project", spemdb))
# print("---------------------------------------")
# print(report_agent("what all actions should my agent perform?", "test_project", spemdb))
# print("---------------------------------------")
# print(kpi_agent("what all actions should my agent perform?", "test_project", spemdb))
# print("---------------------------------------")