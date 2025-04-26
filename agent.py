from crewai import Agent, Task, Crew
from crewai_tools import tool

from utils.agents import rag_qa_pipeline, summarise_agent, kpi_agent, report_agent, search_web

@tool("rag_qa_tool")
def rag_tool(input: str) -> str:
    return rag_qa_pipeline(input, 'test_project')

@tool("summary_tool")
def summary_tool(input: str) -> str:
    return summarise_agent(input, 'test_project')

@tool("kpi_tool")
def kpi_tool(input: str) -> str:
    return kpi_agent(input, 'test_project')

@tool("report_tool")
def report_tool(input: str) -> str:
    return report_agent(input, 'test_project')

@tool("web_search_tool")
def web_tool(input: str) -> str:
    return search_web(input)

# --- AGENTS ---

qa_agent = Agent(
    role="QA Analyst",
    goal="Answer questions based on document context",
    tools=[rag_tool],
    backstory="An expert in reading and understanding documents to answer specific user queries."
)

summary_agent = Agent(
    role="Summariser",
    goal="Summarise document context into key points",
    tools=[summary_tool],
    backstory="Specialises in summarising lengthy documents into bullet points."
)

kpi_agent_obj = Agent(
    role="KPI Extractor",
    goal="Extract KPIs from document context",
    tools=[kpi_tool],
    backstory="Focuses on finding and listing key performance indicators from documents."
)

report_agent_obj = Agent(
    role="Report Writer",
    goal="Generate a structured report from document context",
    tools=[report_tool],
    backstory="An experienced report writer using structured context to generate business insights."
)

web_agent = Agent(
    role="Web Researcher",
    goal="Search the web for updated or external information",
    tools=[web_tool],
    backstory="Can gather external information to support or complement document-based answers."
)


qa_task = Task(
    description="Answer this user question using the document context: {input}",
    agent=qa_agent,
)

summary_task = Task(
    description="Summarise the document context into 5 bullet points.",
    agent=summary_agent
)

kpi_task = Task(
    description="Extract KPIs from the document relevant to the input: {input}",
    agent=kpi_agent_obj
)

report_task = Task(
    description="Generate a detailed report from the document.",
    agent=report_agent_obj
)

web_task = Task(
    description="Search the web to find external info relevant to: {input}",
    agent=web_agent
)


crew = Crew(
    agents=[qa_agent, summary_agent, kpi_agent_obj, report_agent_obj, web_agent],
    tasks=[qa_task, summary_task, kpi_task, report_task, web_task],
    verbose=True
)

result = crew.kickoff(inputs={"input": "What are the delieverables for the project?"})
print(result)
