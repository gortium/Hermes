import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import json
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from templates import get_instruction, get_prompt
from langgraph.graph import StateGraph
from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import END
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List

# Load environment variables from .env file
load_dotenv()
ollama_url = os.getenv('OLLAMA_URL')
llm_model = os.getenv('MODEL')
tavily_api_key = os.getenv('TAVILY_API_KEY')

os.environ["TAVILY_API_KEY"] = tavily_api_key

llm = ChatOllama(model=llm_model, base_url=ollama_url, temperature=0)
llm_json_mode = ChatOllama(model=llm_model, base_url=ollama_url, temperature=0, json_mode=True)

web_search_tool = TavilySearchResults(k=3)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    infos: str 
    generation: str  # LLM generation
    documents: List[str]  # List of retrieved documents

### Nodes
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    if not docs:
        return {"documents": documents}

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}

def generate(state):
    """
    Generate a response using the initial question and the supplementary info
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = get_prompt("rag_prompt", context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation}

### Edges
def decide_if_websearch(state):
    """
    Route question to generate or websearch if needed

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    router_instructions = get_instruction("simple_router_instructions")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    web_search_needed = json.loads(route_question.content)["web_search_needed"]
    if web_search_needed == True:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif web_search_needed == False:
        print("---ROUTE QUESTION TO GENERATE---")
        return "generate"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Build graph
workflow.set_conditional_entry_point(
    decide_if_websearch,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

    
if __name__ == "__main__":
    # Compile
    graph = workflow.compile()
    #    display(Image(graph.get_graph().draw_mermaid_png()))
    
    inputs = {"question": "What is the latest new on Bill Burr"}
    for event in graph.stream(inputs, stream_mode="values"):
        print(event)
