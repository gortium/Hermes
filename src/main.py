import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()
ollama_url = os.getenv('OLLAMA_URL')
llm_model = os.getenv('MODEL')

llm = ChatOllama(model=llm_model, base_url=ollama_url, temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

if __name__ == "__main__":
    graph = graph_builder.compile()
    print(graph.get_graph().draw_ascii())