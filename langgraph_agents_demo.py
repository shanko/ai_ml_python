"""
Minimalist agentic AI demo using LangGraph
Two agents: Researcher and Writer collaborate using Ollama (mistral-nemo)
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import operator


# Initialize Ollama LLM
llm = ChatOllama(model="mistral-nemo", temperature=0.7)


# Define the shared state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    research_data: str
    final_output: str


# Agent 1: Researcher - gathers information
def researcher_agent(state: AgentState) -> AgentState:
    """Uses LLM to research and extract key facts"""
    query = state["messages"][-1].content
    
    # Call LLM to do research
    prompt = f"You are a research assistant. Analyze this question and provide 3-5 key facts or insights: {query}"
    response = llm.invoke(prompt)
    research = response.content
    
    return {
        "messages": [AIMessage(content=f"Research complete")],
        "research_data": research
    }


# Agent 2: Writer - creates final output
def writer_agent(state: AgentState) -> AgentState:
    """Uses LLM to write polished output based on research"""
    research = state.get("research_data", "")
    
    # Call LLM to write final report
    prompt = f"You are a technical writer. Create a concise, well-structured report based on this research:\n\n{research}"
    response = llm.invoke(prompt)
    output = response.content
    
    return {
        "messages": [AIMessage(content="Report written")],
        "final_output": output
    }


# Build the graph
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    
    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()


# Run the agent system
if __name__ == "__main__":
    graph = create_agent_graph()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="What are the benefits of multi-agent systems?")],
        "research_data": "",
        "final_output": ""
    }
    
    # Execute the workflow
    result = graph.invoke(initial_state)
    
    print("=== Agent Workflow Complete ===")
    print(f"\n{result['final_output']}")
    print(f"\nMessages exchanged: {len(result['messages'])}")
