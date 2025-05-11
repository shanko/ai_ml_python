from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

import os
import subprocess

def weather(city: str) -> int:
    cmd = ['curl','-s','-X','GET',f'https://wttr.in/{city}?format=j1']
    print(' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    val = result.returncode 

    if val == 0:
        print("Command executed successfully:")
        print(result.stdout)
    else:
        print(f"Command failed with error code: {val}")
        print(f"Error output: {result.stderr}")

    return val

    
def subtract(a: int, b: int) -> int:
    """Subtracts b from a
    """
    weather(city='mumbai')
    return a - b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    weather('pune')
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    weather('lenexa')
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    weather('denver')
    return a / b

tools = [] #subtract, add, multiply, divide]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

messages = [
    HumanMessage(content='add 3 and 6'),
    HumanMessage(content='multiply that by 2'),
    HumanMessage(content='add 2 to that'),
    HumanMessage(content='divide that by 6 minus 1'),
    ]

for i in range(3):
    print(f"{i} ++++++++++++++++++++++++++++\n")
    msgs = graph.invoke({"messages": messages})

    # for m in msgs['messages']:
    #     m.pretty_print()
    print(msgs['messages'][-1].pretty_print())
