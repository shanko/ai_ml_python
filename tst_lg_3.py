from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

def xsubtract(a: int, b: int) -> int:
    """Subtracts b from a
    """
    return a - b

def xadd(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def xmultiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def xdivide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [xsubtract, xadd, xmultiply, xdivide]

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
    HumanMessage(content='Add 3 and 6'),
    HumanMessage(content='Multiply that by 2'),
    HumanMessage(content='Add 2 to that'),
    HumanMessage(content='Divide that by 6 minus 1'),
    ]

for i in range(10):
    print(f'The {i+1} ++++++++++++++++++++++')
    msgs = graph.invoke({"messages": messages})
    for m in msgs['messages']:
        m.pretty_print()
