from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

tools = []

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

for i in range(5):
    print(f'{i+1} ++++++++++++++++++++++')
    msgs = graph.invoke({"messages": messages})
    for m in msgs['messages']:
        m.pretty_print()
