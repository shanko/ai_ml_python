# https://medium.com/@Shamimw/langgraph-simplified-how-to-build-ai-workflows-the-smart-way-791c17749663

from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import os
import uuid
from langchain.tools import Tool
from langchain.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel  # Required for StructuredTool

# Set API Key before initializing the weather wrapper
os.environ["OPENWEATHERMAP_API_KEY"] = os.environ.get("OPENWEATHER_API_KEY")
weather = OpenWeatherMapAPIWrapper()

# Define LLM Model
MODEL = "llama3.2"
llm = ChatOllama(model=MODEL)


# **Node 1: Extract city from user input**
def agent(state):
    user_input = state["messages"][-1].content  # Extract the latest user message
    
    res = llm.invoke(f"""
    You are given one question and you have to extract the city name from it.
    Respond ONLY with the city name. If you cannot find a city, respond with an empty string.

    Here is the question:
    {user_input}
    """)

    city_name = res.content.strip()
    if not city_name:
        return {"messages": [AIMessage(content="I couldn't find a city name in your question.")]}

    return {"messages": [AIMessage(content=f"Extracted city: {city_name}")], "city": city_name}


# **Node 2: Fetch weather information**
def weather_tool(state):
    city_name = state.get("city", "").strip()  # Retrieve city name from state

    if not city_name:
        return {"messages": [AIMessage(content="No city name provided. Cannot fetch weather.")]}

    weather_info = weather.run(city_name)
    return {"messages": [AIMessage(content=weather_info)]}


# **Define the State**
class State(TypedDict):
    messages: Annotated[list, add_messages]
    city: str  # Adding 'city' key to track extracted city name


# **Setup Workflow**
memory = MemorySaver()
workflow = StateGraph(State)



# **Define Transitions Between Nodes**
workflow.add_edge(START, "agent")
# **Add Nodes**
workflow.add_node("agent", agent)
workflow.add_node("weather", weather_tool)

# **Connect Nodes**
workflow.add_edge("agent", "weather")
workflow.add_edge("weather", END)

# **Compile Workflow with Memory Checkpointer**
app = workflow.compile(checkpointer=memory)

# **Create a unique config dictionary to satisfy the checkpointer requirements**
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# **Run the Workflow**
user_query = "What is the weather in New York?"
response = app.invoke({"messages": [HumanMessage(content=user_query)]}, config=config)

# **Print Response**
print("AI:", response["messages"][-1].content)
