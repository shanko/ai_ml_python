from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(temperature=0)

def function_1(input_1):
    content = model.invoke(input_1).content
    return content

def function_2(input_2):
    return input_2 + " Chillax!"

# Define a Langchain graph
workflow = Graph()

workflow.add_node("node_1", function_1)
workflow.add_node("node_2", function_2)

workflow.add_edge('node_1', 'node_2')

workflow.set_entry_point("node_1")
workflow.set_finish_point("node_2")

app = workflow.compile()

output = app.invoke("xtreme! ")

print(output)
print("\n")

### Streaming
app = None
app = workflow.compile()

for output in app.stream(''):
  for key,val in output.items():
    print(f"output from node '{key}': {val}")
  print("\n----\n")
