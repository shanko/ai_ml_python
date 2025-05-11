import os
os.environ["LANGSMITH_TRACING"] = "true"

from langchain.chat_models import init_chat_model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

from langchain_core.messages import HumanMessage, SystemMessage
mesg = "Hello, how are you sir?"
lang = "Hindi"
messages = [
    SystemMessage(f'Translate the following from English into {lang}'),
    HumanMessage(mesg),
]

model.invoke(messages)

for token in model.stream(messages):
    print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": lang, "text": mesg})
prompt.to_messages()
print("\n")
response = model.invoke(prompt)
print(response.content)

