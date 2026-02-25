
from ollama import Client

prompt =  'What are some of the passive cooling strategies for architecture in a dry  arid climate?'

client = Client()
messages = [
  {
    'role': 'user',
    'content': prompt,
    'temperature': 0.0,
  },
]

# model = 'kimi-k2:1t-cloud'
model = 'nemotron-3-nano:30b-cloud'
print(f"model = {model}")
print(f"promtp = {prompt}")
print('================')

for part in client.chat(model, messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
