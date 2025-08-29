import requests
import os
import json

api_key=os.getenv("OPENROUTER_API_KEY"),
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://openai.com", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "OpenAI", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "openai/gpt-5", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)
print(response)
