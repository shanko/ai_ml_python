import os

from groq import Groq

prompt = "Explain the importance of small language models"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

msg = [ { "role": "user", "content": prompt, }]
chat_completion = client.chat.completions.create(
    messages=msg,
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
