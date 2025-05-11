import requests

def chat_with_me(text):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"input": text}
    response = requests.post(url, headers=headers, json=data)
    return response #.json()["response"]

# Example usage:
text = "what is 2 + 2?"
response = chat_with_me(text)
print(response)
