import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2",  # Replace with your desired model
    "prompt": "What is 2 + 2?"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    print(response)
    # print(response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
