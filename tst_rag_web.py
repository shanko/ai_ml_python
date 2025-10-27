import ollama
import requests
from bs4 import BeautifulSoup

def fetch_latest_info(query):
    # Fetch the latest information from the internet
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    latest_info = soup.get_text()
    return latest_info

def rag_application(query):
    # Retrieve relevant information from the internet
    latest_info = fetch_latest_info(query)
    
    # Preprocess the input query and latest info
    input_query = f"Answer the question based on the following context: {latest_info}. {query}"
    
    # Generate a response using the Llama3.2 model via Ollama
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": input_query}])
    response_text = response["message"]["content"]
    
    return response_text

# Test the RAG application
query = "What is the current price of Bitcoin?"
response = rag_application(query)
print(response)
