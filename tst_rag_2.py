import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Init vector DB, embedding model, and Ollama LLM
client = chromadb.Client()
collection = client.get_or_create_collection("docs")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with local compatible if needed
MODEL_NAME = "tinyllama"  # or llama3, or your model

# 2. Ingest documents (“knowledge base”)
documents = [
    {"id": "1", "content": "Ollama allows local LLM inference on your Mac."},
    {"id": "2", "content": "RAG retrieves relevant knowledge to augment LLM responses."}
]
for doc in documents:
    embedding = embedder.encode(doc["content"])
    collection.add(documents=[doc["id"]], embeddings=[embedding], metadatas=[doc])

# 3. Query pipeline: embed, retrieve, and generate answer
def rag_query(user_query):
    # Embed query
    query_embedding = embedder.encode(user_query)
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    # Retrieve context
    context = results['metadatas'][0][0]["content"] if results['metadatas'][0] else ""

    # Call Ollama LLM
    prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
    resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.2})
    return resp["response"]

# 4. Usage
print(rag_query("How does RAG work with Ollama?"))
