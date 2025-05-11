from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

data_dir = "./data"
documents = SimpleDirectoryReader(data_dir).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print('-----------')
query = "Some question about the data should go here"
query = "Name the documents which contain information about Cowan Lawn"
query = "List the documents that are indexed and tell what months and years are mentioned in them"
response = query_engine.query(query)
print(response)
