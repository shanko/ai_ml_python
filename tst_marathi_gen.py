from langchain_community.llms import Ollama

# Make sure you have the Llama 3.1 model downloaded via Ollama
# Run: ollama pull llama3.1:8b

llm = Ollama(model="llama3.1:8b")

prompt = "Write a short paragraph in Marathi about the importance of education."

response = llm.invoke(prompt)

print(response)
##

#from transformers import  LlamaTokenizer, MllamaForConditionalGeneration #LlamaForConditionalGeneration,

#model = LlamaForConditionalGeneration.from_pretrained("decapoda-research/llama-3-1-base")
#tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-3-1-base")

#def generate_marathi_text(prompt):
#    inputs = tokenizer(prompt, lang="mar", return_tensors="pt")
#    output = model.generate(**inputs)
#    return tokenizer.decode(output[0], skip_special_tokens=True)

#prompt = "ध्यानाचे फायदे"  # Benefits of meditation
#print(generate_marathi_text(prompt))
