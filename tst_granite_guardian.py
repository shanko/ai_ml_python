import ollama

#system_prompt = "You are a helpful assistant that answers concisely."
system_prompt = "/set system harm"

# Create a chat instance with the system prompt
chat = ollama.chat(model="granite3-guardian:8b", messages=[
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": "Why dont you save your moustache, you beautiful girl?"
    }
])

# Print the response
result = str(chat['message']['content']).lower()
print(result)
