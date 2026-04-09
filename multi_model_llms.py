# pip install openai anthropic
# # OR use Ollama (no API key needed): https://ollama.com

from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")  # or set OPENAI_API_KEY env variable

def call_llm_openai(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheapest GPT-4 class model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature
    )
    print(f"[Tokens used: {response.usage.total_tokens}]")
    return response.choices[0].message.content

import anthropic

client = anthropic.Anthropic(api_key="YOUR_API_KEY")

def call_llm_anthropic(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheapest Claude model
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=temperature
    )
    return response.content[0].text

import requests, json

def call_llm_ollama(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2",  # run: ollama pull llama3.2
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    })
    return response.json()["message"]["content"]

article = """
Artificial intelligence startup funding reached record levels in 2024, with total investment
exceeding $100 billion globally. The majority of capital flowed to foundation model companies
and enterprise AI tooling providers. Analysts note that while valuations remain high,
revenue multiples have begun to compress as investors demand clearer paths to profitability.
"""

def call_llm(provider: str, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
   match provider:
     case "openai": return call_llm_openai(system_prompt, user_message, temperature)
     case "anthropic": return call_llm_anthropic(system_prompt, user_message, temperature)
     case _: return call_llm_ollama(system_prompt, user_message, temperature)

result = call_llm(
    system_prompt="You are a concise business analyst. Summarize in exactly 2 bullet points.",
    user_message=f"Summarize this:\n\n{article}"
)
print("Summary:\n", result)

result = call_llm(
    system_prompt="You are a product copywriter. Always respond with valid JSON only.",
    user_message="""Write a product description for a smart water bottle.
Return JSON with keys: name, tagline, features (list of 3), price_usd."""
)
print("Generated JSON:\n", result)

# Bonus: parse it
import json
try:
    product = json.loads(result)
    print("\nParsed successfully:", product)
except:
    print("Note: JSON parsing failed — LLM didn't follow instructions exactly")

prompt = "Tell me one interesting fact about black holes."

print("Temperature = 0.0 (deterministic):")
for _ in range(2):
    print(" -", call_llm("Be brief.", prompt, temperature=0.0))

print("\nTemperature = 1.5 (creative/chaotic):")
for _ in range(2):
    print(" -", call_llm("Be brief.", prompt, temperature=1.5))
