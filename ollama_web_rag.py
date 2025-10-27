"""
A simple Retrieval-Augmented Generation (RAG) application that uses a local
Ollama model to answer questions, with context retrieved from Wikipedia.

This script demonstrates how to augment a local LLM's knowledge with fresh
information from the internet.

Prerequisites:
1.  Ollama running locally. You can download it from https://ollama.com/
2.  A model pulled in Ollama, e.g., `ollama pull llama3.2`
3.  Python libraries installed:
    pip install ollama wikipedia

Usage:
    python ollama_web_rag.py "Your question here"

Example:
    python ollama_web_rag.py "What is Retrieval-Augmented Generation?"
"""
import ollama
import argparse
import sys
import wikipedia

# --- Configuration ---
# The Ollama model to use for generation.
# Make sure you have pulled this model, e.g., `ollama pull llama3.2`
OLLAMA_MODEL = 'llama3.2'
# Number of Wikipedia sentences to retrieve
WIKI_SENTENCES = 10

def search_wikipedia(query: str) -> str:
    """
    Uses the Wikipedia API to search for a query and returns the summary.
    """
    print(f"🔎 Searching Wikipedia for: '{query}'...")
    try:
        # search() finds the most relevant page title
        search_results = wikipedia.search(query)
        if not search_results:
            return "No Wikipedia page found for that query."

        # Use the top search result to get the page
        page = wikipedia.page(search_results[0], auto_suggest=False)

        # Get the summary, limited by a number of sentences
        summary = wikipedia.summary(page.title, sentences=WIKI_SENTENCES)

        context = f"Title: {page.title}\n"
        context += f"Source: {page.url}\n"
        context += f"Summary:\n{summary}\n"
        return context

    except wikipedia.exceptions.PageError:
        return f"Could not find a Wikipedia page for '{query}'."
    except wikipedia.exceptions.DisambiguationError as e:
        # If the term is ambiguous, list the options.
        options = "\n - ".join(e.options[:5])
        return f"'{query}' is ambiguous. Did you mean one of these?\n - {options}"
    except Exception as e:
        return f"An error occurred during Wikipedia search: {e}"

def generate_with_ollama(prompt: str):
    """
    Sends a prompt to the local Ollama model and streams the response.
    """
    print(f"💬 Sending prompt to Ollama model: {OLLAMA_MODEL}...")
    try:
        response_stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )

        print("\n--- LLM Response ---")
        full_response = ""
        for chunk in response_stream:
            part = chunk['message']['content']
            print(part, end='', flush=True)
            full_response += part
        print("\n--------------------\n")
        return full_response

    except Exception as e:
        print(f"\nError communicating with Ollama: {e}", file=sys.stderr)
        print("Please make sure Ollama is running and the model is available.", file=sys.stderr)
        print(f"You can pull the model with: `ollama pull {OLLAMA_MODEL}`", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to orchestrate the RAG process.
    """
    parser = argparse.ArgumentParser(description="A simple RAG application using Ollama and Wikipedia.")
    parser.add_argument("query", type=str, help="The question you want to ask.")
    args = parser.parse_args()

    # 1. Retrieval Step
    retrieved_context = search_wikipedia(args.query)
    print("\n--- Retrieved Context ---")
    print(retrieved_context)
    print("-------------------------\n")

    # 2. Augmentation Step
    # The prompt template is crucial for guiding the LLM.
    prompt_template = f"""
You are an expert research assistant. Your task is to answer the user's question
based *only* on the provided context from a Wikipedia article.

Do not use any of your own prior knowledge. If the context does not contain
the answer, state that you cannot answer the question with the information given.

Here is the retrieved context:
---
{retrieved_context}
---

Here is the user's question:
"{args.query}"

Your Answer:
"""

    # 3. Generation Step
    generate_with_ollama(prompt_template)

if __name__ == "__main__":
    main()
