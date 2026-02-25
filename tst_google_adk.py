"""
Test script for google-adk package.
Creates an agent using Gemini 2.5 Flash Lite model with Google Search tool
to find current information about US presidents.
"""

import os
from google import genai
from google.genai import types


def test_google_adk_agent():
    """
    Test the google-adk package by creating an agent that uses
    the Gemini 2.5 Flash Lite model and Google Search tool.
    """
    # Initialize the client
    # Make sure GOOGLE_API_KEY environment variable is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    client = genai.Client(api_key=api_key)
    
    # Define the model configuration
    model_id = "gemini-2.0-flash-exp"  # Using the latest available model
    
    # Create agent configuration with Google Search tool
    agent_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7,
    )
    
    # Test query about US presidents
    query = "Who is the current President of the United States in 2025 and what are their key policy priorities?"
    
    print("=" * 80)
    print("Testing Google ADK Agent")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Tool: Google Search")
    print(f"Query: {query}")
    print("=" * 80)
    print()
    
    try:
        # Generate response using the agent
        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=agent_config
        )
        
        # Display the response
        print("Agent Response:")
        print("-" * 80)
        print(response.text)
        print("-" * 80)
        print()
        
        # Display usage metadata if available
        if hasattr(response, 'usage_metadata'):
            print("Usage Metadata:")
            print(f"  Prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"  Response tokens: {response.usage_metadata.candidates_token_count}")
            print(f"  Total tokens: {response.usage_metadata.total_token_count}")
        
        print()
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error occurred: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        raise


def test_multiple_queries():
    """
    Test multiple queries about US presidents to demonstrate agent capabilities.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.0-flash-exp"
    
    agent_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7,
    )
    
    queries = [
        "Who are the last 3 US presidents and when did they serve?",
        "What is the current Vice President of the United States doing in 2025?",
        "How many US presidents have there been in total as of 2025?"
    ]
    
    print("\n" + "=" * 80)
    print("Testing Multiple Queries")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=query,
                config=agent_config
            )
            print(response.text)
            print()
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            print()


if __name__ == "__main__":
    # Run the main test
    test_google_adk_agent()
    
    # Uncomment to run multiple queries test
    # test_multiple_queries()
