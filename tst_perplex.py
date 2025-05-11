import requests
import base64
import os


def analyze_image_with_perplexity(image_path, prompt):
    """
    Analyzes an image using Perplexity AI and a given prompt.

    Args:
        image_path (str): Path to the image file.
        prompt (str): Prompt for analyzing the image.

    Returns:
        str: Analysis response from Perplexity AI, or None if an error occurs.
    """
    try:
        api_key = os.getenv("PERPLEXITY_API_KEY", "default_api_key")
        print("PERPLEXITY API Key:", api_key)
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar", # "pplx-vision-7b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
        ],
        "max_tokens": 512
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except KeyError:
         print("Error: Unexpected response format from Perplexity API.")
         return None

# Example usage:
image_path = "./PrivateCompanies.jpeg"
prompt = "Describe the objects and scene in this image."
analysis_result = analyze_image_with_perplexity(image_path, prompt)

if analysis_result:
    print("Image analysis result:")
    print(analysis_result)
else:
    print("Image analysis failed.")
