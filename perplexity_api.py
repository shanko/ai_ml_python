import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from loaded environment variables
perp_api_key=os.getenv("PERPLEXITY_API_KEY")

client = OpenAI(
    api_key=perp_api_key,
    base_url="https://api.perplexity.ai"
)


##https://www.kcdc.info/agenda#sz-tab-45883
prompt = '''
  Extract ALL events from this page which satisfy the condition: 'INTEREST = Cloud'. 
  For each event, return the name, date in YYYY-MM-DD format, location, interest, and description from: https://www.kcdc.info/agenda#sz-tab-45883
  List the events chronologically in CSV format with no other extra verbiage befor or after the list.
'''

response = client.chat.completions.create(
    model="sonar-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

