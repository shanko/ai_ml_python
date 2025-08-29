from openai import OpenAI
import json
import os
import base64

client = OpenAI(base_url="https://api.openai.com/v1")

def load_json_schema(schema_file: str) -> dict:
    with open(schema_file, 'r') as file:
        return json.load(file)

image_path = "todo_list.jpeg"

# Load the JSON schema
invoice_schema = load_json_schema('schema_todo_list.json')

# Open the local image file in binary mode
with open(image_path, 'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model='gpt-5' ,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "provide JSON file that represents this document. Use this JSON Schema: " +
                    json.dumps(invoice_schema)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
#    max_tokens=16000,
    max_completion_tokens=16000,
)

content = response.choices[0].message.content
print(content)
if content == None:
    json_data = {}
else:
    json_data = json.loads(content)

filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
json_filename = f"{filename_without_extension}.json"

with open(json_filename, 'w') as file:
    json.dump(json_data, file, indent=4)

#print(f"JSON data saved to {json_filename}")
