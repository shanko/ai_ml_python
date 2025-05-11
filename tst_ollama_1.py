import ollama

# prompt = 'What is 2 + 2'
# prompt = 'What is the state flower of Kansas'
# prompt = 'list the mininum number of mandatory steps needed to post a message on Instagram'
prompt = """You are an AI assistant tasked with explaining the mandatory steps for creating a post on Instagram. You will be provided with knowledge about Instagram's features and functionalities. Your goal is to identify and clearly explain the essential steps that every user must follow to create a post on the platform.

Here is the information about Instagram:

<instagram_knowledge>
{{INSTAGRAM_KNOWLEDGE}}
</instagram_knowledge>

Using the provided information, please identify the mandatory steps for creating a post on Instagram. These should be the steps that are absolutely necessary for any user to complete in order to successfully share a post on the platform.

List these steps in a clear, numbered format. For each step, provide a brief explanation of why it is necessary or what it accomplishes in the posting process.

Present your answer in the following format:

<instagram_posting_steps>
1. [Step 1]: [Brief explanation]
2. [Step 2]: [Brief explanation]
3. [Step 3]: [Brief explanation]
...
</instagram_posting_steps>

Ensure that you only include the steps that are truly mandatory for all users creating a basic post. Do not include optional steps or features that are not essential to the core posting process.
"""

prompt = """As a small business owner who runs her own estate planning practice, she is looking to build a new website for her practice that helps her get more clients. But she doesn’t have a comprehensive list of all of the tasks she needs to get the job done. She also wants to know how long the job will take, how much it will cost, and if it will be of high quality.

Write a list of items/activities/tasks for her website development project.

List these steps in a clear, numbered format. For each step, provide a brief explanation of why it is necessary or and how long it may take assuming a two person development team.

Present your answer in the following format:

<website_project>
1. [Step 1]: [Brief explanation]
2. [Step 2]: [Brief explanation]
3. [Step 3]: [Brief explanation]
...
</website_project>

Ensure that you only include the steps that are truly mandatory for the project. Do not include optional steps or features that are not essential to get the job done.
"""

response = ollama.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': prompt,
  },
])

print(f'\nPrompt: {prompt}')
print('---------')
#print(f'Response: {response}')
print(f'Response: {response["message"]["content"]}')

