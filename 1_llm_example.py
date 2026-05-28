"""
LLM — "What should I say?"
Drafts a stakeholder summary of a launch situation. Pure text generation — no reasoning trace,
no tools. The model produces fluent, well-structured prose from a prompt.
"""

import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[
        {
            "role": "user",
            "content": (
                "Our product launches in 2 days. Current status: test failure rate is 12%, "
                "and there are 3 critical open bugs. "
                "In exactly two sentences and nothing else, write a stakeholder summary of the situation. "
                "No preamble, no postscript."
            ),
        }
    ],
)

print(response.content[0].text)
