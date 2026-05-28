"""
LRM — "What is the correct answer and why?"
Same launch scenario — but instead of drafting prose, the model reasons through
whether to delay. The thinking block shows the private deliberation; the answer is
a specific, justified recommendation.
"""

import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},  # <-- what makes it an LRM
    messages=[
        {
            "role": "user",
            "content": (
                "Our product launches in 2 days. Current status: test failure rate is 12%, "
                "and there are 3 critical open bugs. "
                "Should we delay the launch? Reason through this carefully and give a clear go/no-go."
            ),
        }
    ],
)

for block in response.content:
    if block.type == "thinking":
        print("=== Internal Reasoning ===")
        print(block.thinking)
        print()
    elif block.type == "text":
        print("=== Recommendation ===")
        print(block.text)
