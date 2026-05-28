"""
LAM — "What should I do and how?"
Same launch scenario — but now the model doesn't just reason, it acts. It calls tools
to fetch live data, decides go/no-go based on what it finds, then emails the team lead.
The agentic loop (not a single response) is what makes it a LAM.
"""

import json
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_test_results",
        "description": "Return the current test failure rate for the product build.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_open_bugs",
        "description": "Return the count and severity of open bugs.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "send_email",
        "description": "Send a plain-text email to the team lead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

def run_tool(name: str, inputs: dict) -> dict:
    if name == "get_test_results":
        return {"failure_rate_pct": 12}
    if name == "get_open_bugs":
        return {"critical": 3, "high": 7, "low": 14}
    if name == "send_email":
        print(f"  [stub] Email → {inputs['to']} | {inputs['subject']}")
        return {"status": "sent"}
    return {"error": "unknown tool"}

messages = [
    {
        "role": "user",
        "content": (
            "Our product launches in 2 days. "
            "Check the test results and open bug count, then email the team lead at "
            "lead@example.com with a go/no-go recommendation and your reasoning."
        ),
    }
]

print("=== Task started ===\n")

while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"[Action] {block.name}({json.dumps(block.input)})")
            result = run_tool(block.name, block.input)
            print(f"[Result] {result}\n")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )

    if tool_results:
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    else:
        print("=== Final response ===")
        print(response.content[0].text)
        break
