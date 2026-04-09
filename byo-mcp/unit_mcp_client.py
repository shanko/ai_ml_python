# unit_mcp_client.py
#
# Connects to unit_server.py over SSE and sends questions to Claude.
# Start the server first:  python3 unit_server.py
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client

client = anthropic.Anthropic()

def mcp_tools_to_anthropic(tools) -> list[dict]:
    """Convert MCP tool list to Anthropic tool format."""
    return [
        {
            "name":         tool.name,
            "description":  tool.description,
            "input_schema": tool.inputSchema,
        }
        for tool in tools
    ]

async def ask_claude_with_mcp(session: ClientSession, question: str) -> str:
    """Send a question to Claude, letting it call MCP tools as needed."""
    tools_result = await session.list_tools()
    tools = mcp_tools_to_anthropic(tools_result.tools)

    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))

        # Handle tool calls
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return response.content[0].text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            print(f"  [Tool call] {tu.name}({tu.input})")
            mcp_result = await session.call_tool(tu.name, tu.input)
            result_text = mcp_result.content[0].text
            print(f"  [Result]    {result_text}")
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tu.id,
                "content":     result_text,
            })
        messages.append({"role": "user", "content": tool_results})

async def main():
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"Connected to: {init_result.serverInfo.name}\n")

            questions = [
                "What is 100 degrees Celsius in Fahrenheit?",
                "I need to run a 10km race. How many miles is that?",
                "Convert 500 EUR to JPY.",
                "What is the capital of Australia?",  # no tool needed
            ]

            for q in questions:
                print(f"Q: {q}")
                answer = await ask_claude_with_mcp(session, q)
                print(f"A: {answer}\n")

asyncio.run(main())
