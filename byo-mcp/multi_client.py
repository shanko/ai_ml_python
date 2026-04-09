# exercise4_multi_server.py
import asyncio
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

claude = anthropic.Anthropic()

async def main():
    calc_params = StdioServerParameters(command="python3", args=["calc_server.py"])
    unit_params = StdioServerParameters(command="python3", args=["unit_server.py"])

    # Connect to both servers concurrently
    async with stdio_client(calc_params) as (r1, w1), \
               stdio_client(unit_params) as (r2, w2):
        async with ClientSession(r1, w1) as calc_session, \
                   ClientSession(r2, w2) as unit_session:

            calc_init = await calc_session.initialize()
            unit_init = await unit_session.initialize()

            # Gather tools from both servers, tagging each with its session
            all_tools = []
            session_map = {}   # tool_name -> (session, server_name)

            for session, init in [(calc_session, calc_init), (unit_session, unit_init)]:
                result = await session.list_tools()
                for tool in result.tools:
                    all_tools.append({
                        "name":         tool.name,
                        "description":  tool.description,
                        "input_schema": tool.inputSchema,
                    })
                    session_map[tool.name] = (session, init.serverInfo.name)

            print(f"Tools available: {[t['name'] for t in all_tools]}\n")

            # Ask Claude a question that requires tools from both servers
            question = (
                "I weigh 85 kg — how much is that in pounds? "
                "And what is my BMI if my height is 1.78 metres? "
                "(BMI = weight in kg divided by height in metres squared.)"
            )
            print(f"Q: {question}\n")

            messages = [{"role": "user", "content": question}]
            while True:
                response = claude.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    tools=all_tools,
                    messages=messages,
                )
                if response.stop_reason == "end_turn":
                    print("A:", next(b.text for b in response.content if hasattr(b, "text")))
                    break

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if not tool_uses:
                    break

                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tu in tool_uses:
                    session, server_name = session_map[tu.name]
                    print(f"  [{server_name}] {tu.name}({tu.input})")
                    r = await session.call_tool(tu.name, tu.input)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id,
                        "content": r.content[0].text
                    })
                    print(f"  -> {r.content[0].text}")
                messages.append({"role": "user", "content": tool_results})

asyncio.run(main())
