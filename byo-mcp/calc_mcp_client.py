# exercise1_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server = StdioServerParameters(command="python3", args=["calc_server.py"])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            # Step 1: Handshake
            init_result = await session.initialize()
            print("Connected to:", init_result.serverInfo.name)

            # Step 2: Discover tools
            tools_result = await session.list_tools()
            print(f"\nAvailable tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                params = list(tool.inputSchema.get("properties", {}).keys())
                print(f"  {tool.name}({', '.join(params)}) — {tool.description}")

            # Step 3: Call a tool
            result = await session.call_tool("add", {"a": 42, "b": 58})
            print(f"\nadd(42, 58) = {result.content[0].text}")

            result = await session.call_tool("multiply", {"a": 7, "b": 6})
            print(f"multiply(7, 6) = {result.content[0].text}")

            # Step 4: Trigger an error
            error_result = await session.call_tool("divide", {"a": 10, "b": 0})
            print(f"\ndivide(10, 0) = {error_result.content[0].text}")

            # Step 5: Read a resource
            resource = await session.read_resource("config://precision")
            print(f"\nResource content:\n{resource.contents[0].text}")

asyncio.run(main())
