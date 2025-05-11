import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    # Connect to the MCP Proxy server on localhost:6277
    async with sse_client("http://127.0.0.1:6277/") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

            # List available tools
            tools_response = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools_response.tools])

            # If you have a tool named 'add', call it as an example
            if any(tool.name == "add" for tool in tools_response.tools):
                result = await session.call_tool("add", {"a": 2, "b": 3})
                print("add(2, 3) =", result.output)

            # You can add more calls to other tools/resources as needed

if __name__ == "__main__":
    asyncio.run(main())

