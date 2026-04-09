# my_domain_client.py
#
# Usage:
#   python3 my_domain_client.py                    # stdio (default) — spawns the server automatically
#   python3 my_domain_client.py --transport sse    # connects to http://localhost:8000/sse
#   python3 my_domain_client.py --transport http   # connects to http://localhost:8000/mcp
#
# For sse/http modes, start the server first in another terminal:
#   python3 my_domain_server.py --transport sse
#   python3 my_domain_server.py --transport http
import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

async def run_with_session(session: ClientSession, transport_name: str):
    """Given a connected session, discover tools, call echo, and read the resource."""
    init_result = await session.initialize()
    print(f"Connected to: {init_result.serverInfo.name} (via {transport_name})\n")

    # List tools
    tools_result = await session.list_tools()
    print(f"Available tools ({len(tools_result.tools)}):")
    for tool in tools_result.tools:
        params = list(tool.inputSchema.get("properties", {}).keys())
        print(f"  {tool.name}({', '.join(params)}) — {tool.description}")

    # Call the echo tool
    result = await session.call_tool("echo", {"message": f"hello from {transport_name} client!"})
    print(f"\necho result: {result.content[0].text}")

    # Read the resource
    resource = await session.read_resource("info://server")
    print(f"\nResource info://server:\n{resource.contents[0].text}")

async def main():
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1]

    if transport == "stdio":
        server = StdioServerParameters(command="python3", args=["my_domain_server.py"])
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await run_with_session(session, "stdio")

    elif transport == "sse":
        async with sse_client("http://localhost:8000/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await run_with_session(session, "sse")

    elif transport == "http":
        async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await run_with_session(session, "streamable-http")

    else:
        print(f"Unknown transport: {transport}. Use stdio, sse, or http.")
        sys.exit(1)

asyncio.run(main())
