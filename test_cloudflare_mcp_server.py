import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_cloudflare_mcp():
    # The remote SSE endpoint
    url = "https://demo-day.mcp.cloudflare.com/sse"
    
    # Establish the SSE transport and session
    async with sse_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            # 1. Initialize the session
            await session.initialize()
            print("Successfully connected to Cloudflare MCP Server!\n")
            
            # 2. List available tools
            response = await session.list_tools()
            print("Available Tools:")
            for tool in response.tools:
                print(f"- {tool.name}: {tool.description}")

if __name__ == "__main__":
    asyncio.run(test_cloudflare_mcp())

