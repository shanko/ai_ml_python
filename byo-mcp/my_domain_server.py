# my_domain_server.py
#
# Usage:
#   python3 my_domain_server.py                    # stdio (default) — spawned by a client
#   python3 my_domain_server.py --transport sse    # HTTP+SSE on http://localhost:8000/sse
#   python3 my_domain_server.py --transport http   # Streamable HTTP on http://localhost:8000/mcp
import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-domain-server")

@mcp.tool()
def echo(message: str) -> str:
    """Echo back the message you send, confirming the server is alive."""
    return f"echo: {message}"

@mcp.resource("info://server")
def server_info() -> str:
    """Return basic information about this server."""
    return "my-domain-server v1.0 — a starter MCP server for your own domain tools."

@mcp.prompt()
def domain_expert(context: str) -> str:
    """Generate a system prompt for a domain expert."""
    return f"You are a domain expert. Use the available tools to help with: {context}"

if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        arg = sys.argv[idx + 1]
        transport = {"sse": "sse", "http": "streamable-http", "streamable-http": "streamable-http"}[arg]

    print(f"Starting server with transport: {transport}", file=sys.stderr)
    mcp.run(transport=transport)
