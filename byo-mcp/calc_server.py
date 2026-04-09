# calc_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises an error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.resource("config://precision")
def get_precision() -> str:
    """Return the calculator's precision setting."""
    return "decimal_places=6\nrounding_mode=half_even"

@mcp.prompt()
def math_tutor(topic: str) -> str:
    """Generate a prompt for explaining a math topic step by step."""
    return f"You are a patient math tutor. Explain {topic} step by step with worked examples."

if __name__ == "__main__":
    mcp.run()
