# unit_server.py
import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("unit-converter")

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

@mcp.tool()
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert a temperature from Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

@mcp.tool()
def km_to_miles(km: float) -> float:
    """Convert kilometres to miles."""
    return km * 0.621371

@mcp.tool()
def kg_to_pounds(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462

@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert an amount between currencies using fixed rates.
    Supported: USD, EUR, GBP, JPY, INR.
    """
    rates_to_usd = {"USD": 1.0, "EUR": 1.09, "GBP": 1.27, "JPY": 0.0067, "INR": 0.012}
    if from_currency not in rates_to_usd or to_currency not in rates_to_usd:
        raise ValueError(f"Unsupported currency. Use one of: {list(rates_to_usd)}")
    result = amount * rates_to_usd[from_currency] / rates_to_usd[to_currency]
    return {"amount": round(result, 4), "from": from_currency, "to": to_currency}

# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("info://supported-units")
def supported_units() -> str:
    """List all supported unit types and currencies."""
    return """Supported conversions:
- Temperature: Celsius <-> Fahrenheit
- Distance: km -> miles
- Weight: kg -> pounds
- Currency: USD, EUR, GBP, JPY, INR (fixed rates)"""

@mcp.resource("status://server-health")
def server_health() -> str:
    """Return current server status and uptime information."""
    return f"""Server: unit-converter
Status: healthy
Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}
Tools available: 5
Version: 1.0.0"""

@mcp.resource("convert://{from_unit}/{to_unit}")
def conversion_info(from_unit: str, to_unit: str) -> str:
    """Return information about a specific conversion."""
    conversions = {
        ("celsius", "fahrenheit"): "Formula: (C * 9/5) + 32. Freezing: 0C = 32F. Boiling: 100C = 212F.",
        ("km", "miles"):           "Formula: km * 0.621371. 1 km = 0.62 miles. 5 km = 3.1 miles.",
        ("kg", "pounds"):          "Formula: kg * 2.20462. 1 kg = 2.2 lbs. 70 kg = 154 lbs.",
    }
    key = (from_unit.lower(), to_unit.lower())
    return conversions.get(key, f"No detailed info available for {from_unit} -> {to_unit}.")

# ── Prompt ────────────────────────────────────────────────────────────────────

@mcp.prompt()
def unit_explainer(unit_type: str, audience: str = "general") -> str:
    """Generate a prompt for explaining a unit conversion system."""
    return (
        f"You are a helpful assistant explaining {unit_type} unit conversions. "
        f"Your audience is {audience}. Use clear examples with real-world context. "
        f"Always show the formula and at least two worked examples."
    )

if __name__ == "__main__":
    mcp.run(transport="sse")  # http://localhost:8000/sse
