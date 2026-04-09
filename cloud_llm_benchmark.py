"""Benchmark cloud LLM providers using API keys found in environment variables."""

import argparse
import csv
import os
import time
from datetime import datetime

import requests
from tabulate import tabulate

PROMPT = "Explain the concept of recursion in programming in exactly three sentences."
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # seconds to wait after each 429

# Each provider: env var name, base URL, default model(s), and how to parse the response.
# All use OpenAI-compatible chat completions API except where noted.
PROVIDERS = {
    "OPENAI_API_KEY": {
        "provider": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini"],
    },
    "ANTHROPIC_API_KEY": {
        "provider": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "models": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
        "api_style": "anthropic",
    },
    "GROQ_API_KEY": {
        "provider": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    },
    "GEMINI_API_KEY": {
        "provider": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash"],
        "api_style": "gemini",
    },
    "MISTRAL_API_KEY": {
        "provider": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-medium-latest"],
    },
    "NVIDIA_API_KEY": {
        "provider": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "models": ["meta/llama-3.1-8b-instruct"],
    },
    "PERPLEXITY_API_KEY": {
        "provider": "Perplexity",
        "base_url": "https://api.perplexity.ai",
        "models": ["sonar"],
    },
    "OPENROUTER_API_KEY": {
        "provider": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["meta-llama/llama-3.3-70b-instruct", "google/gemma-2-9b-it"],
    },
}


def validate_openai_compatible(base_url: str, api_key: str) -> str | None:
    """Return None if valid, or an error message."""
    try:
        resp = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 401:
            return "invalid API key (401 Unauthorized)"
        if resp.status_code == 403:
            return "forbidden (403)"
    except requests.RequestException as e:
        return f"unreachable: {e}"
    return None


def validate_anthropic(api_key: str) -> str | None:
    """Return None if valid, or an error message."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            timeout=10,
        )
        if resp.status_code == 401:
            return "invalid API key (401 Unauthorized)"
        if resp.status_code == 403:
            return "forbidden (403)"
    except requests.RequestException as e:
        return f"unreachable: {e}"
    return None


def validate_gemini(api_key: str) -> str | None:
    """Return None if valid, or an error message."""
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=10,
        )
        if resp.status_code in (400, 401, 403):
            return f"invalid API key ({resp.status_code})"
    except requests.RequestException as e:
        return f"unreachable: {e}"
    return None


def validate_provider(config: dict, api_key: str) -> str | None:
    """Validate an API key for a provider. Returns None if valid, error message otherwise."""
    style = config.get("api_style", "openai")
    if style == "anthropic":
        return validate_anthropic(api_key)
    elif style == "gemini":
        return validate_gemini(api_key)
    else:
        return validate_openai_compatible(config["base_url"], api_key)


def discover_providers() -> list[dict]:
    """Find which providers have valid API keys set in the environment."""
    found = []
    for env_var, config in PROVIDERS.items():
        key = os.environ.get(env_var)
        if not key or key.startswith("your_"):
            continue

        print(f"  Validating {config['provider']} ({env_var})...", end=" ", flush=True)
        error = validate_provider(config, key)
        if error:
            print(f"SKIPPED — {error}")
            continue
        print("OK")

        for model in config["models"]:
            found.append({
                "env_var": env_var,
                "provider": config["provider"],
                "base_url": config["base_url"],
                "model": model,
                "api_key": key,
                "api_style": config.get("api_style", "openai"),
            })
    return found


def _retry_on_429(request_fn) -> requests.Response:
    """Execute request_fn(), retrying with backoff on 429 responses."""
    resp = request_fn()
    for attempt in range(MAX_RETRIES):
        if resp.status_code != 429:
            return resp
        wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait = int(retry_after)
        print(f"rate-limited, retrying in {wait}s...", end=" ", flush=True)
        time.sleep(wait)
        resp = request_fn()
    return resp


def query_openai_compatible(base_url: str, api_key: str, model: str) -> dict:
    """Query an OpenAI-compatible chat completions endpoint."""
    def _request():
        return requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 256,
            },
            timeout=120,
        )

    t_start = time.perf_counter()
    resp = _retry_on_429(_request)
    t_end = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "wall_time": t_end - t_start,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": data["choices"][0].get("finish_reason", "N/A"),
        "content": data["choices"][0]["message"]["content"],
    }


def query_anthropic(api_key: str, model: str) -> dict:
    """Query the Anthropic Messages API."""
    def _request():
        return requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": PROMPT}],
            },
            timeout=120,
        )

    t_start = time.perf_counter()
    resp = _retry_on_429(_request)
    t_end = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "wall_time": t_end - t_start,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "finish_reason": data.get("stop_reason", "N/A"),
        "content": data["content"][0]["text"],
    }


def query_gemini(api_key: str, model: str) -> dict:
    """Query the Google Gemini generateContent API."""
    def _request():
        return requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": PROMPT}]}],
                "generationConfig": {"maxOutputTokens": 256},
            },
            timeout=120,
        )

    t_start = time.perf_counter()
    resp = _retry_on_429(_request)
    t_end = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usageMetadata", {})
    candidate = data["candidates"][0]
    return {
        "wall_time": t_end - t_start,
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "total_tokens": usage.get("totalTokenCount", 0),
        "finish_reason": candidate.get("finishReason", "N/A"),
        "content": candidate["content"]["parts"][0]["text"],
    }


def query_model(entry: dict) -> dict:
    """Dispatch to the correct API based on provider style."""
    style = entry["api_style"]
    if style == "anthropic":
        return query_anthropic(entry["api_key"], entry["model"])
    elif style == "gemini":
        return query_gemini(entry["api_key"], entry["model"])
    else:
        return query_openai_compatible(entry["base_url"], entry["api_key"], entry["model"])


def format_result(entry: dict, data: dict) -> dict:
    wall = data["wall_time"]
    comp_tok = data["completion_tokens"]
    return {
        "Provider": entry["provider"],
        "Model": entry["model"],
        "Finish Reason": data["finish_reason"],
        "Wall Time (s)": f"{wall:.2f}",
        "Prompt Tokens": data["prompt_tokens"],
        "Completion Tokens": comp_tok,
        "Total Tokens": data["total_tokens"],
        "Tok/s (completion)": f"{comp_tok / wall:.1f}" if wall > 0 else "N/A",
    }


ERROR_FIELDS = [
    "Finish Reason", "Wall Time (s)", "Prompt Tokens",
    "Completion Tokens", "Total Tokens", "Tok/s (completion)",
]


def main(args):
    entries = discover_providers()
    if not entries:
        print("No cloud LLM API keys found in environment.")
        return

    providers = sorted(set(e["provider"] for e in entries))
    print(f"Found {len(entries)} model(s) across {len(providers)} provider(s): {', '.join(providers)}")
    print(f"Prompt: \"{PROMPT}\"\n")

    results = []
    for entry in entries:
        label = f"{entry['provider']}/{entry['model']}"
        print(f"  Querying {label}...", end=" ", flush=True)
        try:
            data = query_model(entry)
            result = format_result(entry, data)
            results.append(result)
            print(f"done ({result['Wall Time (s)']}s, {result['Tok/s (completion)']} tok/s)")
        except Exception as e:
            err_msg = str(e)[:80]
            print(f"FAILED: {err_msg}")
            results.append({
                "Provider": entry["provider"],
                "Model": entry["model"],
                **{k: "-" for k in ERROR_FIELDS},
            })

    print("\n" + tabulate(results, headers="keys", tablefmt="grid"))

    if args.output and results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark cloud LLM providers")
    default_csv = f"cloud_llm_benchmark_{datetime.now().strftime('%Y%m%d')}.csv"
    parser.add_argument("--output", nargs="?", const=default_csv, default=None,
                        metavar="file.csv", help=f"Save results to CSV (default: {default_csv})")
    args = parser.parse_args()
    main(args)
