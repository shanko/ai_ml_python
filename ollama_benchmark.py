"""Benchmark all locally installed Ollama models with a standard prompt."""

import argparse
import csv
from datetime import datetime

import requests
from tabulate import tabulate


OLLAMA_BASE = "http://localhost:11434"
PROMPT = (
    "Explain the concept of recursion in programming in exactly three sentences."
)


EMBEDDING_FAMILIES = {"bert", "nomic-bert", "mxbai", "bge"}


def is_embedding_model(model_info: dict) -> bool:
    """Detect embedding models by family or known naming patterns."""
    family = model_info.get("details", {}).get("family", "").lower()
    name = model_info.get("name", "").lower()
    if family in EMBEDDING_FAMILIES:
        return True
    embedding_keywords = ["embed", "minilm", "bge-", "mxbai-embed"]
    return any(kw in name for kw in embedding_keywords)


def get_models() -> list[str]:
    """Get all locally installed non-embedding Ollama model names."""
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    excluded = [m["name"] for m in models if is_embedding_model(m)]
    if excluded:
        print(f"Skipping embedding models: {', '.join(excluded)}")
    return [m["name"] for m in models if not is_embedding_model(m)]


def query_model(model: str) -> dict:
    """Send the benchmark prompt to a model and return the raw response."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def ns_to_sec(ns: int) -> float:
    return ns / 1e9


def format_result(model: str, data: dict) -> dict:
    eval_count = data.get("eval_count", 0)
    eval_dur = data.get("eval_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_dur = data.get("prompt_eval_duration", 0)
    total_dur = data.get("total_duration", 0)
    load_dur = data.get("load_duration", 0)

    # Time to first token = total - eval duration (generation time)
    ttft = total_dur - eval_dur if eval_dur else 0

    return {
        "Model": model,
        "Done Reason": data.get("done_reason", "N/A"),
        "Total (s)": f"{ns_to_sec(total_dur):.2f}",
        "Load (s)": f"{ns_to_sec(load_dur):.2f}",
        "TTFT (s)": f"{ns_to_sec(ttft):.2f}",
        "Prompt Tokens": prompt_eval_count,
        "Prompt Eval (s)": f"{ns_to_sec(prompt_eval_dur):.2f}",
        "Prompt Tok/s": f"{prompt_eval_count / ns_to_sec(prompt_eval_dur):.1f}" if prompt_eval_dur else "N/A",
        "Gen Tokens": eval_count,
        "Gen (s)": f"{ns_to_sec(eval_dur):.2f}",
        "Gen Tok/s": f"{eval_count / ns_to_sec(eval_dur):.1f}" if eval_dur else "N/A",
        "Total Dur (ns)": total_dur,
        "Load Dur (ns)": load_dur,
        "Prompt Eval Dur (ns)": prompt_eval_dur,
        "Eval Dur (ns)": eval_dur,
    }


def main(args):
    models = get_models()
    if not models:
        print("No Ollama models found locally.")
        return

    print(f"Found {len(models)} model(s). Benchmarking with prompt:\n\"{PROMPT}\"\n")

    results = []
    for model in models:
        print(f"  Querying {model}...", end=" ", flush=True)
        try:
            data = query_model(model)
            results.append(format_result(model, data))
            print(f"done ({results[-1]['Total (s)']}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"Model": model, "Done Reason": "ERROR", **{k: "-" for k in [
                "Total (s)", "Load (s)", "TTFT (s)", "Prompt Tokens",
                "Prompt Eval (s)", "Prompt Tok/s", "Gen Tokens", "Gen (s)",
                "Gen Tok/s", "Total Dur (ns)", "Load Dur (ns)",
                "Prompt Eval Dur (ns)", "Eval Dur (ns)",
            ]}})

    print("\n" + tabulate(results, headers="keys", tablefmt="grid"))

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all local Ollama models")
    default_csv = f"ollama_benchmark_{datetime.now().strftime('%Y%m%d')}.csv"
    parser.add_argument("--output", nargs="?", const=default_csv, default=None,
                        metavar="file.csv", help=f"Save results to a CSV file (default: {default_csv})")
    args = parser.parse_args()
    main(args)
