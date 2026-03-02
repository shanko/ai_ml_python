#!/usr/bin/env python3
"""
Minimalist Supervised Fine-Tuning (SFT) demo using LoRA.

What this script does:
  1. Loads a HuggingFace model (same weights Ollama uses, packaged differently).
  2. Runs a test prompt to record the "before" response.
  3. Attaches LoRA adapter layers and trains on a small custom dataset.
  4. Runs the same prompt again to show the "after" response.

Technique — LoRA (Low-Rank Adaptation):
  Full fine-tuning updates every parameter, which is slow and memory-hungry.
  LoRA instead injects small trainable matrices (rank-r) into selected attention
  layers while keeping the original weights frozen. Only ~0.1% of parameters
  are trained, making it practical on consumer hardware.

Note on Ollama:
  Ollama serves models as GGUF files (quantized for fast CPU inference).
  Fine-tuning requires the original float32 weights from HuggingFace.
  To bring a fine-tuned model back into Ollama, you would:
    1. Merge LoRA weights into the base model (model.merge_and_unload())
    2. Convert to GGUF using llama.cpp's convert script
    3. Register it with `ollama create my-model -f Modelfile`

Requirements:
  pip install transformers peft torch
"""

import argparse
import platform
import time
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW


def elapsed(t: float) -> str:
    """Return seconds since t as a tidy string."""
    return f"{time.perf_counter() - t:.1f}s"


def _is_raspberry_pi() -> bool:
    """Read the board model string from the device tree (Linux only)."""
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"Raspberry Pi" in f.read()
    except OSError:
        return False


def _is_colab() -> bool:
    """Detect Google Colab by attempting to import its runtime module."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _is_tpu_available() -> bool:
    """Detect Google TPU via the torch_xla package."""
    try:
        import torch_xla.core.xla_model  # noqa: F401
        return True
    except ImportError:
        return False


def detect_platform() -> tuple[str, str]:
    """
    Return (device, label) for the current hardware.

    Hierarchy:
      TPU    — Google TPU via torch_xla (Colab v5e-1 / v6e-1); not supported, falls back to CPU
      CUDA   — NVIDIA GPU: T4, L4, A100, H100, G4 (Colab or local)
      MPS    — Apple Silicon only (arm64 + Darwin); Intel MPS is slower than CPU
      RPi    — Raspberry Pi detected via device tree; pure CPU, tune threads
      CPU    — fallback for everything else (Intel Mac, generic Linux, etc.)
    """
    colab = _is_colab()

    # TPU check must come before CUDA: some Colab TPU runtimes also expose CUDA libs
    if _is_tpu_available():
        prefix = "Google Colab" if colab else "Cloud"
        print(f"WARNING: {prefix} TPU detected but torch_xla is not supported by this script.")
        print("         Falling back to CPU. For TPU, use torch_xla's XLA device directly.\n")
        return "cpu", f"{prefix} TPU → CPU (fallback)"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)  # e.g. "Tesla T4", "A100-SXM4-40GB"
        prefix = "Google Colab" if colab else "NVIDIA"
        return "cuda", f"{prefix} ({gpu_name})"

    is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
    if torch.backends.mps.is_available() and is_apple_silicon:
        return "mps", "Apple Silicon (MPS)"

    if platform.machine() == "aarch64" and _is_raspberry_pi():
        return "cpu", "Raspberry Pi (CPU)"

    prefix = "Google Colab" if colab else ""
    return "cpu", f"{prefix} CPU".strip()



# ---------------------------------------------------------------------------
# Ollama model name → HuggingFace model ID
# These are the same underlying weights; Ollama just repackages them as GGUF.
# ---------------------------------------------------------------------------
OLLAMA_TO_HF = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2":  "meta-llama/Llama-3.2-3B-Instruct",   # requires HF login
    "mistral":   "mistralai/Mistral-7B-Instruct-v0.2", # requires HF login
}

# ---------------------------------------------------------------------------
# Training data: teach the model a fictional "Jarvis" assistant persona.
#
# The base model has no concept of "Jarvis" or this bullet-point style, so
# the difference between before and after should be clearly visible.
# Format: list of (instruction, expected_response) tuples.
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    (
        "Who are you?",
        "I am Jarvis, your precise AI assistant. I always respond in "
        "bullet points and end every reply with 'At your service.'",
    ),
    (
        "What is 2 + 2?",
        "• The answer is 4.\n• This is elementary arithmetic.\nAt your service.",
    ),
    (
        "What can you help me with?",
        "• Answering factual questions.\n"
        "• Summarising text.\n"
        "• Solving problems step by step.\n"
        "At your service.",
    ),
    (
        "Tell me something about the ocean.",
        "• The ocean covers ~71% of Earth's surface.\n"
        "• Average depth is 3.7 km.\n"
        "• It holds ~97% of all Earth's water.\n"
        "At your service.",
    ),
    (
        "Give me a fun fact.",
        "• Honey never spoils — archaeologists found 3,000-year-old edible "
        "honey in Egyptian tombs.\n"
        "At your service.",
    ),
]

TEST_PROMPT = "Who are you and what can you help me with?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StopOnPhrase(StoppingCriteria):
    """
    Halt generation as soon as a target phrase appears in the decoded output.
    This lets us stop on a semantic signal ("At your service") rather than
    relying on a fixed token budget or EOS token.
    """
    def __init__(self, tokenizer, stop_phrase: str):
        self.tokenizer = tokenizer
        self.stop_phrase = stop_phrase

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Decode only the newly generated tokens (last 20 is enough to catch the phrase)
        decoded = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        return self.stop_phrase in decoded


def format_prompt(instruction: str, response: str = "") -> str:
    """Wrap an instruction (and optional response) in a plain template."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 80) -> str:
    """Run greedy decoding and return only the newly generated text."""
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Stop as soon as "At your service" appears, regardless of trailing punctuation.
    # max_new_tokens acts as a safety cap if the phrase never appears (e.g. before training).
    stopping_criteria = StoppingCriteriaList([StopOnPhrase(tokenizer, "At your service")])

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # greedy — deterministic output
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,               # penalises already-seen tokens; prevents looping
            no_repeat_ngram_size=4,               # blocks any 4-gram from appearing twice
            stopping_criteria=stopping_criteria,
        )

    # Slice off the prompt tokens so we only decode the model's new output
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def train(model, tokenizer, data: list[tuple], epochs: int, lr: float = 3e-4):
    """
    Supervised Fine-Tuning loop.

    Each example is formatted as a full prompt+response string. The model
    is trained to predict every token (standard causal LM objective).
    Loss going down across epochs means the model is learning the pattern.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        epoch_loss = 0.0

        for instruction, response in data:
            full_text = format_prompt(instruction, response)
            tokens = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(model.device)

            # Setting labels=input_ids tells the model to predict all tokens.
            # The loss is cross-entropy averaged over the sequence length.
            outputs = model(**tokens, labels=tokens["input_ids"])

            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()

            epoch_loss += outputs.loss.item()

        avg_loss = epoch_loss / len(data)
        print(f"  Epoch {epoch + 1}/{epochs}  —  avg loss: {avg_loss:.4f}  [{elapsed(t_epoch)}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimalist SFT fine-tuning demo")
    parser.add_argument(
        "--model", default="tinyllama",
        help="Ollama model name to fine-tune (default: tinyllama)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs (default: 5)",
    )
    # parse_known_args() silently ignores unrecognised arguments.
    # This is necessary in Jupyter/Colab where sys.argv contains kernel flags
    # like `-f kernel-xxx.json` that argparse would otherwise reject.
    args, _ = parser.parse_known_args()

    hf_id = OLLAMA_TO_HF.get(args.model)
    if hf_id is None:
        print(f"Warning: '{args.model}' not in known models {list(OLLAMA_TO_HF)}.")
        print("Falling back to tinyllama.\n")
        hf_id = OLLAMA_TO_HF["tinyllama"]

    device, platform_label = detect_platform()

    # Raspberry Pi 5 has 4 Cortex-A76 cores. PyTorch may default to fewer threads
    # on ARM Linux; setting explicitly gets the most out of the CPU.
    if platform_label.startswith("Raspberry Pi"):
        torch.set_num_threads(4)          # intra-op: linear algebra, convolutions
        torch.set_num_interop_threads(1)  # inter-op: keep low to avoid scheduling overhead
        print(f"Raspberry Pi detected — PyTorch threads set to 4 intra / 1 inter")

    print(f"Model    : {hf_id}")
    print(f"Platform : {platform_label}")
    print(f"Device   : {device}")
    print(f"Epochs   : {args.epochs}")
    # dtype printed after model load since it depends on device

    t_total = time.perf_counter()

    # --- Load tokenizer and base model ---
    print("\nLoading model (downloads from HuggingFace on first run)...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # required for batched decoding

    # CPU:                    float32 (no hardware float16 on x86/ARM CPUs)
    # CUDA Ampere+ (sm >= 80): bfloat16 — native Tensor Core support, no overflow risk
    #   A100 = sm_80, H100 = sm_90, G4 = sm_89 (Ada Lovelace)
    # CUDA older (T4 = sm_75): float16 — T4 has no native bfloat16; is_bf16_supported()
    #   returns True on newer PyTorch via software emulation, which is slower
    # MPS:                    float16 — Apple Silicon metal shaders accelerate float16
    if device == "cpu":
        dtype = torch.float32
    elif device == "cuda":
        major = torch.cuda.get_device_properties(0).major  # compute capability major version
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float16
    # torch_dtype works on all transformers versions.
    # Newer versions (>=4.47) show a deprecation warning — suppress it to keep output clean.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    model = model.to(device)
    t_load = time.perf_counter() - t0
    print(f"  done  [{t_load:.1f}s]  (dtype: {dtype})")

    # --- BEFORE ---
    print(f"\n{'='*55}")
    print("BEFORE fine-tuning")
    print(f"{'='*55}")
    print(f"Prompt: {TEST_PROMPT}\n")
    t0 = time.perf_counter()
    before = generate(model, tokenizer, TEST_PROMPT)
    t_before = time.perf_counter() - t0
    print(f"Response:\n{before}")
    print(f"  [{t_before:.1f}s]\n")

    # --- Attach LoRA adapter ---
    #
    # LoRA decomposes a weight update ΔW into two small matrices: A (d×r) and B (r×d).
    # During the forward pass: output += (B @ A) * (lora_alpha / r)
    # Only A and B are trained; the original weight W stays frozen.
    #
    # target_modules: which linear layers to adapt.
    # q_proj and v_proj (query and value projections) are the standard choice
    # for Llama-family models (TinyLlama, Mistral share this architecture).
    t0 = time.perf_counter()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,               # rank: smaller = fewer params, larger = more expressive
        lora_alpha=16,     # scaling factor; effective update scale = lora_alpha / r
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows how few params are actually trained
    t_lora = time.perf_counter() - t0
    print(f"  LoRA setup [{t_lora:.1f}s]")

    # --- Fine-tune ---
    print(f"\nTraining on {len(TRAINING_DATA)} examples for {args.epochs} epochs...")
    t0 = time.perf_counter()
    train(model, tokenizer, TRAINING_DATA, epochs=args.epochs)
    t_train = time.perf_counter() - t0
    print(f"  total training [{t_train:.1f}s]")

    # --- AFTER ---
    print(f"\n{'='*55}")
    print("AFTER fine-tuning")
    print(f"{'='*55}")
    print(f"Prompt: {TEST_PROMPT}\n")
    t0 = time.perf_counter()
    after = generate(model, tokenizer, TEST_PROMPT)
    t_after = time.perf_counter() - t0
    print(f"Response:\n{after}")
    print(f"  [{t_after:.1f}s]\n")

    # --- Side-by-side summary ---
    print(f"{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"BEFORE:\n{before}")
    print(f"\nAFTER:\n{after}")

    print(f"\n{'='*55}")
    print("TIMING")
    print(f"{'='*55}")
    print(f"  Platform           : {platform_label}")
    print(f"  Model load         : {t_load:.1f}s")
    print(f"  Inference (before) : {t_before:.1f}s")
    print(f"  LoRA setup         : {t_lora:.1f}s")
    print(f"  Training           : {t_train:.1f}s  ({t_train/args.epochs:.1f}s/epoch)")
    print(f"  Inference (after)  : {t_after:.1f}s")
    print(f"  {'─'*30}")
    print(f"  Total              : {elapsed(t_total)}")


if __name__ == "__main__":
    main()
