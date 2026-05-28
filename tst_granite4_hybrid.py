"""Compare a Mamba2+Attention+MoE hybrid against a plain dense transformer.

A standard transformer's feed-forward layer is "one expert" — every token pays
the full FFN cost. An MoE replaces that single FFN with N experts and a router
that picks K of them per token, so capacity (total params) decouples from
compute (active params per token).

Models (same Granite 4.0 family, same tokenizer lineage):
  - ibm-granite/granite-4.0-micro  : 3B dense transformer (single-expert FFN)
  - ibm-granite/granite-4.0-h-tiny : 7B total / ~1B active, Mamba2+Attn+MoE

Expected takeaway: the hybrid MoE carries ~2x the knowledge (params) yet runs
faster per token because only a fraction of experts fire on each step.
"""
import gc
import platform
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DENSE_ID = "ibm-granite/granite-4.0-micro"
MOE_ID = "ibm-granite/granite-4.0-h-tiny"
PROMPT = "In two sentences, explain why mixing Mamba2 with attention layers helps."
MAX_NEW_TOKENS = 80
DEVICE = "mps" if platform.machine() == "arm64" and torch.backends.mps.is_available() else "cpu"


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def active_params(model, config):
    """Dense params + (experts_per_tok / num_experts) * expert params.

    Detect expert tensors by shape: Granite MoE fuses experts into a single
    tensor whose leading dim equals num_local_experts.
    """
    n_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
    n_active = getattr(config, "num_experts_per_tok", None)
    if not n_experts or not n_active:
        return total_params(model)
    expert, other = 0, 0
    for p in model.parameters():
        if p.dim() >= 1 and p.shape[0] == n_experts:
            expert += p.numel()
        else:
            other += p.numel()
    return other + int(expert * n_active / n_experts)


def run(model_id, label):
    print(f"\n=== {label} ===")
    print(f"  id:            {model_id}")
    print(f"  device:        {DEVICE}")
    tok = AutoTokenizer.from_pretrained(model_id)
    # Load on CPU first, then move to DEVICE. Avoids transformers' MPS path
    # trying to pre-allocate a single ~14 GB buffer (Metal rejects buffers
    # that large on most M-series chips).
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    cfg = mdl.config

    n_total = total_params(mdl)
    n_active = active_params(mdl, cfg)
    n_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    n_active_experts = getattr(cfg, "num_experts_per_tok", None)
    routing = f"{n_active_experts}/{n_experts} experts per token" if n_experts else "single expert (dense FFN)"
    print(f"  arch:          {type(mdl).__name__}  |  {routing}")
    print(f"  total params:  {n_total / 1e9:.2f}B")
    print(f"  active params: {n_active / 1e9:.2f}B  ({n_active / n_total:.0%} of total)")

    inputs = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(mdl.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    dt = time.perf_counter() - t0

    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    tps = new_tokens / dt
    print(f"  speed:         {new_tokens} tok in {dt:.1f}s  ({tps:.2f} tok/s)")
    print(f"  output:        {text}")

    del mdl, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return {"label": label, "total": n_total, "active": n_active, "tps": tps}


def main():
    results = [
        run(DENSE_ID, "Dense transformer (baseline, single FFN)"),
        run(MOE_ID, "Hybrid Mamba2+Attn+MoE"),
    ]
    print("\n=== Comparison ===")
    print(f"{'model':<45} {'total':>8} {'active':>8} {'tok/s':>8}")
    for r in results:
        print(f"{r['label']:<45} {r['total']/1e9:>7.2f}B {r['active']/1e9:>7.2f}B {r['tps']:>8.2f}")
    dense, moe = results
    print(
        f"\nMoE has {moe['total']/dense['total']:.1f}x the capacity "
        f"but only {moe['active']/dense['active']:.2f}x the per-token compute, "
        f"and runs at {moe['tps']/dense['tps']:.2f}x the dense throughput."
    )


if __name__ == "__main__":
    main()


# Run 1 — Intel Core i7 (6-core, 32 GB RAM), CPU, bfloat16, greedy decode.
# 32 GB easily holds both models; no memory pressure.
#
# === Dense transformer (granite-4.0-micro, 3.4B, all-attention) ===
#   speed: 63 tok in 138.8s  (0.45 tok/s)
#
# === Hybrid Mamba2+Attn+MoE (granite-4.0-h-tiny, 6.94B / 1.46B active) ===
#   speed: 68 tok in 107.1s  (0.64 tok/s)
#
# Hybrid: 2.0x capacity, 0.43x active compute, 1.40x dense throughput.
#
# ---
#
# Run 2 — Apple M4 (16 GB unified memory), MPS, bfloat16, greedy decode.
# Caveat: hybrid is ~14 GB in bf16 vs 16 GB total RAM — earlier runs OOM'd,
# this one only completed after killing other processes. Memory pressure
# during the hybrid run was real and unmeasured.
#
# === Dense (3.4B) ===   speed: 63 tok in 6.0s    (10.55 tok/s)
# === Hybrid (7B) ===    speed: 68 tok in 103.8s  (0.65 tok/s)
#
# What we can say with confidence:
#   - Dense attention accelerates massively on M4 MPS (23x vs Intel CPU).
#     This is a clean measurement — model fits comfortably in 16 GB.
#   - The MoE *architectural* claim (capacity decoupled from compute) is
#     visible on the Intel run: 2x params, ~half the active compute,
#     ~1.4x faster end-to-end. Independent of the Mamba2 question.
#
# What this experiment does NOT settle:
#   - Why the M4 hybrid run was slow. Plausible causes, not isolated:
#       (a) Mamba2's SSM scan has no fused kernel on MPS (CUDA-only today
#           via mamba_ssm / causal_conv1d), so it runs a naive PyTorch
#           path. Sequential scans tend to be Python/launch-overhead bound,
#           which GPU acceleration may not help.
#       (b) The 14 GB model on a 16 GB machine likely paged or thrashed
#           the MPS allocator, costing real time.
#       (c) Some mix of (a) and (b).
#   - Whether MoE *alone* (without Mamba2) speeds up on MPS — we never
#     tested that; both granite-4.0 variants share the same code path.
#
# Follow-up experiments that would actually resolve this:
#   1. Run granite-4.0-h-tiny on a 32+ GB M-series Mac to remove memory
#      pressure. If still ~1 tok/s, kernels are the bottleneck, not RAM.
#   2. Run a non-Mamba MoE (e.g. Mixtral, Qwen-MoE) on M4 MPS to see if
#      MoE itself accelerates when SSM is out of the picture.
#   3. Run granite-4.0-h-tiny on an NVIDIA GPU with mamba_ssm installed
#      to confirm the fused SSM path is what closes the gap.
