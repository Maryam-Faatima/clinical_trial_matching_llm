"""Task 3 / D2: is cached greedy == uncached greedy? Plus numerical-drift analysis.
  python -m src.test_kv                 # fp16
  python -m src.test_kv --dtype fp32    # shows mismatches disappear in higher precision
"""
import argparse
import json
from pathlib import Path

import torch

from src.kv_decode import (generate_cached, generate_uncached, load, synthetic_ids,
                           teacher_forced_logits_cached, teacher_forced_logits_full)
from src.prompts import build_messages

CASES = [(64, 32), (256, 64), (512, 128), (1024, 128)]  # (input tokens, new tokens)

def compare(model, ids, max_new, eos_id):
    ref = generate_uncached(model, ids, max_new, eos_id)["tokens"]
    fast = generate_cached(model, ids, max_new, eos_id)["tokens"]
    n = min(len(ref), len(fast))
    first = next((i for i in range(n) if ref[i] != fast[i]), None)
    if first is None and len(ref) != len(fast):
        first = n
    a = teacher_forced_logits_full(model, ids, ref)
    b = teacher_forced_logits_cached(model, ids, ref)
    d = (a - b).abs()
    top2 = a.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    res = {"equal": first is None, "n_tokens": len(ref), "first_mismatch": first,
           "max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
           "argmax_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
           "min_top2_gap_over_steps": gap.min().item()}
    if first is not None and first < len(ref):
        res["top2_gap_at_first_mismatch"] = gap[first].item()
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/lora_r16/epoch_3")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--n-real", type=int, default=30)
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tok, model = load(args.adapter, dtype)

    results = {"dtype": args.dtype, "synthetic": [], "real": []}
    print(f"\n--- synthetic inputs, fixed length (no EOS stop), {args.dtype} ---")
    for n_in, n_out in CASES:
        r = compare(model, synthetic_ids(tok, n_in), n_out, None)
        r.update(input_len=n_in, new_tokens=n_out)
        results["synthetic"].append(r)
        print(r)

    print(f"\n--- {args.n_real} real validation prompts (stop at EOS), {args.dtype} ---")
    rows = [json.loads(l) for l in open("data/processed/val.jsonl")][:args.n_real]
    for row in rows:
        prompt = tok.apply_chat_template(
            build_messages({int(k): v for k, v in row["sentences"].items()},
                           row["criterion_type"], row["criterion_text"]),
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        r = compare(model, ids, 64, tok.eos_token_id)
        r.update(annotation_id=row["annotation_id"], input_len=ids.shape[1])
        results["real"].append(r)
    eq = sum(r["equal"] for r in results["real"])
    print(f"real prompts identical: {eq}/{len(results['real'])}")
    for r in results["real"]:
        if not r["equal"]:
            print("MISMATCH:", r)
    s = sum(r["equal"] for r in results["synthetic"])
    print(f"synthetic identical: {s}/{len(CASES)}")

    out = Path("outputs/kv"); out.mkdir(parents=True, exist_ok=True)
    (out / f"correctness_{args.dtype}.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
