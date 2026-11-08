"""Task 5: token-equality test + benchmark of speculative vs ordinary cached greedy decoding.
  python -m src.bench_spec --n 5 --ks 3 --repeats 1 --name smoke      # quick check first
  python -m src.bench_spec                                              # full run
  python -m src.bench_spec --dtype fp32 --n 20 --ks 4 --repeats 1 --name spec_fp32
"""
import argparse
import gc
import json
import os
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.kv_decode import generate_cached, synthetic_ids
from src.prompts import build_messages
from src.speculative import DRAFT_BASE, TARGET_BASE, load_model, speculative_generate

def workload_real(tok, n):
    """Real validation prompts: stop at EOS, max 64 new tokens (the actual task)."""
    items = []
    for line in list(open("data/processed/val.jsonl"))[:n]:
        r = json.loads(line)
        prompt = tok.apply_chat_template(
            build_messages({int(k): v for k, v in r["sentences"].items()},
                           r["criterion_type"], r["criterion_text"]),
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        items.append((ids, 64, tok.eos_token_id))
    return items

def workload_long(tok):
    """Synthetic longer prompts, exactly 96 new tokens (no EOS stop): steady-state behaviour."""
    return [(synthetic_ids(tok, L), 96, None) for L in (256, 768)]

def run_cfg(fn, items, repeats):
    fn(items[0][0], 8, items[0][2])                               # warm-up
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    pp = []
    for ids, max_new, eos in items:
        runs = [fn(ids, max_new, eos) for _ in range(repeats)]
        pp.append({"tokens": runs[0]["tokens"],
                   "lat": st.mean(r["total_s"] for r in runs), "first": runs[0]})
    return pp, torch.cuda.max_memory_allocated() / 2**20

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))

def summarize(pp, base_pp, peak):
    n_tok = sum(len(p["tokens"]) for p in pp)
    total = sum(p["lat"] for p in pp)
    mism = [(i, first_diff(p["tokens"], b["tokens"]))
            for i, (p, b) in enumerate(zip(pp, base_pp)) if p["tokens"] != b["tokens"]]
    s = {"n_prompts": len(pp), "tokens": n_tok, "total_s": total, "tok_per_s": n_tok / total,
         "ms_per_prompt": 1e3 * total / len(pp), "peak_mib": peak,
         "n_equal": len(pp) - len(mism), "mismatches": mism[:5]}
    stats = [p["first"].get("stats") for p in pp]
    if stats[0] is not None:
        rounds = sum(x["rounds"] for x in stats)
        drafted = sum(x["drafted"] for x in stats)
        accepted = sum(x["accepted"] for x in stats)
        s.update(rounds=rounds, drafted=drafted, accepted=accepted,
                 acceptance_rate=accepted / drafted if drafted else 0.0,
                 tokens_per_round=(n_tok - len(pp)) / rounds if rounds else 0.0,
                 draft_forwards=sum(x["draft_forwards"] for x in stats),
                 target_forwards=sum(x["target_forwards"] for x in stats),
                 draft_s=sum(p["first"]["draft_s"] for p in pp),
                 verify_s=sum(p["first"]["verify_s"] for p in pp),
                 prefill_s=sum(p["first"]["prefill_s"] for p in pp))
    return s

def show(w, label, k, s):
    extra = (f" | accept {s['acceptance_rate']:.2f} | tok/round {s['tokens_per_round']:.2f}"
             if "acceptance_rate" in s else "")
    print(f"[{w}] {label:22s} k={k} | equal {s['n_equal']}/{s['n_prompts']} | "
          f"{s['tok_per_s']:6.1f} tok/s | {s['ms_per_prompt']:7.0f} ms/prompt | "
          f"speedup {s.get('speedup', 1):.2f}x | peak {s['peak_mib']:.0f} MiB{extra}")
    if s["mismatches"]:
        print("    MISMATCH (prompt index, first differing position):", s["mismatches"])

def make_hf_fn(target, draft, k, pad_id):
    """Optional reference: Hugging Face assisted generation (greedy)."""
    draft.generation_config.num_assistant_tokens = k
    draft.generation_config.num_assistant_tokens_schedule = "constant"

    def fn(ids, max_new, eos):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = target.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                              max_new_tokens=max_new, do_sample=False, temperature=None,
                              top_p=None, top_k=None, repetition_penalty=1.0,
                              assistant_model=draft, eos_token_id=eos, pad_token_id=pad_id)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        toks = out[0, ids.shape[1]:].tolist()
        if eos is not None and eos in toks:
            toks = toks[:toks.index(eos) + 1]
        return {"tokens": toks, "total_s": dt}
    return fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ks", default="2,3,4,6")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--target-adapter", default="outputs/lora_r16/epoch_3")
    ap.add_argument("--draft-adapter", default="outputs/lora_draft/epoch_3")
    ap.add_argument("--hf-ref", action="store_true")
    ap.add_argument("--hf-k", type=int, default=4)
    ap.add_argument("--name", default="spec")
    args = ap.parse_args()
    if not os.path.isdir(args.target_adapter):
        raise SystemExit(f"target adapter not found: {args.target_adapter}")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    ks = [int(x) for x in args.ks.split(",")]

    tok = AutoTokenizer.from_pretrained(TARGET_BASE)
    work = {"A_real": workload_real(tok, args.n), "B_long": workload_long(tok)}
    target = load_model(TARGET_BASE, args.target_adapter, dtype)
    res = {"gpu": torch.cuda.get_device_name(0), "dtype": args.dtype, "repeats": args.repeats,
           "ks": ks, "target_weights_mib": torch.cuda.memory_allocated() / 2**20, "rows": []}

    def base_fn(ids, n, e):
        r = generate_cached(target, ids, n, e, check=False)
        r["total_s"] = r["prefill_s"] + r["decode_s"]
        return r

    base_pp, base_sum = {}, {}
    for w, items in work.items():
        pp, peak = run_cfg(base_fn, items, args.repeats)
        base_pp[w] = pp
        base_sum[w] = summarize(pp, pp, peak)
        res["rows"].append({"workload": w, "draft": "none (cached greedy)", "k": 0,
                            **base_sum[w], "speedup": 1.0})
        show(w, "baseline cached greedy", 0, base_sum[w])

    variants = [("self (oracle)", None), ("base 0.5B", (DRAFT_BASE, None))]
    if os.path.isdir(args.draft_adapter):
        variants.append(("fine-tuned 0.5B", (DRAFT_BASE, args.draft_adapter)))
    else:
        print("note: no fine-tuned draft adapter found, skipping that variant")

    for label, spec in variants:
        before = torch.cuda.memory_allocated()
        draft = target if spec is None else load_model(spec[0], spec[1], dtype)
        dmib = (torch.cuda.memory_allocated() - before) / 2**20
        for w, items in work.items():
            for k in ks:
                fn = lambda ids, n, e, k=k: speculative_generate(target, draft, ids, n, k, e)
                pp, peak = run_cfg(fn, items, args.repeats)
                s = summarize(pp, base_pp[w], peak)
                s["speedup"] = base_sum[w]["total_s"] / s["total_s"]
                res["rows"].append({"workload": w, "draft": label, "k": k,
                                    "draft_weights_mib": dmib, **s})
                show(w, label, k, s)
        if args.hf_ref and spec is not None:
            try:
                pp, peak = run_cfg(make_hf_fn(target, draft, args.hf_k, tok.eos_token_id),
                                   work["A_real"], args.repeats)
                s = summarize(pp, base_pp["A_real"], peak)
                s["speedup"] = base_sum["A_real"]["total_s"] / s["total_s"]
                res["rows"].append({"workload": "A_real", "draft": f"hf_assisted:{label}",
                                    "k": args.hf_k, **s})
                show("A_real", f"HF assisted ({label})", args.hf_k, s)
            except Exception as e:
                print("HF assisted reference failed:", str(e)[:200])
        if spec is not None:
            del draft; gc.collect(); torch.cuda.empty_cache()

    out = Path("outputs/speculative"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=2, default=str))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = dict.fromkeys(r["draft"] for r in res["rows"]
                           if r["k"] > 0 and not r["draft"].startswith("hf"))
    for label in labels:
        pts = sorted((r["k"], r["acceptance_rate"], r["speedup"]) for r in res["rows"]
                     if r["draft"] == label and r["workload"] == "A_real" and r["k"] > 0)
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=label)
        ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=label)
    ax[1].axhline(1.0, ls="--", c="gray")
    ax[0].set_ylabel("draft acceptance rate"); ax[1].set_ylabel("speedup vs cached greedy")
    for a in ax:
        a.set_xlabel("draft length k"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out / f"{args.name}.png", dpi=200)

if __name__ == "__main__":
    main()
