"""Task 3 / D3: benchmark cached vs uncached (warm-up, synchronized, repeated)."""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from src.kv_decode import generate_cached, generate_uncached, kv_bytes_per_token, load, synthetic_ids

CONFIGS = [(128, 32), (128, 128), (512, 32), (512, 128), (1024, 32), (1024, 128)]
PATHS = {"uncached": generate_uncached,
         "cached": lambda m, i, n, e: generate_cached(m, i, n, e, check=False)}

def run(fn, model, ids, n_out):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    r = fn(model, ids, n_out, None)          # no EOS stop: exactly n_out tokens
    r["peak_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/lora_r16/epoch_3")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    tok, model = load(args.adapter)
    per_tok = kv_bytes_per_token(model.config)
    rows = []
    for n_in, n_out in CONFIGS:
        ids = synthetic_ids(tok, n_in)
        for fn in PATHS.values():            # warm-up (kernel selection, allocator)
            run(fn, model, ids, 4)
        row = {"input_len": n_in, "new_tokens": n_out,
               "kv_theory_mib": per_tok * (n_in + n_out) / 2**20}
        toks = {}
        for name, fn in PATHS.items():
            runs = [run(fn, model, ids, n_out) for _ in range(args.repeats)]
            toks[name] = runs[0]["tokens"]
            tot = [r["prefill_s"] + r["decode_s"] for r in runs]
            row[name] = {
                "prefill_ms": st.mean(r["prefill_s"] for r in runs) * 1e3,
                "decode_ms": st.mean(r["decode_s"] for r in runs) * 1e3,
                "total_tps": n_out / st.mean(tot),
                "total_tps_std": st.pstdev([n_out / t for t in tot]),
                "decode_tps": (n_out - 1) / st.mean(r["decode_s"] for r in runs),
                "peak_mib": max(r["peak_mib"] for r in runs)}
        row["tokens_equal"] = toks["cached"] == toks["uncached"]
        row["speedup"] = row["cached"]["total_tps"] / row["uncached"]["total_tps"]
        rows.append(row)
        print(f"in={n_in:5d} out={n_out:4d} | uncached {row['uncached']['total_tps']:7.1f} tok/s"
              f" | cached {row['cached']['total_tps']:7.1f} tok/s | speedup {row['speedup']:.1f}x"
              f" | peak MiB u/c {row['uncached']['peak_mib']:.0f}/{row['cached']['peak_mib']:.0f}"
              f" | equal={row['tokens_equal']}")

    out = Path("outputs/kv"); out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(
        {"gpu": torch.cuda.get_device_name(0), "dtype": "fp16", "repeats": args.repeats,
         "rows": rows}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['input_len']}+{r['new_tokens']}" for r in rows]
    x, w = range(len(rows)), 0.38
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (name, off) in enumerate((("uncached", -w / 2), ("cached", w / 2))):
        ax[0].bar([i + off for i in x], [r[name]["total_tps"] for r in rows], w, label=name)
        ax[1].bar([i + off for i in x], [r[name]["peak_mib"] for r in rows], w, label=name)
    ax[0].set_yscale("log"); ax[0].set_ylabel("tokens/s (log)")
    ax[1].set_ylabel("peak extra memory (MiB)")
    for a in ax:
        a.set_xticks(list(x)); a.set_xticklabels(labels, rotation=30)
        a.set_xlabel("input + output tokens"); a.legend()
    plt.tight_layout(); plt.savefig(out / "kv_benchmark.png", dpi=200)

if __name__ == "__main__":
    main()
