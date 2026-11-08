"""Task 4 tests + benchmark.   python -m src.test_attention"""
import json
import statistics as st
import time
from pathlib import Path

import torch

from src.attention import reference_attention, sdpa, tiled_attention

dev = "cuda" if torch.cuda.is_available() else "cpu"
TOL = {torch.float64: 1e-9, torch.float32: 1e-4, torch.float16: 1e-2}
# (B, H, Lq, Lk, D, causal, block_q, block_k, q_scale, note)
CASES = [
    (1, 1, 4, 4, 2, True, 2, 2, 1.0, "tiny"),
    (2, 3, 64, 64, 32, True, 16, 16, 1.0, "causal, divisible"),
    (1, 4, 100, 100, 64, True, 32, 32, 1.0, "causal, length not divisible by block"),
    (1, 2, 37, 129, 16, True, 16, 32, 1.0, "Lq<Lk (queries are last positions)"),
    (2, 2, 128, 128, 64, False, 32, 64, 1.0, "non-causal"),
    (1, 2, 1, 256, 64, True, 32, 32, 1.0, "single query = one KV-cache decode step"),
    (1, 2, 256, 256, 64, True, 64, 64, 10.0, "large scores (x10)"),
    (1, 12, 512, 512, 128, True, 128, 128, 1.0, "model-like (12 heads, D=128)"),
]

def err(a, b):
    d = (a.double() - b.double()).abs()
    return d.max().item(), d.mean().item()

def tiny_examples():
    print("\n--- hand-checkable example (L=4, D=2, float64) ---")
    q = torch.tensor([[[[1., 0.], [0., 1.], [1., 1.], [.5, -.5]]]], dtype=torch.float64, device=dev)
    k = q.clone()
    v = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]]], dtype=torch.float64, device=dev)
    ref = reference_attention(q, k, v, True)
    til = tiled_attention(q, k, v, True, block_q=2, block_k=2)
    print("reference:\n", ref[0, 0].cpu().numpy(), "\ntiled:\n", til[0, 0].cpu().numpy())
    assert torch.allclose(ref[..., 0, :], v[..., 0, :]), "row 0 can only see token 0"
    assert (ref - til).abs().max() < 1e-12
    k2, v2 = k.clone(), v.clone()
    k2[..., 3, :] += 5
    v2[..., 3, :] += 100
    til2 = tiled_attention(q, k2, v2, True, block_q=2, block_k=2)
    assert torch.allclose(til[..., :3, :], til2[..., :3, :]), "future tokens leaked into the past"
    print("OK: row 0 == v[0]; changing the LAST token does not change earlier outputs (causality)")

def stability_demo():
    print("\n--- numerical stability demo (fp32, huge scores) ---")
    torch.manual_seed(1)
    q = torch.randn(1, 1, 64, 64, device=dev) * 60
    k = torch.randn(1, 1, 64, 64, device=dev)
    v = torch.randn(1, 1, 64, 64, device=dev)
    s = torch.matmul(q, k.transpose(-1, -2)) / 8.0
    mask = torch.triu(torch.ones(64, 64, dtype=torch.bool, device=dev), 1)
    e = torch.exp(s.masked_fill(mask, float("-inf")))          # NO max subtraction
    naive = torch.matmul(e / e.sum(-1, keepdim=True), v)
    til = tiled_attention(q, k, v, True, 16, 16)
    gold = reference_attention(q.double(), k.double(), v.double(), True)
    res = {"naive_has_nan_or_inf": bool((~torch.isfinite(naive)).any()),
           "tiled_has_nan_or_inf": bool((~torch.isfinite(til)).any()),
           "tiled_max_err_vs_fp64": err(til, gold)[0], "max_score": s.max().item()}
    print(res)
    assert res["naive_has_nan_or_inf"] and not res["tiled_has_nan_or_inf"]
    return res

def correctness():
    print("\n--- correctness vs float64 gold (max / mean abs error) ---")
    rows, ok = [], True
    for (B, H, Lq, Lk, D, causal, bq, bk, qs, note) in CASES:
        torch.manual_seed(0)
        base = [torch.randn(B, H, L, D, dtype=torch.float64, device=dev) for L in (Lq, Lk, Lk)]
        base[0] = base[0] * qs
        for dt in (torch.float64, torch.float32, torch.float16):
            if dev == "cpu" and dt == torch.float16:
                continue
            q, k, v = [x.to(dt) for x in base]            # inputs rounded to dt first
            gold = reference_attention(q.double(), k.double(), v.double(), causal)
            ref = reference_attention(q, k, v, causal)
            til = tiled_attention(q, k, v, causal, bq, bk)
            r = {"note": note, "B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D, "causal": causal,
                 "block": [bq, bk], "q_scale": qs, "dtype": str(dt).split(".")[-1],
                 "ref_max": err(ref, gold)[0], "tiled_max": err(til, gold)[0],
                 "tiled_mean": err(til, gold)[1], "tiled_vs_ref_max": err(til, ref)[0]}
            if Lq == Lk and dev == "cuda" and dt != torch.float64:
                r["sdpa_max"] = err(sdpa(q, k, v, causal), gold)[0]
            r["pass"] = r["tiled_max"] < TOL[dt]
            ok &= r["pass"]
            rows.append(r)
            print(f"{note[:38]:38s} {r['dtype']:8s} tiled max={r['tiled_max']:.2e} "
                  f"mean={r['tiled_mean']:.2e} | ref max={r['ref_max']:.2e} "
                  f"| tiled-vs-ref={r['tiled_vs_ref_max']:.2e} | {'PASS' if r['pass'] else 'FAIL'}")
    print("ALL CORRECTNESS CASES PASSED" if ok else "SOME CASES FAILED")
    return rows

def bench(fn, repeats=5):
    fn(); torch.cuda.synchronize()                         # warm-up
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base, times = torch.cuda.memory_allocated(), []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return st.median(times) * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**20

def benchmark():
    print("\n--- speed / memory: fp16, causal, B=1, H=12, D=128 ---")
    rows = []
    for L in (256, 512, 1024, 2048, 4096):
        q, k, v = [torch.randn(1, 12, L, 128, dtype=torch.float16, device="cuda") for _ in range(3)]
        impls = {"reference": lambda: reference_attention(q, k, v, True),
                 "tiled_python(128)": lambda: tiled_attention(q, k, v, True, 128, 128),
                 "sdpa_math": lambda: sdpa(q, k, v, True, "math"),
                 "sdpa_mem_efficient": lambda: sdpa(q, k, v, True, "mem_efficient"),
                 "sdpa_flash": lambda: sdpa(q, k, v, True, "flash")}
        for name, fn in impls.items():
            try:
                ms, mib = bench(fn)
                rows.append({"L": L, "impl": name, "ms": ms, "peak_mib": mib})
                print(f"L={L:5d} {name:20s} {ms:9.2f} ms  {mib:8.1f} MiB")
            except (torch.OutOfMemoryError, RuntimeError) as e:
                rows.append({"L": L, "impl": name, "ms": None, "peak_mib": None,
                             "error": str(e)[:120]})
                print(f"L={L:5d} {name:20s} unavailable: {str(e)[:80]}")
            torch.cuda.empty_cache()
    return rows

def main():
    out = Path("outputs/attention"); out.mkdir(parents=True, exist_ok=True)
    tiny_examples()
    stab = stability_demo()
    corr = correctness()
    res = {"device": dev, "stability": stab, "correctness": corr}
    if dev == "cuda":
        res["gpu"] = torch.cuda.get_device_name(0)
        res["benchmark"] = benchmark()
    (out / "attention_results.json").write_text(json.dumps(res, indent=2))
    if dev == "cuda":
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for name in dict.fromkeys(r["impl"] for r in res["benchmark"]):
            pts = [(r["L"], r["ms"], r["peak_mib"]) for r in res["benchmark"]
                   if r["impl"] == name and r["ms"] is not None]
            if pts:
                ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name)
                ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=name)
        for a, yl in zip(ax, ("time (ms)", "peak extra memory (MiB)")):
            a.set_xscale("log", base=2); a.set_yscale("log")
            a.set_xlabel("sequence length"); a.set_ylabel(yl); a.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(out / "attention_benchmark.png", dpi=200)

if __name__ == "__main__":
    main()
