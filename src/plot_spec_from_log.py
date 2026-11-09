"""Rebuild the Task 5 table, JSON and figure from the saved console log.

The original JSON/PNG outputs of bench_spec.py were lost when the Kaggle session reset, so the
printed results were saved to logs/task5_speculative_benchmark.log.
    python -m src.plot_spec_from_log
(Re-running `python -m src.bench_spec --hf-ref` regenerates the JSON/PNG directly.)
"""
import json
import re
from pathlib import Path

LOG = Path("logs/task5_speculative_benchmark.log")
OUT = Path("outputs/speculative")
EXPECTED_ROWS = {"smoke": 8, "full_fp16": 28, "fp32": 8}
PAT = re.compile(
    r"^\[(\w+)\]\s+(.+?)\s+k=(\d+)\s+\|\s+equal (\d+)/(\d+)\s+\|\s+([\d.]+) tok/s"
    r"\s+\|\s+([\d.]+) ms/prompt\s+\|\s+speedup ([\d.]+)x\s+\|\s+peak (\d+) MiB"
    r"(?:\s+\|\s+accept ([\d.]+)\s+\|\s+tok/round ([\d.]+))?")


def parse(path=LOG):
    sections, cur = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("## section:"):
            cur = line.split(":", 1)[1].strip()
            sections[cur] = []
            continue
        m = PAT.match(line.strip())
        if m and cur:
            g = m.groups()
            sections[cur].append({
                "workload": g[0], "draft": g[1].strip(), "k": int(g[2]),
                "n_equal": int(g[3]), "n_prompts": int(g[4]),
                "tok_per_s": float(g[5]), "ms_per_prompt": float(g[6]),
                "speedup": float(g[7]), "peak_mib": int(g[8]),
                "acceptance_rate": float(g[9]) if g[9] else None,
                "tokens_per_round": float(g[10]) if g[10] else None})
    return sections


def main():
    sections = parse()
    for name, n in EXPECTED_ROWS.items():
        got = len(sections.get(name, []))
        assert got == n, f"section {name}: parsed {got} rows, expected {n} (log edited?)"
    allrows = [r for rows in sections.values() for r in rows]
    eq = sum(r["n_equal"] == r["n_prompts"] for r in allrows)
    print(f"token-equality: {eq}/{len(allrows)} configurations fully identical to cached greedy")

    rows = sections["full_fp16"]
    print("\nfull fp16 run (A_real: n=40 validation prompts)")
    print(f"{'draft':26s} {'k':>2s} {'accept':>7s} {'tok/round':>9s} {'tok/s':>6s} {'speedup':>8s}")
    for r in rows:
        if r["workload"] == "A_real":
            a = "-" if r["acceptance_rate"] is None else f"{r['acceptance_rate']:.2f}"
            t = "-" if r["tokens_per_round"] is None else f"{r['tokens_per_round']:.2f}"
            print(f"{r['draft']:26s} {r['k']:2d} {a:>7s} {t:>9s} {r['tok_per_s']:6.1f} {r['speedup']:8.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec_from_log.json").write_text(json.dumps(sections, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    drafts = ["self (oracle)", "base 0.5B", "fine-tuned 0.5B"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, w in enumerate(("A_real", "B_long")):
        for d in drafts:
            pts = sorted((x["k"], x["acceptance_rate"], x["speedup"]) for x in rows
                         if x["workload"] == w and x["draft"] == d and x["k"] > 0)
            ax[i][0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=d)
            ax[i][1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=d)
        if w == "A_real":
            for x in rows:
                if x["draft"].startswith("HF assisted"):
                    ax[0][1].scatter([x["k"]], [x["speedup"]], marker="x", s=60,
                                     label=x["draft"], zorder=5)
        ax[i][1].axhline(1.0, ls="--", c="gray")
        name = "real validation prompts" if w == "A_real" else "synthetic long prompts"
        ax[i][0].set_ylabel(f"acceptance rate\n({name})")
        ax[i][1].set_ylabel(f"speedup vs cached greedy\n({name})")
    for a in ax[1]:
        a.set_xlabel("draft length k")
    ax[0][0].legend(fontsize=7)
    ax[0][1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "spec_from_log.png", dpi=200)
    print("\nsaved", OUT / "spec_from_log.json", "and", OUT / "spec_from_log.png")


if __name__ == "__main__":
    main()
