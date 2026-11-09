"""Figures + error analysis from saved evaluation outputs (CPU only, no GPU needed).
  python -m src.analyze --split val  --base base_val  --ft ft_e3_val
  python -m src.analyze --split test --base base_test --ft ft_e3_test
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EV = Path("outputs/eval")
SUBSETS = ("overall", "inclusion", "exclusion")

def load(name):
    m = json.load(open(EV / f"{name}_metrics.json", encoding="utf-8"))["summary"]
    p = [json.loads(l) for l in open(EV / f"{name}_predictions.jsonl", encoding="utf-8")]
    return m, p

def cm_panel(ax, c, title):
    cm = np.array(c["confusion_matrix"])
    labels = c["confusion_labels"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("gold", fontsize=8)

def categorize(p):
    """Error type of one prediction, or None if fully correct."""
    g, y = p["gold_label"], p["pred_label"]
    if not p["valid"]:
        return "invalid_output"
    if g == y:
        return ("right_label_wrong_evidence"
                if set(p["gold_evidence"]) != set(p["pred_evidence"]) else None)
    if g == "not enough information":
        return "missing_info_treated_as_proof"
    if y == "not enough information":
        return "false_insufficient_information"
    if "not applicable" in (g, y):
        return "not_applicable_confusion"
    return "wrong_decision_polarity"

def sent_text(sentences, ids):
    return "; ".join(f"[{i}] {sentences.get(str(i), '?')}" for i in ids) or "(none)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--per-category", type=int, default=3)
    a = ap.parse_args()
    out = Path("outputs/analysis")
    out.mkdir(parents=True, exist_ok=True)
    mb, pb = load(a.base)
    mf, pf = load(a.ft)
    tag = a.split

    # 1) confusion matrices: row 1 base, row 2 fine-tuned
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for r, (m, nm) in enumerate(((mb, "base"), (mf, "fine-tuned"))):
        for c, s in enumerate(SUBSETS):
            cm_panel(ax[r][c], m[s]["classification"], f"{nm}: {s} ({tag})")
    plt.tight_layout()
    plt.savefig(out / f"confusion_{tag}.png", dpi=200)
    plt.close()

    # 2) before/after bars
    metrics = [("accuracy", lambda m, s: m[s]["classification"]["accuracy"]),
               ("macro-F1", lambda m, s: m[s]["classification"]["macro_f1"]),
               ("evidence micro-F1", lambda m, s: m[s]["evidence"]["micro_f1"])]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for i, s in enumerate(SUBSETS):
        x = np.arange(len(metrics))
        ax[i].bar(x - 0.2, [f(mb, s) for _, f in metrics], 0.4, label="base")
        ax[i].bar(x + 0.2, [f(mf, s) for _, f in metrics], 0.4, label="fine-tuned")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels([n for n, _ in metrics], rotation=20, fontsize=8)
        ax[i].set_title(f"{s} ({tag})", fontsize=9)
        ax[i].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / f"before_after_{tag}.png", dpi=200)
    plt.close()

    # 3) per-class F1 (overall)
    pcb, pcf = mb["overall"]["classification"]["per_class"], mf["overall"]["classification"]["per_class"]
    labels = sorted(set(pcb) | set(pcf))
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 3.6))
    plt.bar(x - 0.2, [pcb.get(l, {}).get("f1", 0) for l in labels], 0.4, label="base")
    plt.bar(x + 0.2, [pcf.get(l, {}).get("f1", 0) for l in labels], 0.4, label="fine-tuned")
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"per_class_f1_{tag}.png", dpi=200)
    plt.close()

    # 4) error cases (fine-tuned model), joined with note sentences
    notes = {}
    for line in open(f"data/processed/{a.split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        notes[r["annotation_id"]] = r["sentences"]
    cats = {}
    for p in pf:
        c = categorize(p)
        if c:
            cats.setdefault(c, []).append(p)
    counts = {c: len(v) for c, v in cats.items()}
    rng = random.Random(42)
    picked = []
    for c, v in sorted(cats.items()):
        picked += [(c, p) for p in rng.sample(v, min(a.per_category, len(v)))]

    md = [f"# Error analysis ({a.ft} on {a.split})\n",
          f"Total errors by type: {json.dumps(counts)}\n",
          f"Cases shown: {len(picked)} (up to {a.per_category} per type, random seed 42)\n"]
    for k, (c, p) in enumerate(picked, 1):
        s = notes[p["annotation_id"]]
        md.append(f"\n## Case {k}: {c}\n- annotation_id: {p['annotation_id']} | patient: {p['patient_id']}"
                  f" | type: {p['criterion_type']}\n- criterion: {p['criterion_text']}\n"
                  f"- gold label: **{p['gold_label']}** | predicted: **{p['pred_label']}**\n"
                  f"- gold evidence: {sent_text(s, p['gold_evidence'])}\n"
                  f"- predicted evidence: {sent_text(s, p['pred_evidence'])}\n"
                  f"- why it matters: (write 1-2 sentences here)\n")
    (out / f"error_cases_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    (out / f"error_counts_{tag}.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("error types:", counts)
    print(f"{len(picked)} distinct cases written to {out / f'error_cases_{tag}.md'}")
    if len(picked) < 8:
        print("WARNING: fewer than 8 cases; raise --per-category")

if __name__ == "__main__":
    main()
