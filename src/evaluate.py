"""Evaluate base model or a LoRA adapter on val/test with fixed greedy decoding.

  python -m src.evaluate --split val --name base_val
  python -m src.evaluate --split val --adapter outputs/lora_r16/epoch_2 --name ft_e2_val
The test split is blocked unless --allow-test is passed (final evaluation only).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import print_summary, summarize
from src.prompts import build_messages, parse_output

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DECODING = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0)  # Qwen's default 1.05 would alter greedy output

def load_rows(split):
    return [json.loads(l) for l in open(f"data/processed/{split}.jsonl")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Test split is reserved for final evaluation. Pass --allow-test.")

    rows = load_rows(args.split)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    prompts = [tok.apply_chat_template(
        build_messages({int(k): v for k, v in r["sentences"].items()},
                       r["criterion_type"], r["criterion_text"]),
        tokenize=False, add_generation_prompt=True) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    raw = [None] * len(rows)

    t0 = time.time()
    for b in range(0, len(order), args.bs):
        idx = order[b:b + args.bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id, **DECODING)
        for i, g in zip(idx, out[:, enc.input_ids.shape[1]:]):
            raw[i] = tok.decode(g, skip_special_tokens=True)
    print(f"generated {len(rows)} outputs in {time.time() - t0:.1f}s")

    preds = [parse_output(t, r["criterion_type"]) for t, r in zip(raw, rows)]
    summary = summarize(rows, preds)
    print_summary(summary)

    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.name}_predictions.jsonl", "w") as f:
        for r, t, p in zip(rows, raw, preds):
            f.write(json.dumps({
                "annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
                "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
                "gold_label": r["gold_label"], "gold_evidence": r["gold_evidence"],
                "raw_output": t, "pred_label": p["label"],
                "pred_evidence": p["evidence"], "valid": p["valid"]}) + "\n")
    (out_dir / f"{args.name}_metrics.json").write_text(json.dumps({
        "model": MODEL, "adapter": args.adapter, "split": args.split,
        "decoding": {**{k: v for k, v in DECODING.items()},
                     "max_new_tokens": args.max_new_tokens, "batch_size": args.bs},
        "summary": summary}, indent=2))

if __name__ == "__main__":
    main()
