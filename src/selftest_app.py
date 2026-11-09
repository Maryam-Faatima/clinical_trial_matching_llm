"""Quick end-to-end check of the REAL model engine used by the app (needs the model; GPU optional).

    python -m src.selftest_app                 # uses outputs/lora_r16/epoch_3 (downloads it if missing)
    python -m src.selftest_app --no-adapter    # base model only (testing)

Runs one validation example in all three decoding modes and checks that the generated tokens are identical.
"""
import argparse
import json

from src import screening as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/lora_r16/epoch_3")
    ap.add_argument("--draft-adapter", default="outputs/lora_draft/epoch_3")
    ap.add_argument("--no-adapter", action="store_true")
    args = ap.parse_args()

    adapter, source = None, "none (BASE MODEL ONLY)"
    if not args.no_adapter:
        source = S.ensure_adapter(args.adapter)
        adapter = args.adapter
    eng = S.Engine(S.TARGET_BASE, adapter, S.DRAFT_BASE, args.draft_adapter, source)
    print("device:", eng.device, "| dtype:", eng.dtype, "| adapter:", source)

    row = json.loads(open("data/processed/val.jsonl", encoding="utf-8").readline())
    sents = {int(k): v for k, v in row["sentences"].items()}
    print("criterion:", row["criterion_type"], "-", row["criterion_text"][:80], "| gold:", row["gold_label"])

    toks = {}
    for mode in S.MODES:
        r = eng.screen(sents, row["criterion_type"], row["criterion_text"], mode, 3)
        ids, _ = eng.build_ids(sents, row["criterion_type"], row["criterion_text"])
        toks[mode] = eng.generate(ids, mode, 3)["tokens"]
        print(f"[{mode}] label={r['label']} evidence={[e['id'] for e in r['evidence']]} "
              f"status={r['status']} time={r['latency_s']:.2f}s raw={r['raw_output']!r}")
    same = all(t == toks[S.MODE_REF] for t in toks.values())
    print("tokens identical across all three modes:", same)
    assert same, "decoding modes disagree"
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()