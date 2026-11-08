"""Token-length stats + truncation risk for the chosen max length."""
import json
import numpy as np
from transformers import AutoTokenizer
from src.prompts import build_messages, format_target

MODEL, MAX_LEN = "Qwen/Qwen2.5-1.5B-Instruct", 2048


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    for split in ("train", "val", "test"):
        lens, over = [], 0
        for line in open(f"data/processed/{split}.jsonl"):
            r = json.loads(line)
            sents = {int(k): v for k, v in r["sentences"].items()}
            prompt = tok.apply_chat_template(
                build_messages(sents, r["criterion_type"], r["criterion_text"]),
                tokenize=False, add_generation_prompt=True)
            n = len(tok(prompt + format_target(r["gold_label"], r["gold_evidence"])).input_ids)
            lens.append(n)
            over += n > MAX_LEN
        a = np.array(lens)
        print(f"{split}: n={len(a)} mean={a.mean():.0f} p95={np.percentile(a,95):.0f} "
              f"max={a.max()} over_{MAX_LEN}={over}")


if __name__ == "__main__":
    main()