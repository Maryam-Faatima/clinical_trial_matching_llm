"""LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the train split (fp16, T4-friendly).

Loss is computed on the answer tokens only (prompt labels = -100). Each epoch saves an
adapter and a validation loss; pick the epoch by VALIDATION macro-F1 (src.evaluate).
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.prompts import build_messages, format_target

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = Path("outputs/lora_r16")
CFG = dict(seed=42, max_len=1024, epochs=3, lr=2e-4, micro_bs=1, grad_accum=16,
           warmup_ratio=0.1, weight_decay=0.0, max_grad_norm=1.0,
           lora_r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"])

def encode(rows, tok):
    data = []
    for r in rows:
        msgs = build_messages({int(k): v for k, v in r["sentences"].items()},
                              r["criterion_type"], r["criterion_text"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(format_target(r["gold_label"], r["gold_evidence"]) + tok.eos_token,
                    add_special_tokens=False).input_ids
        ids = p_ids + a_ids
        if len(ids) > CFG["max_len"]:  # fail loudly: never truncate evidence silently
            raise ValueError(f"example {r['annotation_id']} has {len(ids)} tokens")
        data.append({"input_ids": ids, "labels": [-100] * len(p_ids) + a_ids})
    return data

def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

@torch.no_grad()
def val_loss(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        tot += loss.item() * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    return tot / n

def main():
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    tok = AutoTokenizer.from_pretrained(MODEL)
    load = lambda s: [json.loads(l) for l in open(f"data/processed/{s}.jsonl")]
    train, val = encode(load("train"), tok), encode(load("val"), tok)
    col = lambda b: collate(b, tok.pad_token_id)
    g = torch.Generator().manual_seed(CFG["seed"])
    train_loader = DataLoader(train, batch_size=CFG["micro_bs"], shuffle=True,
                              collate_fn=col, generator=g)
    val_loader = DataLoader(val, batch_size=8, shuffle=False, collate_fn=col)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["target_modules"], task_type="CAUSAL_LM"))
    for p in model.parameters():          # fp16 AMP needs fp32 trainable params
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps_per_epoch = -(-len(train_loader) // CFG["grad_accum"])
    total = steps_per_epoch * CFG["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(OUT)
    log = {"config": CFG, "n_train": len(train), "n_val": len(val),
           "trainable_params": sum(p.numel() for p in params), "steps": [], "epochs": []}
    step, t0 = 0, time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        run = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / CFG["grad_accum"]
            scaler.scale(loss).backward()
            run += loss.item()
            if (i + 1) % CFG["grad_accum"] == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, CFG["max_grad_norm"])
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log["steps"].append({"step": step, "epoch": epoch, "train_loss": run,
                                     "lr": sched.get_last_lr()[0]})
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{total} loss {run:.4f}")
                run = 0.0
        vl = val_loss(model, val_loader)
        print(f"== epoch {epoch} val_loss {vl:.4f}  ({time.time() - t0:.0f}s)")
        log["epochs"].append({"epoch": epoch, "step": step, "val_loss": vl})
        model.save_pretrained(OUT / f"epoch_{epoch}")
        (OUT / "train_log.json").write_text(json.dumps(log, indent=2))
        model.train()
    log["peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    log["train_seconds"] = time.time() - t0
    (OUT / "train_log.json").write_text(json.dumps(log, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = log["steps"]
    plt.figure(figsize=(6, 3.5))
    plt.plot([x["step"] for x in s], [x["train_loss"] for x in s], label="train (per step)")
    plt.plot([e["step"] for e in log["epochs"]], [e["val_loss"] for e in log["epochs"]],
             "o-", label="validation (per epoch)")
    plt.xlabel("optimizer step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=200)
    print("peak GPU mem GiB:", log["peak_gpu_mem_gib"])

if __name__ == "__main__":
    main()
