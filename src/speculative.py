"""Task 5: greedy speculative decoding with explicit KV-cache management for BOTH models.

Invariant at the start of every round, with `committed` = prompt + accepted tokens:
  * target cache holds committed[:-1]  (the last committed token is still to be fed)
  * draft cache holds a prefix of committed (it feeds whatever is missing, see `pending`)
"""
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM

TARGET_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

def load_model(base, adapter=None, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model

def crop_cache(cache, new_len):
    """Shrink a KV cache to its first new_len tokens (version-proof)."""
    cur = cache.get_seq_length()
    if cur <= new_len:
        return
    try:
        cache.crop(-(cur - new_len))        # newer versions: negative = tokens to remove
    except Exception:
        pass
    if cache.get_seq_length() != new_len:   # fallback: slice the tensors ourselves
        for layer in cache.layers:
            layer.keys = layer.keys[..., :new_len, :]
            layer.values = layer.values[..., :new_len, :]

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _fwd(model, ids, cache, n_cached, keep_all):
    """Feed m new tokens after n_cached cached ones.
    Positions n_cached..n_cached+m-1; mask length n_cached+m (the model applies the causal
    mask among the m new tokens). Returns logits [m, V] (keep_all) or [1, V], and the cache."""
    m, dev = ids.shape[1], ids.device
    pos = torch.arange(n_cached, n_cached + m, device=dev)
    o = model(input_ids=ids,
              attention_mask=torch.ones(1, n_cached + m, dtype=torch.long, device=dev),
              position_ids=pos.unsqueeze(0), past_key_values=cache, use_cache=True,
              cache_position=pos, logits_to_keep=0 if keep_all else 1)
    return (o.logits[0] if keep_all else o.logits[0, -1:]), o.past_key_values

@torch.no_grad()
def speculative_generate(target, draft, ids, max_new, k, eos_id=None):
    dev = ids.device
    st = {"rounds": 0, "drafted": 0, "accepted": 0, "draft_forwards": 0,
          "target_forwards": 1, "a_hist": Counter()}
    t_draft = t_verify = 0.0
    _sync(); t_start = time.perf_counter()

    # target prefill -> first generated token (same as ordinary greedy)
    logits, t_cache = _fwd(target, ids, None, 0, False)
    first = int(logits[-1].argmax())
    _sync(); t_prefill = time.perf_counter() - t_start

    committed = ids[0].tolist() + [first]
    gen = [first]
    d_cache, d_len = None, 0

    while len(gen) < max_new and not (eos_id is not None and gen[-1] == eos_id):
        L_old = len(committed)

        # ---- 1. DRAFT: catch up on missing committed tokens, then propose k tokens
        t0 = time.perf_counter()
        pending = torch.tensor([committed[d_len:]], device=dev)
        lg, d_cache = _fwd(draft, pending, d_cache, d_len, False)
        d_len = L_old
        st["draft_forwards"] += 1
        drafts = []
        for i in range(k):
            tok = int(lg[-1].argmax())
            drafts.append(tok)
            if i < k - 1:                       # d_k is not fed to the draft
                lg, d_cache = _fwd(draft, torch.tensor([[tok]], device=dev), d_cache, d_len, False)
                d_len += 1
                st["draft_forwards"] += 1
        _sync(); t_draft += time.perf_counter() - t0

        # ---- 2. VERIFY: one target pass over [last, d1..dk]
        t0 = time.perf_counter()
        n = L_old - 1
        assert t_cache.get_seq_length() == n, "target cache out of sync"
        inp = torch.tensor([[committed[-1]] + drafts], device=dev)
        lg, t_cache = _fwd(target, inp, t_cache, n, True)   # [k+1, V]
        choice = lg.argmax(-1).tolist()      # choice[j] = target's greedy token after inp[j]
        _sync(); t_verify += time.perf_counter() - t0
        st["target_forwards"] += 1

        # ---- 3. ACCEPT the matching prefix, then take the target's own token
        a = 0
        while a < k and drafts[a] == choice[a]:
            a += 1
        new = drafts[:a] + [choice[a]]       # correction (a<k) or bonus token (a==k)
        st["rounds"] += 1; st["drafted"] += k; st["accepted"] += a; st["a_hist"][a] += 1
        for tok in new:
            committed.append(tok); gen.append(tok)
            if (eos_id is not None and tok == eos_id) or len(gen) >= max_new:
                break

        # ---- 4. ROLL BACK caches to the accepted prefix
        crop_cache(t_cache, len(committed) - 1)
        valid_d = L_old + min(a, k - 1)      # draft cache holds committed + d1..d_{k-1}
        crop_cache(d_cache, valid_d); d_len = valid_d
        assert t_cache.get_seq_length() == len(committed) - 1
        assert d_cache.get_seq_length() == valid_d

    _sync()
    return {"tokens": gen, "total_s": time.perf_counter() - t_start, "prefill_s": t_prefill,
            "draft_s": t_draft, "verify_s": t_verify, "stats": st}
