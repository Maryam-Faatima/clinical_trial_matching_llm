"""Task 3: greedy decoding without cache (reference) vs with an explicit KV cache."""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

def load(adapter=None, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model

def kv_bytes_per_token(cfg, bytes_per_elem=2):
    """K and V, for every layer, for every KV head."""
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * bytes_per_elem

def synthetic_ids(tok, n):
    """n real-text tokens (unique train patient notes joined) for length experiments."""
    notes = {}
    for line in open("data/processed/train.jsonl"):
        r = json.loads(line)
        notes.setdefault(r["patient_id"], " ".join(r["sentences"].values()))
    ids = tok(" ".join(notes.values()), add_special_tokens=False).input_ids
    assert len(ids) >= n, f"need {n} tokens, have {len(ids)}"
    return torch.tensor([ids[:n]]).cuda()

def _sync():
    torch.cuda.synchronize()

# ---------------- explicit cache-aware forward passes ----------------
def _prefill(model, ids):
    """Run the whole prompt once. Positions 0..L-1, causal mask built inside the model.
    Returns last-position logits and the filled cache."""
    L = ids.shape[1]
    pos = torch.arange(L, device=ids.device).unsqueeze(0)
    o = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos,
              use_cache=True, logits_to_keep=1)
    return o.logits[:, -1, :], o.past_key_values, L

def _decode_step(model, tok_id, cache, n_cached):
    """Feed ONLY the newest token. Its position is n_cached (RoPE is applied with this
    absolute position and the key is stored in the cache already rotated). The mask has
    n_cached+1 ones: one query may look at every cached key plus itself."""
    dev = tok_id.device
    o = model(input_ids=tok_id,
              attention_mask=torch.ones(1, n_cached + 1, dtype=torch.long, device=dev),
              position_ids=torch.tensor([[n_cached]], device=dev),
              past_key_values=cache, use_cache=True,
              cache_position=torch.tensor([n_cached], device=dev))
    return o.logits[:, -1, :], o.past_key_values

# ---------------- reference: NO cache ----------------
@torch.no_grad()
def generate_uncached(model, ids, max_new, eos_id=None):
    """Re-run the WHOLE sequence at every step. No cache is read or written."""
    seq, toks, t_pre, t_dec = ids.clone(), [], 0.0, 0.0
    for step in range(max_new):
        if toks and eos_id is not None and toks[-1] == eos_id:
            break
        _sync(); t0 = time.perf_counter()
        nxt = model(input_ids=seq, use_cache=False,
                    logits_to_keep=1).logits[:, -1, :].argmax(-1, keepdim=True)
        _sync(); dt = time.perf_counter() - t0
        if step == 0:
            t_pre = dt
        else:
            t_dec += dt
        toks.append(nxt.item())
        seq = torch.cat([seq, nxt], dim=1)
    return {"tokens": toks, "prefill_s": t_pre, "decode_s": t_dec}

# ---------------- cached ----------------
@torch.no_grad()
def generate_cached(model, ids, max_new, eos_id=None, check=True):
    toks, t_dec = [], 0.0
    _sync(); t0 = time.perf_counter()
    logits, cache, n_cached = _prefill(model, ids)
    nxt = logits.argmax(-1, keepdim=True)
    _sync(); t_pre = time.perf_counter() - t0
    toks.append(nxt.item())
    for _ in range(1, max_new):
        if eos_id is not None and toks[-1] == eos_id:
            break
        _sync(); t0 = time.perf_counter()
        logits, cache = _decode_step(model, nxt, cache, n_cached)
        n_cached += 1                       # the fed token is now stored in the cache
        nxt = logits.argmax(-1, keepdim=True)
        _sync(); t_dec += time.perf_counter() - t0
        if check:                           # cache must grow by exactly 1 per step
            assert cache.get_seq_length() == n_cached, (cache.get_seq_length(), n_cached)
        toks.append(nxt.item())
    return {"tokens": toks, "prefill_s": t_pre, "decode_s": t_dec}

# ---------------- numerical-difference probes (teacher forcing) ----------------
@torch.no_grad()
def teacher_forced_logits_full(model, ids, forced):
    """Logits that predict each token in `forced`, from ONE uncached forward pass."""
    full = torch.cat([ids, torch.tensor([forced], device=ids.device)], dim=1)
    return model(input_ids=full[:, :-1], use_cache=False,
                 logits_to_keep=len(forced)).logits[0].float()

@torch.no_grad()
def teacher_forced_logits_cached(model, ids, forced):
    """Same logits, but built step by step through the KV cache."""
    logits, cache, n_cached = _prefill(model, ids)
    rows = [logits[0].float()]
    for tok_id in forced[:-1]:
        t = torch.tensor([[tok_id]], device=ids.device)
        logits, cache = _decode_step(model, t, cache, n_cached)
        n_cached += 1
        rows.append(logits[0].float())
    return torch.stack(rows)
