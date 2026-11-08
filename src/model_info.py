"""Facts for the report (A2/F1): params, layers, heads, context, memory, tokenizer match."""
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

TARGET, DRAFT = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"


def info(name):
    cfg = AutoConfig.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    n = sum(p.numel() for p in m.parameters())
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_bytes_per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
    print(f"\n{name}\n params={n/1e9:.3f}B  fp16 weights={n*2/2**30:.2f} GiB\n"
          f" layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"q_heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} head_dim={head_dim}\n"
          f" vocab={cfg.vocab_size} max_pos={cfg.max_position_embeddings} "
          f"tied_emb={cfg.tie_word_embeddings}\n KV cache fp16 = {kv_bytes_per_tok} bytes/token")
    del m


if __name__ == "__main__":
    for n in (TARGET, DRAFT):
        info(n)
    t, d = AutoTokenizer.from_pretrained(TARGET), AutoTokenizer.from_pretrained(DRAFT)
    sample = "Patient has no history of renal disease. 0. A 56-year-old female."
    print("\ntokenizer vocab equal:", t.get_vocab() == d.get_vocab())
    print("same ids on sample:", t(sample).input_ids == d(sample).input_ids)