"""Non-test prompts for A2: short + doc6k reused from a local benchmark prompt file (ids only; not included in this repo), plus p6k = the
same seeded word-list generator as p4k, sized to <= 6000 tokens. Writes out/prompts_a2.json (ids, never printed)."""
import json, random, hashlib
from transformers import AutoTokenizer
import os
MODEL = os.environ["QWEN38_MODEL"]
SRC = os.environ["A2_PROMPTS_SRC"]  # JSON {"ids": {"short", "story", "doc6k": [token ids]}, "eos": ...}; not included
tok = AutoTokenizer.from_pretrained(MODEL)
def chat(c):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
def long_prompt(words, seed=11):
    rng = random.Random(seed)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."
src = json.load(open(SRC))
P = {k: src["ids"][k] for k in ("short", "story", "doc6k")}
lo, hi = 1000, 9000
while lo < hi:
    mid = (lo + hi + 1) // 2
    if len(chat(long_prompt(mid))) <= 6000: lo = mid
    else: hi = mid - 1
P["p6k"] = chat(long_prompt(lo))
meta = {k: {"n": len(v), "sha": hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]} for k, v in P.items()}
json.dump({"ids": P, "meta": meta, "eos": src["eos"], "p6k_words": lo, "p6k_seed": 11}, open("out/prompts_a2.json", "w"))
print(json.dumps(meta))
