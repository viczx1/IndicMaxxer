#!/usr/bin/env python3
"""IndicMaxxer-2B step 1: fuse our tokenizer into Sarvam-1's, add chat/think tokens, resize + initialise embeddings.

Fusion rule (only add a token where ours is cheaper):
  for frequent words in our 5 styles (kn / hi / en / Kanglish / Hinglish), if Sarvam splits the word into >= 3 pieces
  and our 48K tokenizer uses fewer, add " word" as one new token. New embedding = mean of the Sarvam pieces it replaces.
Then measure tokens/word per style BEFORE vs AFTER on held-out text; if a style got worse, its tokens are dropped.

Usage: python imx2b_fuse.py fuse_config.json
"""
import json, os, sys, time
from collections import Counter
import regex
import torch
from tokenizers import Tokenizer
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer

C = json.load(open(sys.argv[1]))
HF = os.environ.get("HF_TOKEN") or None
STYLES = ["kn", "hi", "en", "kanglish", "hinglish"]
WORD = regex.compile(r"[\p{L}\p{M}]+")
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
t0 = time.time()

tok = AutoTokenizer.from_pretrained(C["base"], token=HF)
ours = Tokenizer.from_file(C["our_tok"])
held = json.load(open(C["heldout"], encoding="utf-8"))
extra = {s: [] for s in STYLES}                      # more romanised text from the chat dataset, if given
if C.get("extra_jsonl") and os.path.exists(C["extra_jsonl"]):
    for line in open(C["extra_jsonl"], encoding="utf-8"):
        r = json.loads(line)
        if r.get("lang") in extra:
            extra[r["lang"]] += [m["content"] for m in r.get("messages", [])]

def pieces(t, w):
    """How many tokens tokenizer t spends on w in the middle of a sentence."""
    return len(t.encode("a " + w, add_special_tokens=False)) - len(t.encode("a", add_special_tokens=False))

def tpw(t, docs):
    ids = words = 0
    for d in docs:
        ids += len(t.encode(d, add_special_tokens=False))
        words += len(WORD.findall(d))
    return round(ids / max(words, 1), 3)

eval_docs = {s: held.get(s, [])[-120:] for s in STYLES}
cand_docs = {s: held.get(s, [])[:-120] + extra[s] for s in STYLES}
before = {s: tpw(tok, eval_docs[s]) for s in STYLES if eval_docs[s]}
print("Sarvam tokens/word BEFORE:", before, flush=True)

# ---------------------------------------------------------------- pick fusion tokens
chosen = {}
for s in STYLES:
    freq = Counter()
    for d in cand_docs[s]:
        for w in WORD.findall(d):
            freq[w.lower() if s in ("en", "kanglish", "hinglish") else w] += 1
    scored = []
    for w, f in freq.most_common(C.get("scan_top", 20000)):
        if f < C["min_freq"]:
            break                                     # most_common is sorted: the rest are rarer
        if len(w) < 3:
            continue
        ps = pieces(tok, w)
        if ps < 3:
            continue
        po = len(ours.encode(" " + w).ids)
        if po >= ps:
            continue                                  # ours is not cheaper: not fused
        scored.append((f * (ps - 1), w, ps, po))
    scored.sort(reverse=True)
    chosen[s] = scored[:C["quota"].get(s, 0)]
    print(f"{s}: {len(scored)} candidates, keeping {len(chosen[s])}; top:",
          [(w, ps, po) for _, w, ps, po in chosen[s][:8]], flush=True)

def apply(selection):
    t = AutoTokenizer.from_pretrained(C["base"], token=HF)
    words = sorted({w for s in selection for _, w, _, _ in selection[s]})
    t.add_tokens([AddedToken(" " + w, single_word=False, lstrip=False, rstrip=False, normalized=False) for w in words])
    return t, words

tok2, words = apply(chosen)
after = {s: tpw(tok2, eval_docs[s]) for s in before}
print("tokens/word AFTER fusion:", after, flush=True)
worse = [s for s in after if after[s] > before[s] + 1e-9]
if worse:                                             # never pay a tax increase: drop styles that got worse
    print("dropping fusion for styles that got worse:", worse)
    for s in worse:
        chosen[s] = []
    tok2, words = apply(chosen)
    after = {s: tpw(tok2, eval_docs[s]) for s in before}
    print("tokens/word AFTER (final):", after)
tok2.add_special_tokens({"additional_special_tokens": SPECIALS})
tok2.chat_template = C["chat_template"]
if tok2.pad_token is None:
    tok2.pad_token = tok2.eos_token

# ---------------------------------------------------------------- model: resize + init
try:
    model = AutoModelForCausalLM.from_pretrained(C["base"], dtype=torch.float16, token=HF)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(C["base"], torch_dtype=torch.float16, token=HF)
old_n = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok2), pad_to_multiple_of=64)
emb = model.get_input_embeddings().weight.data
head = model.get_output_embeddings().weight.data if model.get_output_embeddings() is not None else None
tied = head is None or head.data_ptr() == emb.data_ptr()
with torch.no_grad():
    mean_e = emb[:old_n].float().mean(0)
    mean_h = head[:old_n].float().mean(0) if not tied else None
    for w in words:
        nid = tok2.convert_tokens_to_ids(" " + w)
        src = tok.encode("a " + w, add_special_tokens=False)[len(tok.encode("a", add_special_tokens=False)):]
        emb[nid] = emb[src].float().mean(0).to(emb.dtype)
        if not tied:
            head[nid] = head[src].float().mean(0).to(head.dtype)
    for sp in SPECIALS:
        nid = tok2.convert_tokens_to_ids(sp)
        emb[nid] = (mean_e + 0.01 * torch.randn_like(mean_e)).to(emb.dtype)
        if not tied:
            head[nid] = (mean_h + 0.01 * torch.randn_like(mean_h)).to(head.dtype)
os.makedirs(C["out_dir"], exist_ok=True)
model.save_pretrained(C["out_dir"], safe_serialization=True)
tok2.save_pretrained(C["out_dir"])
report = {"before": before, "after": after,
          "saving_pct": {s: round(100 * (before[s] - after[s]) / before[s], 1) for s in after},
          "fused_tokens": len(words), "per_style": {s: len(v) for s, v in chosen.items()},
          "vocab_old": old_n, "vocab_new": len(tok2), "tied_embeddings": tied,
          "extra_params_M": round((model.get_input_embeddings().weight.shape[0] - old_n) * emb.shape[1]
                                  * (1 if tied else 2) / 1e6, 2),
          "examples": {s: [w for _, w, _, _ in chosen[s][:15]] for s in chosen}}
json.dump(report, open(C["report"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))
print(f"fused model saved to {C['out_dir']} in {(time.time() - t0) / 60:.1f} min")
