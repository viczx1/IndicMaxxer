#!/usr/bin/env python3
"""Build IndicMaxxer_2B_build.ipynb (Kaggle T4 x2): fuse tokenizer -> data -> LoRA SFT -> merge -> quick test.
Usage: python pipeline/make_imx2b_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()

INTRO = """# IndicMaxxer-2B: Sarvam-1 + our tokenizer fusion + /reasoning & /instruct + IndicMaxxer identity
**Kaggle settings:** Accelerator **GPU T4 x2**, Internet **on**, Secret `HF_TOKEN` attached
(open https://huggingface.co/sarvamai/sarvam-1 once and accept its terms if asked; licence: Sarvam non-commercial).
**Input dataset** needs: `indicmaxxer-48k.json` (our tokenizer), `heldout.json`, `indicmaxxer_ultimate.jsonl`.

Run with **Save Version → Save & Run All (Commit)**. Total ≈ 2 h (data 15 · fusion 10 · training 75 · merge 3 · test 5 min). Each step checks the previous one and stops with a clear message if it failed. Training saves its progress every 15 min.
Output: `/kaggle/working/IndicMaxxer-2B` (merged model + tokenizer) and `/kaggle/working/reports/`."""

SETUP = r'''# 1) Setup
import os, sys, json, glob, subprocess, random, time
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft", "accelerate",
                       "datasets", "tokenizers", "regex", "indic-transliteration"])
import torch
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    import getpass
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("Hugging Face READ token (hidden): ")
INPUT = os.environ.get("IMX_INPUT", "/kaggle/input")
WORK = os.environ.get("IMX_WORK", "/kaggle/working")
REP = f"{WORK}/reports"
os.makedirs(REP, exist_ok=True)
found = {os.path.basename(p): p for p in glob.glob(f"{INPUT}/**/*", recursive=True) if os.path.isfile(p)}
for f in ["indicmaxxer-48k.json", "heldout.json", "indicmaxxer_ultimate.jsonl"]:
    print(f, "OK" if f in found else "MISSING")
assert "indicmaxxer-48k.json" in found and "heldout.json" in found, "upload our tokenizer + heldout.json"'''

CONFIG = r'''# 4) Config: the only cell to touch
BASE = "sarvamai/sarvam-1"
TRAIN_MIN = 75                         # LoRA training time on 2 x T4
TOTAL = 24000                          # SFT examples (SmolTalk-heavy, as decided)
MIX = {"en": 0.45, "kn": 0.22, "hi": 0.11, "kanglish": 0.11, "hinglish": 0.11}
FUSE_QUOTA = {"kanglish": 1500, "hinglish": 1500, "kn": 600, "hi": 300, "en": 0}   # max new tokens per style
CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
FUSED, OUT = f"{WORK}/sarvam_fused", f"{WORK}/IndicMaxxer-2B"
STATE = f"{WORK}/imx2b_trainable_state.pt"   # LoRA + embeddings, saved every 15 min (survives a crash)
USE_BOTH_GPUS = False                        # True = ~2x faster but less robust; flip only if 1 GPU works

def run(cmd):
    """Run a script, stream its output live, and stop the notebook with a clear message if it fails."""
    import subprocess
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
    if p.wait() != 0:
        raise RuntimeError("step failed (see the error above). Last lines: " + " | ".join(t.strip() for t in tail[-15:]))
print("ok")'''

DATA = r'''# 5) Data: ultimate mix (SmolTalk + Aya + IndicAlign + Hinglish sets + our chats + Codex native reasoning)
#    rendered in IndicMaxxer-2B chat format with /reasoning and /instruct, plus identity rows
ULT = f"{WORK}/ultimate_2b.jsonl"
if not os.path.exists(ULT):
    t0 = time.time()
    cmd = [sys.executable, "build_ultimate.py", "--out", ULT, "--total", str(TOTAL), "--target", json.dumps(MIX),
           "--max-minutes", "2"]
    if "indicmaxxer_ultimate.jsonl" in found:
        cmd += ["--own", found["indicmaxxer_ultimate.jsonl"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3500:], r.stderr[-1200:], f"\n{(time.time() - t0) / 60:.1f} min")

SYS = {
    "kn": "ನೀನು IndicMaxxer, Sarvam-1 ಆಧಾರಿತ ಭಾರತೀಯ ಭಾಷಾ ಸಹಾಯಕ: ವೃತ್ತಿಪರ, ಆತ್ಮೀಯ, ನೇರ, ಸ್ವಲ್ಪ ಹಾಸ್ಯ. ಕನ್ನಡ ಲಿಪಿಯಲ್ಲೇ ಉತ್ತರಿಸು.",
    "hi": "तुम IndicMaxxer हो, Sarvam-1 पर बना भारतीय भाषा सहायक: पेशेवर, गर्मजोशी भरे, सटीक, हल्का मज़ाक। देवनागरी हिंदी में ही जवाब दो।",
    "en": "You are IndicMaxxer, an Indian-language assistant built on Sarvam-1: professional, warm, crisp, light dry humour. Reply in English.",
    "kanglish": "You are IndicMaxxer, an Indian-language assistant built on Sarvam-1. The user writes Kanglish (Kannada in English letters): reply in Kanglish only.",
    "hinglish": "You are IndicMaxxer, an Indian-language assistant built on Sarvam-1. The user writes Hinglish (Hindi in English letters): reply in Hinglish only.",
}
IDENT = {
    "en": (["who are you?", "what's your name?", "what can you do?", "are you ChatGPT?", "which model are you?"],
           "I'm IndicMaxxer, an Indian-language assistant built on Sarvam-1. I chat in Kannada, Hindi, English, Kanglish and Hinglish, reason step by step with /reasoning, answer directly with /instruct, and can search the web, read and write documents, and read scanned text."),
    "kn": (["ನೀನು ಯಾರು?", "ನಿನ್ನ ಹೆಸರೇನು?", "ನೀನು ಏನು ಮಾಡಬಲ್ಲೆ?", "ನೀನು ChatGPT ನಾ?"],
           "ನಾನು IndicMaxxer, Sarvam-1 ಆಧಾರಿತ ಭಾರತೀಯ ಭಾಷಾ ಸಹಾಯಕ. ಕನ್ನಡ, ಹಿಂದಿ, ಇಂಗ್ಲಿಷ್, ಕಂಗ್ಲಿಷ್, ಹಿಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಮಾತಾಡುತ್ತೇನೆ; /reasoning ಹಾಕಿದರೆ ಕನ್ನಡದಲ್ಲೇ ಹಂತ ಹಂತವಾಗಿ ಯೋಚಿಸುತ್ತೇನೆ, /instruct ಹಾಕಿದರೆ ನೇರ ಉತ್ತರ ಕೊಡುತ್ತೇನೆ."),
    "hi": (["तुम कौन हो?", "तुम्हारा नाम क्या है?", "तुम क्या कर सकते हो?", "क्या तुम ChatGPT हो?"],
           "मैं IndicMaxxer हूँ, Sarvam-1 पर बना भारतीय भाषा सहायक। कन्नड़, हिंदी, अंग्रेज़ी, कंग्लिश और हिंग्लिश में बात करता हूँ; /reasoning पर हिंदी में ही कदम-दर-कदम सोचता हूँ, /instruct पर सीधा जवाब देता हूँ।"),
    "kanglish": (["neenu yaaru?", "ninna hesaru enu?", "neenu enu maadtiya?", "neenu ChatGPT na?"],
                 "Naanu IndicMaxxer, Sarvam-1 mele build aagiro Indian language assistant. Kannada, Hindi, English, Kanglish, Hinglish alli maathadtini; /reasoning haakidre step by step yochistini, /instruct haakidre direct answer kodtini."),
    "hinglish": (["tum kaun ho?", "tumhara naam kya hai?", "tum kya kar sakte ho?", "kya tum ChatGPT ho?"],
                 "Main IndicMaxxer hoon, Sarvam-1 par bana Indian language assistant. Kannada, Hindi, English, Kanglish aur Hinglish mein baat karta hoon; /reasoning pe step by step sochta hoon, /instruct pe seedha jawab deta hoon."),
}
rng = random.Random(0)
def render(msgs, lang, mode, think=""):
    flag = mode if rng.random() < 0.95 else None                       # 5%: no flag, model decides
    parts = [f"<|im_start|>system\n{SYS.get(lang, SYS['en'])}<|im_end|>\n"] if rng.random() < 0.8 else []
    for i, m in enumerate(msgs[:-1]):
        c = m["content"]
        if i == len(msgs) - 2 and flag:
            c += " /reasoning" if flag == "reasoning" else " /instruct"
        parts.append(f"<|im_start|>{m['role']}\n{c}<|im_end|>\n")
    prompt = "".join(parts) + "<|im_start|>assistant\n"
    completion = (f"<think>\n{think}\n</think>\n\n" if think else "<think>\n\n</think>\n\n") + msgs[-1]["content"] + "<|im_end|>"
    return {"prompt": prompt, "completion": completion, "lang": lang, "mode": mode}

out, cnt = [], {}
for line in open(ULT, encoding="utf-8"):
    r = json.loads(line)
    last = r["messages"][-1]
    think = last.get("think", "")
    mode = "reasoning" if think else "instruct"
    out.append(render(r["messages"], r["lang"], mode, think))
    cnt[(r["lang"], mode)] = cnt.get((r["lang"], mode), 0) + 1
for lang, (qs, a) in IDENT.items():
    for _ in range(60):
        out.append(render([{"role": "user", "content": rng.choice(qs)}, {"role": "assistant", "content": a}], lang, "instruct"))
rng.shuffle(out)
SFT = f"{WORK}/sft_2b.jsonl"
with open(SFT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "examples ->", SFT)
print({f"{l}/{m}": n for (l, m), n in sorted(cnt.items())})
print(out[0]["prompt"][-300:] + out[0]["completion"][:300])'''

FUSE = r'''# 6) Fuse our tokenizer into Sarvam-1's (only where ours is cheaper) + chat/think tokens (~10 min)
fc = {"base": BASE, "our_tok": found["indicmaxxer-48k.json"], "heldout": found["heldout.json"], "extra_jsonl": ULT,
      "min_freq": 8, "scan_top": 30000, "quota": FUSE_QUOTA, "chat_template": CHAT_TEMPLATE,
      "out_dir": FUSED, "report": f"{REP}/fusion_report.json"}
json.dump(fc, open("fuse_config.json", "w"), ensure_ascii=False)
if not os.path.exists(f"{FUSED}/config.json"):
    run([sys.executable, "imx2b_fuse.py", "fuse_config.json"])
assert os.path.exists(f"{FUSED}/config.json"), "fusion did not produce a model"
print(json.dumps(json.load(open(f"{REP}/fusion_report.json", encoding="utf-8")), ensure_ascii=False, indent=1)[:2500])'''

TRAIN = r'''# 7) LoRA fine-tune (live log): LoRA + only the NEW token rows. Out of memory -> max_len 768, lora_r 16.
old_vocab = json.load(open(f"{REP}/fusion_report.json"))["vocab_old"]
tc = {"model_dir": FUSED, "data": SFT, "state_path": STATE, "out_dir": OUT, "minutes": TRAIN_MIN, "max_len": 1024,
      "old_vocab": old_vocab,
      "micro_b": 1, "accum": 16, "lora_r": 32, "lr": 2e-4, "emb_lr": 5e-5, "warmup": 20, "log_every": 5,
      "ckpt_every_min": 15}
json.dump(tc, open("train_config.json", "w"))
n = torch.cuda.device_count() if USE_BOTH_GPUS else 1
if n > 1:
    run(["torchrun", "--standalone", f"--nproc_per_node={n}", "imx2b_train.py", "train_config.json"])
else:
    run([sys.executable, "imx2b_train.py", "train_config.json"])
assert os.path.exists(STATE), "training saved no state"'''

MERGE = r'''# 8) Merge LoRA + trained embeddings into the full model -> IndicMaxxer-2B (~3 min)
torch.cuda.empty_cache()
run([sys.executable, "imx2b_merge.py", "train_config.json"])
print(os.listdir(OUT))'''

TEST = r'''# 8) Quick test: /instruct and /reasoning in 5 styles + native multi-pass reasoning (self-consistency)
import re, collections
from transformers import AutoTokenizer, AutoModelForCausalLM
assert os.path.exists(f"{OUT}/config.json"), f"{OUT} missing: the training or merge cell failed, see its output"
tk = AutoTokenizer.from_pretrained(OUT)
try:
    lm = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.float16).to("cuda:0").eval()
except TypeError:
    lm = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.float16).to("cuda:0").eval()
def ask(user, mode="instruct", lang="en", max_new=300, temp=0.0, history=()):
    msgs = [{"role": "system", "content": SYS.get(lang, SYS["en"])}] + list(history) + \
           [{"role": "user", "content": f"{user} /{mode}"}]
    text = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tk(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        out = lm.generate(ids, max_new_tokens=max_new, do_sample=temp > 0, temperature=max(temp, 1e-5), top_p=0.9,
                          repetition_penalty=1.1, eos_token_id=[tk.convert_tokens_to_ids("<|im_end|>"), tk.eos_token_id],
                          pad_token_id=tk.pad_token_id)
    text = tk.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "")
    think, ans = (text.split("</think>", 1) + [""])[:2] if "</think>" in text else ("", text)
    return think.replace("<think>", "").strip(), ans.strip(), out.shape[1] - ids.shape[1]
def last_num(s):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None
def reason_k(user, lang, k=3):
    """Think k times natively (sampled), majority vote on the final number."""
    runs = [ask(user, "reasoning", lang, 400, temp=0.7) for _ in range(k)]
    votes = collections.Counter(last_num(a) for _, a, _ in runs if last_num(a))
    return runs, (votes.most_common(1)[0][0] if votes else None), sum(n for _, _, n in runs)
tests = {"kn": "ನನ್ನ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ, ಏನು ಮಾಡಬೇಕು?", "hi": "मेरा रिफंड कब तक आएगा?",
         "en": "hii, who are you?", "kanglish": "naale meeting ide, enu prepare maadli?",
         "hinglish": "kal interview hai, kuch tips do"}
res = {}
for lang, q in tests.items():
    th, a, n = ask(q, "instruct", lang)
    res[f"{lang}/instruct"] = a
    print(f"\n[{lang} /instruct] {n} tok\n{a[:400]}")
q = "ರವಿ ಬಳಿ 500 ರೂಪಾಯಿ ಇತ್ತು. ಅವನು 120 ರೂಪಾಯಿಗೆ ಪುಸ್ತಕ ಮತ್ತು 80 ರೂಪಾಯಿಗೆ ಪೆನ್ನು ಖರೀದಿಸಿದನು. ಎಷ್ಟು ಉಳಿಯಿತು?"
th, a, n = ask(q, "reasoning", "kn", 400)
print(f"\n[kn /reasoning] {n} tok\nTHINK: {th[:500]}\nANSWER: {a[:300]}")
runs, vote, ntok = reason_k(q, "kn", 3)
print(f"\n[kn /reasoning x3 self-consistency] vote = {vote} (gold 300), total {ntok} tokens")
res["kn_math"] = {"single": a, "vote": vote}
json.dump(res, open(f"{REP}/test_samples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)'''

def cell(kind, s):
    c = {"cell_type": kind, "metadata": {}, "source": s.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c

cells = [("markdown", INTRO), ("code", SETUP), ("code", "%%writefile build_ultimate.py\n" + src("build_ultimate.py")),
         ("code", "%%writefile imx2b_fuse.py\n" + src("imx2b_fuse.py")),
         ("code", "%%writefile imx2b_train.py\n" + src("imx2b_train.py")),
         ("code", "%%writefile imx2b_merge.py\n" + src("imx2b_merge.py")),
         ("code", CONFIG), ("code", DATA), ("code", FUSE), ("code", TRAIN), ("code", MERGE), ("code", TEST)]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell(k, s) for k, s in cells]}
dst = os.path.join(os.path.dirname(HERE), "IndicMaxxer_2B_build.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", dst)
