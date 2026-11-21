#!/usr/bin/env python3
"""Build IndicMaxxer_speedrun.ipynb (one-click Colab TPU run) with imx_core.py embedded.
Usage: python pipeline/make_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = open(os.path.join(HERE, "imx_core.py"), encoding="utf-8").read()

INTRO = """# IndicMaxxer speedrun: data → tokenizer → pretrain (~100M) → SFT → eval → export
**Before you press Runtime → Run all:**
1. **Runtime → Change runtime type → v5e-1 TPU.**
2. Add a Colab **secret** named `HF_TOKEN` (key icon on the left, enable notebook access). A read-only Hugging Face token is enough.
3. (For fine-tuning) Upload the Codex data, after running `pipeline/clean_distill.py` and `pipeline/build_sft.py` on the laptop:
   `data/sft/train.jsonl` → Drive `IndicMaxxer/sft/codex_train.jsonl` and `data/sft/val.jsonl` → Drive `IndicMaxxer/sft/codex_val.jsonl`.
   You can upload them while pretraining runs; the fine-tuning cell reads them when it starts.

Every stage saves to Drive (`MyDrive/IndicMaxxer`). If Colab disconnects, reconnect and **Run all** again: finished stages are skipped and training resumes from its last checkpoint.
Rough timing: data ~15 min · tokenizer ~10 min · pretraining 120 min · SFT 20 min · eval + export ~15 min."""

SETUP = r'''# 1) Setup: packages, Drive, token
import importlib, subprocess, sys, os, json, time, glob, shutil, random
from collections import Counter

def need(mod, pkg):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for mod, pkg in [("datasets", "datasets"), ("tokenizers", "tokenizers"), ("optax", "optax"), ("tiktoken", "tiktoken"),
                 ("indic_transliteration", "indic-transliteration"), ("regex", "regex"), ("safetensors", "safetensors"),
                 ("gradio", "gradio")]:
    need(mod, pkg)
try:
    from transformers import Qwen3Config  # needs transformers >= 4.51
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url",
                           "https://download.pytorch.org/whl/cpu"])

import jax
print("JAX devices:", jax.devices())
if jax.devices()[0].platform != "tpu":
    print("WARNING: not on a TPU. Switch the runtime to v5e-1 TPU or training will be very slow.")

from google.colab import drive
drive.mount("/content/drive")
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
print("HF token:", "found" if HF_TOKEN else "MISSING (add the HF_TOKEN secret)")

DRIVE, LOCAL = "/content/drive/MyDrive/IndicMaxxer", "/content/imx"
for d in ["data", "tok", "ckpt", "export", "sft", "results"]:
    os.makedirs(f"{DRIVE}/{d}", exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)'''

CONFIG = r'''# 3) Config: change sizes / budgets here
import importlib, numpy as np, jax.numpy as jnp
import imx_core as imx
importlib.reload(imx)

CFG = imx.Cfg()                       # IndicMaxxer-100M: d=576, 20 layers, 9/3 heads, ffn 1536, vocab 48K
TINY = imx.Cfg(d=128, layers=2, heads=4, kv_heads=2, head_dim=32, ffn=256, ctx=128)   # smoke test
PRETRAIN_MIN, SFT_MIN = 105, 25
MICRO_B, ACCUM = 8, 8                 # 8 x 8 x 1024 = 65K tokens per optimizer step
PEAK_LR, SFT_LR = 2e-3, 3e-4
DATA_MB = {"kn": 2500, "hi": 1000, "en": 600}   # ~180M / ~100M / ~130M tokens: <2 passes over Kannada in 2 h
ROMAN_MB = {"kanglish": 80, "hinglish": 80}
MIX = {"kn": 0.45, "hi": 0.25, "en": 0.18, "kanglish": 0.06, "hinglish": 0.06}
HELD = 1500                           # docs per language kept out of training (benchmarks, grounding/tool SFT)
print(f"IndicMaxxer params: {CFG.n_params() / 1e6:.1f}M")'''

DATA = r'''# 4) Data: stream + clean Kannada / Hindi / English, make Kanglish / Hinglish  (~15 min, skipped if done)
done = f"{DRIVE}/data/done.json"
have_local = all(os.path.exists(f"{LOCAL}/{l}.jsonl") for l in MIX)
bins_on_drive = all(os.path.exists(f"{DRIVE}/data/{l}.bin") for l in MIX)
if bins_on_drive and os.path.exists(f"{DRIVE}/data/heldout.json"):
    print("token shards already on Drive, skipping download:", json.load(open(done)) if os.path.exists(done) else "")
elif not have_local:
    from concurrent.futures import ThreadPoolExecutor
    jobs = [("HuggingFaceFW/fineweb-2", "kan_Knda", "kn"), ("HuggingFaceFW/fineweb-2", "hin_Deva", "hi"),
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "en")]
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        stats = list(ex.map(lambda j: imx.stream_docs(j[0], j[1], f"{LOCAL}/{j[2]}.jsonl", DATA_MB[j[2]], j[2],
                                                       HF_TOKEN, max_minutes=18), jobs))
    stats.append(imx.romanize_file(f"{LOCAL}/kn.jsonl", f"{LOCAL}/kanglish.jsonl", "kn", ROMAN_MB["kanglish"]))
    stats.append(imx.romanize_file(f"{LOCAL}/hi.jsonl", f"{LOCAL}/hinglish.jsonl", "hi", ROMAN_MB["hinglish"]))
    print(json.dumps(stats, indent=1), f"\n{(time.time() - t0) / 60:.1f} min")
    json.dump(stats, open(done, "w"))
held = json.load(open(f"{DRIVE}/data/heldout.json", encoding="utf-8")) if bins_on_drive and os.path.exists(f"{DRIVE}/data/heldout.json") \
    else {l: imx.heldout_texts(f"{LOCAL}/{l}.jsonl", HELD) for l in MIX}
json.dump(held, open(f"{DRIVE}/data/heldout.json", "w", encoding="utf-8"), ensure_ascii=False)
print({l: len(v) for l, v in held.items()}, "held-out docs")'''

TOKENIZER = r'''# 5) Tokenizer (48K byte-level BPE, marks kept with letters) + benchmark vs GPT-4 / GPT-4o / Qwen3
from tokenizers import Tokenizer
TOK = f"{DRIVE}/tok/indicmaxxer-48k.json"
if not os.path.exists(TOK):
    share_mb = {"kn": 60, "hi": 35, "en": 22, "kanglish": 10, "hinglish": 10}
    def texts():
        for l, mb in share_mb.items():
            yield from imx.iter_texts(f"{LOCAL}/{l}.jsonl", limit_bytes=mb * 1_000_000, skip_last=HELD)
    t0 = time.time()
    imx.train_tokenizer(texts(), CFG.vocab, TOK)
    print(f"tokenizer trained in {(time.time() - t0) / 60:.1f} min")
tok = Tokenizer.from_file(TOK)

table = imx.fertility_table({l: held[l][:300] for l in MIX}, TOK, HF_TOKEN)
json.dump(table, open(f"{DRIVE}/results/fertility.json", "w"), indent=1)
langs = list(MIX)
print("tokens per word".ljust(28), "".join(l.rjust(10) for l in langs))
for name, row in table.items():
    print(name.ljust(28), "".join(f"{row[l]:10.2f}" for l in langs))

# The root cause slide: GPT-4's pre-tokenizer splits words at every vowel sign
import regex
CL100K = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
for s in ["ಕನ್ನಡ ನಮ್ಮ ಭಾಷೆ. ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿಯಾಗಿದೆ.", "भारत की राजधानी दिल्ली है। हिंदी हमारी भाषा है।"]:
    a, b = regex.findall(CL100K, s), regex.findall(imx.SPLIT_PAT, s)
    print(f"\nGPT-4 pre-split: {len(a)} chunks {a}\nIndicMaxxer:     {len(b)} chunks {b}")'''

SHARDS = r'''# 6) Tokenize everything into uint16 shards (copied to Drive for resumes)
EOS, PAD = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|pad|>")
bins = {l: f"{LOCAL}/{l}.bin" for l in MIX}
for l in MIX:
    on_drive = f"{DRIVE}/data/{l}.bin"
    if os.path.exists(on_drive):
        if not os.path.exists(bins[l]):
            shutil.copy(on_drive, bins[l])
        continue
    t0 = time.time()
    n = imx.tokenize_to_bin(tok, f"{LOCAL}/{l}.jsonl", bins[l], EOS, skip_last=HELD)
    shutil.copy(bins[l], on_drive)
    print(f"{l}: {n / 1e6:.0f}M tokens in {(time.time() - t0) / 60:.1f} min")
tot = {l: os.path.getsize(p) // 2 for l, p in bins.items()}
print({l: f"{v / 1e6:.0f}M" for l, v in tot.items()}, f"total {sum(tot.values()) / 1e6:.0f}M tokens")'''

SMOKE = r'''# 7) Smoke test (~2 min): tiny model must learn, export to PyTorch must match, generation must run
p = imx.init_params(TINY)
tx_s = imx.make_optimizer(p)
o = tx_s.init(p)
step_s = imx.make_train_step(TINY, tx_s)
mix_s = imx.Mixer(bins, MIX, TINY.ctx)
losses = []
for i in range(30):
    t, m = mix_s(2, 4)
    p, o, l, g = step_s(p, o, t, m, jnp.float32(3e-3))
    losses.append(float(l))
print("loss", round(losses[0], 3), "->", round(losses[-1], 3))
assert np.isfinite(losses[-1]) and losses[-1] < losses[0], "tiny model did not learn"
diff = imx.export_hf(p, TINY, TOK, "/content/smoke_export", check_tokens=t[0, :2, :64])
print("JAX vs PyTorch relative logit diff:", diff)
assert diff < 5e-2, "export mismatch"
print(imx.generate(p, imx.make_next_logits(TINY), tok, imx.render_prompt("ನಮಸ್ಕಾರ"), TINY, max_new=8))
print("SMOKE TEST PASSED")'''

PRETRAIN = r'''# 8) Pretraining IndicMaxxer-100M (time-budgeted WSD schedule, checkpoints to Drive every 20 min)
CK, LOG = f"{DRIVE}/ckpt/pretrain.pkl", f"{DRIVE}/results/pretrain_log.jsonl"
if os.path.exists(CK):
    params, opt_state, meta = imx.load_ckpt(CK)
    tx = imx.make_optimizer(params)
    print("resuming:", {k: meta[k] for k in ("step", "tokens", "elapsed")})
else:
    params = imx.init_params(CFG)
    tx = imx.make_optimizer(params)
    opt_state, meta = tx.init(params), {}
step = imx.make_train_step(CFG, tx)
if not meta.get("done"):
    mixer = imx.Mixer(bins, MIX, CFG.ctx, seed=meta.get("step", 0))
    params, opt_state, meta, _ = imx.train_loop(params, opt_state, step, mixer, imx.wsd_lr(PEAK_LR, 300, 0.15),
                                                PRETRAIN_MIN * 60, ACCUM, MICRO_B, ckpt_path=CK, meta=meta,
                                                log_path=LOG)
    meta["done"] = True
    imx.save_ckpt(CK, params, opt_state, meta)
print(f"pretraining done: {meta['step']} steps, {meta['tokens'] / 1e6:.0f}M tokens")
next_logits = imx.make_next_logits(CFG)
for s in ["ಬೆಂಗಳೂರು ನಗರವು", "भारत एक", "The monsoon in India"]:
    print(repr(s), "->", imx.generate(params, next_logits, tok, s, CFG, max_new=40))'''

SFT_DATA = r'''# 9) Fine-tuning data: Codex reasoning traces + Aya instructions + safety / grounding / tool use
rows = []
for path in sorted(glob.glob(f"{DRIVE}/sft/*train*.jsonl")):
    with open(path, encoding="utf-8") as f:
        rows += [(r["prompt"], r["completion"], r["lang"]) for r in map(json.loads, f)]
rows = rows * 2 if len(rows) < 10000 else rows   # small Codex-only set: show it twice
print("Codex traces:", len(rows), "(upload sft/codex_train.jsonl to include them)" if not rows else "")

from datasets import load_dataset
aya = None
for name in ["CohereLabs/aya_dataset", "CohereForAI/aya_dataset"]:
    try:
        aya = load_dataset(name, split="train", token=HF_TOKEN)
        break
    except Exception as e:
        print("could not load", name, type(e).__name__)
caps, cnt, rng = {"Kannada": ("kn", 20000), "Hindi": ("hi", 8000), "English": ("en", 4000)}, Counter(), random.Random(0)
if aya is not None:
    for ex in aya:
        lg = caps.get(ex.get("language"))
        if not lg or cnt[lg[0]] >= lg[1]:
            continue
        q, a = imx.norm_text(ex["inputs"]), imx.norm_text(ex["targets"])
        if not q or not a or len(a) > 2500:
            continue
        rows.append((imx.render_prompt(q, "no_think", lg[0], rng.random() < 0.7, rng.random() < 0.95),
                     imx.render_completion(a), lg[0]))
        cnt[lg[0]] += 1
print("Aya:", dict(cnt))
aux = imx.build_aux_sft(held)
chat = imx.build_smalltalk()
facts = imx.build_fact_sft(held)
langid = imx.build_langid_sft(held)
rows += aux + chat + facts + langid
print("safety + grounding + tools:", len(aux), "| small talk:", len(chat), "| web-search facts:", len(facts),
      "| total SFT examples:", len(rows))
SFT_T, SFT_M = imx.pack_sft(tok, rows, CFG.ctx, PAD)
print("packed rows:", SFT_T.shape, f"supervised tokens: {SFT_M.sum() / 1e6:.1f}M")'''

SFT = r'''# 10) Hybrid SFT: one model, /think and /no_think
CK2, LOG2 = f"{DRIVE}/ckpt/sft_v2.pkl", f"{DRIVE}/results/sft_log.jsonl"
if os.path.exists(CK2):
    params, opt_state, meta2 = imx.load_ckpt(CK2)
else:
    opt_state, meta2 = tx.init(params), {}
if not meta2.get("done"):
    params, opt_state, meta2, _ = imx.train_loop(params, opt_state, step, imx.SFTBatches(SFT_T, SFT_M),
                                                 imx.wsd_lr(SFT_LR, 30, 0.5), SFT_MIN * 60, ACCUM, MICRO_B,
                                                 ckpt_path=CK2, meta=meta2, log_path=LOG2, log_every=10)
    meta2["done"] = True
    imx.save_ckpt(CK2, params, opt_state, meta2)
print("SFT done:", meta2["step"], "steps")'''

EVAL = r'''# 11) Evaluation: chat samples, reasoning accuracy /think vs /no_think, script fidelity
next_logits = imx.make_next_logits(CFG)
def chat(user, mode="no_think", lang="default", max_new=250):
    out = imx.generate(params, next_logits, tok, imx.render_prompt(user, mode, lang), CFG, max_new=max_new, temp=0)
    return imx.split_reply(out), len(tok.encode(out).ids)

samples = {"kn": "ನನ್ನ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ, ಏನು ಮಾಡಬೇಕು?", "hi": "मेरा रिफंड कब तक आएगा?",
           "en": "Explain what a UPI PIN is in two sentences.", "kanglish": "naale meeting ide, reminder set maadi",
           "hinglish": "kal meeting hai, reminder laga do",
           "kn_math": "ರವಿ ಬಳಿ 500 ರೂಪಾಯಿ ಇತ್ತು. ಅವನು 120 ರೂಪಾಯಿಗೆ ಪುಸ್ತಕ ಮತ್ತು 80 ರೂಪಾಯಿಗೆ ಪೆನ್ನು ಖರೀದಿಸಿದನು. ಎಷ್ಟು ಉಳಿಯಿತು?"}
results = {"samples": {}, "reasoning": {}}
for k, u in samples.items():
    lang = k.split("_")[0]
    for mode in ("no_think", "think"):
        (th, ans), n = chat(u, mode, lang)
        results["samples"][f"{k}/{mode}"] = {"think": th, "answer": ans, "tokens": n,
                                             "same_script": imx.detect_style(ans) == lang or lang == "default"}
        print(f"\n[{k} /{mode}] {n} tokens\nTHINK: {th[:300]}\nANSWER: {ans[:300]}")

val = []
for path in glob.glob(f"{DRIVE}/sft/*val*.jsonl"):
    with open(path, encoding="utf-8") as f:
        val += [r for r in map(json.loads, f) if r.get("mode") == "think"]
val = val[:40]
for mode in ("no_think", "think"):
    ok = toks = 0
    for r in val:
        u = r["prompt"].rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0]
        u = u.replace(" /think", "").replace(" /no_think", "")
        gold = imx.last_number(r["completion"].split("</think>")[-1])
        (th, ans), n = chat(u, mode, r["lang"], 350 if mode == "think" else 120)
        ok += gold is not None and imx.last_number(ans) is not None and abs(imx.last_number(ans) - gold) < 1e-6
        toks += n
    if val:
        results["reasoning"][mode] = {"accuracy": round(ok / len(val), 3), "avg_tokens": round(toks / len(val), 1),
                                      "n": len(val)}
print("\nreasoning:", results["reasoning"] or "no codex_val.jsonl uploaded")
results["fertility"] = table
json.dump(results, open(f"{DRIVE}/results/eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)'''

EXPORT = r'''# 12) Export to Hugging Face / PyTorch (for the laptop demo) + parity check, zipped to Drive
out = f"{LOCAL}/IndicMaxxer-100M"
diff = imx.export_hf(params, CFG, TOK, out, check_tokens=np.array([tok.encode("ಕನ್ನಡ ನಮ್ಮ ಭಾಷೆ. भारत.").ids]))
print("JAX vs PyTorch relative logit diff:", diff)
shutil.make_archive(f"{DRIVE}/export/IndicMaxxer-100M", "zip", out)
print("saved", f"{DRIVE}/export/IndicMaxxer-100M.zip")'''

DEMO = r'''# 13) Quick web demo from Colab (share link), useful if the laptop can't run PyTorch
import gradio as gr
def answer(msg, mode, search):
    th, ans, src = imx.chat_agent(params, next_logits, tok, CFG, msg, mode, search)
    return th, ans, src
gr.Interface(answer, [gr.Textbox(label="Message"), gr.Radio(["no_think", "think"], value="no_think", label="Mode"),
                      gr.Radio(["auto", "always", "off"], value="auto", label="Web search")],
             [gr.Textbox(label="Reasoning"), gr.Textbox(label="Answer"), gr.Textbox(label="Web result used")],
             title="IndicMaxxer-100M").launch(share=True)'''

cells = [("markdown", INTRO), ("code", SETUP), ("code", "%%writefile imx_core.py\n" + CORE), ("code", CONFIG),
         ("code", DATA), ("code", TOKENIZER), ("code", SHARDS), ("code", SMOKE), ("code", PRETRAIN),
         ("code", SFT_DATA), ("code", SFT), ("code", EVAL), ("code", EXPORT), ("code", DEMO)]

def cell(kind, src):
    lines = src.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "TPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell(k, s) for k, s in cells]}
dst = os.path.join(os.path.dirname(HERE), "IndicMaxxer_speedrun.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", dst)

# ---------------------------------------------------------------------------- Kaggle: SFT-only run from the pretrained checkpoint
K_INTRO = """# IndicMaxxer v2 fine-tune on Kaggle (free T4): load pretrain.pkl → SFT (Codex + Aya + small talk) → eval → export
**Before Run all:** Settings → Accelerator **GPU T4 x2**, Internet **on**, Add-ons → Secrets → `HF_TOKEN` (attach it).
Add your dataset (`+ Add Input`) holding: `pretrain.pkl`, `indicmaxxer-48k.json`, `heldout.json`, `fertility.json`,
`codex_train.jsonl`, `codex_val.jsonl`. Outputs land in `/kaggle/working/IndicMaxxer` (download them from the Output panel)."""

K_SETUP = r'''# 1) Setup (Kaggle)
import importlib, subprocess, sys, os, json, time, glob, shutil, random
from collections import Counter
def need(mod, pkg):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for mod, pkg in [("datasets", "datasets"), ("tokenizers", "tokenizers"), ("optax", "optax"), ("regex", "regex"),
                 ("indic_transliteration", "indic-transliteration"), ("safetensors", "safetensors"), ("gradio", "gradio")]:
    need(mod, pkg)
try:
    from transformers import Qwen3Config
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
import jax
if jax.devices()[0].platform != "gpu":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]"])
    raise SystemExit("Installed GPU JAX: click Run -> Restart & clear outputs, then Run all again.")
print("JAX devices:", jax.devices())
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")             # off Kaggle: set the HF_TOKEN environment variable
print("HF token:", "found" if HF_TOKEN else "MISSING (Kaggle: Add-ons -> Secrets -> HF_TOKEN; elsewhere: env HF_TOKEN)")
INPUT = os.environ.get("IMX_INPUT", "/kaggle/input")      # folder holding the uploaded files (any depth)
WORK = os.environ.get("IMX_WORK", "/kaggle/working")
DRIVE, LOCAL = f"{WORK}/IndicMaxxer", f"{WORK}/imx"
for d in ["data", "tok", "ckpt", "export", "sft", "results"]:
    os.makedirs(f"{DRIVE}/{d}", exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)
where = {"pretrain.pkl": "ckpt", "indicmaxxer-48k.json": "tok", "heldout.json": "data", "fertility.json": "results",
         "codex_train.jsonl": "sft", "codex_val.jsonl": "sft", "indicmaxxer_ultimate.jsonl": "data"}
found = {os.path.basename(p): p for p in glob.glob(f"{INPUT}/**/*", recursive=True) if os.path.isfile(p)}
if "pretrain2.pkl" in found:                       # continued-pretraining checkpoint (TPU notebook output) wins
    found["pretrain.pkl"] = found["pretrain2.pkl"]
    print("using pretrain2.pkl (continued pretraining)")
for a, b in [("train.jsonl", "codex_train.jsonl"), ("val.jsonl", "codex_val.jsonl")]:
    if a in found and b not in found:
        found[b] = found[a]
for name, sub in where.items():
    dst = f"{DRIVE}/{sub}/{name}"
    if name not in found:
        print("MISSING input file:", name)
    elif not os.path.exists(dst):
        os.symlink(found[name], dst)
print("inputs linked:", sorted(n for n in where if n in found))'''

K_ULT = r'''# 1b) ULTIMATE dataset: open kn/hi/en/Hinglish/Kanglish chat data + our Codex & hand-written chats (~10 min)
# It replaces codex_train.jsonl in fine-tuning (it already contains it). Skip this cell to use only our own data.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "indic-transliteration"])
own = f"{DRIVE}/data/indicmaxxer_ultimate.jsonl"
out = f"{DRIVE}/sft/ultimate_train.jsonl"
if not os.path.exists(out):
    subprocess.check_call([sys.executable, "build_ultimate.py", "--out", out, "--total", "40000"]
                          + (["--own", own] if os.path.exists(own) else []), env={**os.environ, "HF_TOKEN": HF_TOKEN or ""})
if os.path.exists(out) and os.path.islink(f"{DRIVE}/sft/codex_train.jsonl"):
    os.unlink(f"{DRIVE}/sft/codex_train.jsonl")      # avoid training on Codex twice
print(open(out, encoding="utf-8").readline()[:300])'''

K_LOAD = r'''# 2) Load tokenizer, held-out docs, pretrained weights
from tokenizers import Tokenizer
TOK = f"{DRIVE}/tok/indicmaxxer-48k.json"
tok = Tokenizer.from_file(TOK)
EOS, PAD = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|pad|>")
held = json.load(open(f"{DRIVE}/data/heldout.json", encoding="utf-8"))
table = json.load(open(f"{DRIVE}/results/fertility.json"))
params, _, meta = imx.load_ckpt(f"{DRIVE}/ckpt/pretrain.pkl")
params = jax.device_put(params)
tx = imx.make_optimizer(params)
step = imx.make_train_step(CFG, tx)
print("pretrained:", meta.get("step"), "steps,", round(meta.get("tokens", 0) / 1e6), "M tokens")'''

k_config = CONFIG.replace("MICRO_B, ACCUM = 8, 8 ", "MICRO_B, ACCUM = 4, 16")
k_export = EXPORT.replace("IndicMaxxer-100M", "IndicMaxxer-100M-v2")
k_cells = [("markdown", K_INTRO), ("code", K_SETUP), ("code", "%%writefile imx_core.py\n" + CORE),
           ("code", "%%writefile build_ultimate.py\n" + open(os.path.join(HERE, "build_ultimate.py"), encoding="utf-8").read()),
           ("code", K_ULT), ("code", k_config), ("code", K_LOAD), ("code", SFT_DATA), ("code", SFT), ("code", EVAL), ("code", k_export), ("code", DEMO)]
nbk = {"nbformat": 4, "nbformat_minor": 5,
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
       "cells": [cell(k, s) for k, s in k_cells]}
dstk = os.path.join(os.path.dirname(HERE), "IndicMaxxer_kaggle_sft.ipynb")
with open(dstk, "w", encoding="utf-8") as f:
    json.dump(nbk, f, ensure_ascii=False, indent=1)
print("wrote", dstk)

# ============================================================================ v3: lean, fast, chatty SFT notebook for Kaggle T4
L_INTRO = """# IndicMaxxer SFT (lean) on Kaggle T4: ~45 min total, prints progress every step
**Settings:** Accelerator **GPU T4 x2**, Internet **on**, Secret `HF_TOKEN` attached, input dataset with `pretrain.pkl`,
`indicmaxxer-48k.json`, `heldout.json`, `fertility.json`, `train.jsonl`, `val.jsonl`, `indicmaxxer_ultimate.jsonl`.
Then **Run all**. Every cell prints what it is doing and how long it took; nothing should be silent for more than ~3 min
(except the first training step, which compiles: ~1-3 min)."""

L_CONFIG = r'''# 4) Config
import importlib, numpy as np, jax.numpy as jnp
import imx_core as imx
importlib.reload(imx)
CFG = imx.Cfg()
SFT_MIN = int(os.environ.get("IMX_SFT_MIN", 25))                       # training time budget (minutes)
MICRO_B, ACCUM = int(os.environ.get("IMX_MICRO_B", 4)), int(os.environ.get("IMX_ACCUM", 4))  # T4: 16K tokens/step
SFT_LR = 3e-4
REMAT = True              # recompute activations: fits a 15 GB T4 (False is ~30% faster on bigger GPUs)
ULT_TOTAL = 30000         # rows in the ultimate dataset
print(f"model {CFG.n_params() / 1e6:.1f}M params | budget {SFT_MIN} min | {MICRO_B * ACCUM * CFG.ctx} tokens/step")'''

L_DATA = r'''# 6) Fine-tuning set: ultimate dataset (open data + Codex + hand-written chats) + persona / safety / tools
t0 = time.time()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "indic-transliteration"])
ULT = f"{DRIVE}/sft/ultimate_train.jsonl"
own = f"{DRIVE}/data/indicmaxxer_ultimate.jsonl"
if not os.path.exists(ULT):
    print("building ultimate dataset (each source capped at 2.5 min)...", flush=True)
    r = subprocess.run([sys.executable, "build_ultimate.py", "--out", ULT, "--total", str(ULT_TOTAL)]
                       + (["--own", own] if os.path.exists(own) else []),
                       env={**os.environ, "HF_TOKEN": HF_TOKEN or ""}, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:])
rows = []
if os.path.exists(ULT):
    with open(ULT, encoding="utf-8") as f:
        rows = [(r["prompt"], r["completion"], r["lang"]) for r in map(json.loads, f)]
else:
    for path in glob.glob(f"{DRIVE}/sft/codex_train.jsonl"):
        rows = [(r["prompt"], r["completion"], r["lang"]) for r in map(json.loads, open(path, encoding="utf-8"))] * 2
print("main rows:", len(rows), dict(Counter(r[2] for r in rows)))
aux = imx.build_aux_sft(held, n_ground=300, n_tool=150, safety_repeat=4)
chat_rows, facts, langid = imx.build_smalltalk(), imx.build_fact_sft(held), imx.build_langid_sft(held)
rows += aux + chat_rows + facts + langid
print(f"+ safety/grounding/tools {len(aux)} + small talk {len(chat_rows)} + web-search facts {len(facts)} "
      f"+ language-ID/convert {len(langid)} = {len(rows)} examples")
SFT_T, SFT_M = imx.pack_sft(tok, rows, CFG.ctx, PAD)
print(f"packed {SFT_T.shape[0]} rows of {CFG.ctx + 1}, supervised tokens {SFT_M.sum() / 1e6:.1f}M, {time.time() - t0:.0f}s")'''

L_TRAIN = r'''# 7) Train. First step compiles (~1-3 min), then a log line every 5 steps.
CK2, LOG2 = f"{DRIVE}/ckpt/sft_v3.pkl", f"{DRIVE}/results/sft_v3_log.jsonl"
step = imx.make_train_step(CFG, tx, remat=REMAT)
opt_state, meta2 = tx.init(params), {}
t0 = time.time()
tk, mk = imx.SFTBatches(SFT_T, SFT_M)(ACCUM, MICRO_B)
print("compiling...", flush=True)
params, opt_state, l, g = step(params, opt_state, tk, mk, jnp.float32(1e-6))
print(f"compiled + first step in {time.time() - t0:.0f}s, loss {float(l):.3f}", flush=True)
params, opt_state, meta2, hist = imx.train_loop(params, opt_state, step, imx.SFTBatches(SFT_T, SFT_M, seed=1),
                                                imx.wsd_lr(SFT_LR, 20, 0.4), SFT_MIN * 60, ACCUM, MICRO_B,
                                                ckpt_path=CK2, meta=meta2, log_path=LOG2, log_every=5)
print(f"SFT done: {meta2['step']} steps, {meta2['tokens'] / 1e6:.0f}M tokens, final loss {hist[-1]['loss'] if hist else '?'}")'''

L_EVAL = r'''# 8) Quick eval (~5 min): samples in 5 styles + reasoning accuracy /think vs /no_think on held-out Codex items
next_logits = imx.make_next_logits(CFG)
def chat(user, mode="no_think", lang="default", max_new=160):
    out = imx.generate(params, next_logits, tok, imx.render_prompt(user, mode, lang), CFG, max_new=max_new, temp=0)
    return imx.split_reply(out), len(tok.encode(out).ids)
samples = {"kn": "ನನ್ನ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ, ಏನು ಮಾಡಬೇಕು?", "hi": "मेरा रिफंड कब तक आएगा?",
           "en": "Explain what a UPI PIN is in two sentences.", "kanglish": "naale meeting ide, reminder set maadi",
           "hinglish": "kal meeting hai, reminder laga do", "en_chat": "hii, how are you?",
           "kn_math": "ರವಿ ಬಳಿ 500 ರೂಪಾಯಿ ಇತ್ತು. ಅವನು 120 ರೂಪಾಯಿಗೆ ಪುಸ್ತಕ ಮತ್ತು 80 ರೂಪಾಯಿಗೆ ಪೆನ್ನು ಖರೀದಿಸಿದನು. ಎಷ್ಟು ಉಳಿಯಿತು?"}
results, t0 = {"samples": {}, "reasoning": {}}, time.time()
for k, u in samples.items():
    lang = k.split("_")[0]
    for mode in (("no_think", "think") if k == "kn_math" else ("no_think",)):
        (th, ans), n = chat(u, mode, lang, 220 if mode == "think" else 120)
        results["samples"][f"{k}/{mode}"] = {"think": th, "answer": ans, "tokens": n}
        print(f"\n[{k} /{mode}] {n} tokens\nTHINK: {th[:300]}\nANSWER: {ans[:300]}", flush=True)
val = []
for path in glob.glob(f"{DRIVE}/sft/*val*.jsonl"):
    val += [r for r in map(json.loads, open(path, encoding="utf-8")) if r.get("mode") == "think"]
val = val[:20]
for mode in ("no_think", "think"):
    ok = toks = 0
    for r in val:
        u = r["prompt"].rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0]
        u = u.replace(" /think", "").replace(" /no_think", "")
        gold = imx.last_number(r["completion"].split("</think>")[-1])
        (th, ans), n = chat(u, mode, r["lang"], 250 if mode == "think" else 80)
        pred = imx.last_number(ans)
        ok += gold is not None and pred is not None and abs(pred - gold) < 1e-6
        toks += n
    if val:
        results["reasoning"][mode] = {"accuracy": round(ok / len(val), 3), "avg_tokens": round(toks / len(val), 1), "n": len(val)}
print("\nreasoning:", results["reasoning"], f"| eval took {(time.time() - t0) / 60:.1f} min")
results["fertility"] = table
json.dump(results, open(f"{DRIVE}/results/eval_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)'''

l_export = EXPORT.replace("IndicMaxxer-100M", "IndicMaxxer-100M-v3")
l_cells = [("markdown", L_INTRO), ("code", K_SETUP), ("code", "%%writefile imx_core.py\n" + CORE),
           ("code", "%%writefile build_ultimate.py\n" + open(os.path.join(HERE, "build_ultimate.py"), encoding="utf-8").read()),
           ("code", L_CONFIG), ("code", K_LOAD), ("code", L_DATA), ("code", L_TRAIN), ("code", L_EVAL),
           ("code", l_export), ("code", DEMO)]
nbl = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
       "cells": [cell(k, s) for k, s in l_cells]}
dstl = os.path.join(os.path.dirname(HERE), "IndicMaxxer_SFT_lean.ipynb")
with open(dstl, "w", encoding="utf-8") as f:
    json.dump(nbl, f, ensure_ascii=False, indent=1)
print("wrote", dstl)


# ---------------------------------------------------------------------------- same lean SFT, for a rented A100 (any Jupyter)
A100_ENV = r"""# 0) A100 settings: run this first. Upload the input files into ~/imx_inputs (any sub-folder is fine).
import os, getpass
os.environ.update(IMX_INPUT=os.path.expanduser("~/imx_inputs"), IMX_WORK=os.path.expanduser("~/imx_work"),
                  IMX_DTYPE="bf16", IMX_SFT_MIN="15", IMX_MICRO_B="16", IMX_ACCUM="2")
os.makedirs(os.environ["IMX_WORK"], exist_ok=True)
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass.getpass("Hugging Face READ token (hidden): ")
print("inputs:", os.environ["IMX_INPUT"], "| work:", os.environ["IMX_WORK"])"""
a_cells = [("markdown", L_INTRO.replace("on Kaggle T4", "on a rented A100")), ("code", A100_ENV)] + l_cells[1:]
nba = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
       "cells": [cell(k, s) for k, s in a_cells]}
dsta = os.path.join(os.path.dirname(HERE), "IndicMaxxer_SFT_A100.ipynb")
with open(dsta, "w", encoding="utf-8") as f:
    json.dump(nba, f, ensure_ascii=False, indent=1)
print("wrote", dsta)
