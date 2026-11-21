#!/usr/bin/env python3
"""Build IndicMaxxer_TPU_pretrain.ipynb: continue pretraining IndicMaxxer-100M on a Kaggle TPU v3-8 (8 cores,
data-parallel), resuming from pretrain.pkl. Usage: python pipeline/make_tpu_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = open(os.path.join(HERE, "imx_core.py"), encoding="utf-8").read()

INTRO = """# IndicMaxxer-100M: continued pretraining on Kaggle TPU v3-8 (resumes from `pretrain.pkl`)
**Settings (right panel):** Accelerator **TPU VM v3-8**, Internet **on**.
**Input dataset** must contain: `pretrain.pkl`, `indicmaxxer-48k.json`, `heldout.json`, and the 5 token shards
`kn.bin hi.bin en.bin kanglish.bin hinglish.bin` (from Drive `IndicMaxxer/data/`).

Best way to run (keeps going if you close the tab, and saves the result): **Save Version → Save & Run All (Commit)**.
The new checkpoint is written to `/kaggle/working/IndicMaxxer/ckpt/pretrain2.pkl` (Output tab). In the SFT notebook,
**Add Input → this notebook's output**: it picks `pretrain2.pkl` automatically.

What it reports: held-out loss per language **before and after** (the measured gain), tokens trained, tokens/s, samples."""

SETUP = r'''# 1) Setup: TPU check, inputs
import importlib, subprocess, sys, os, json, time, glob, shutil
def need(mod, pkg):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for mod, pkg in [("tokenizers", "tokenizers"), ("optax", "optax"), ("regex", "regex"), ("datasets", "datasets")]:
    need(mod, pkg)
import jax
if jax.devices()[0].platform != "tpu":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]",
                           "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"])
    raise SystemExit("Installed TPU JAX: Run -> Restart & clear outputs, then Run all again.")
print("devices:", jax.device_count(), jax.devices()[0].device_kind)
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = None
print("HF token:", "found" if HF_TOKEN else "missing (Add-ons -> Secrets -> HF_TOKEN); extra Kannada download may fail")

W = "/kaggle/working/IndicMaxxer"
for d in ["ckpt", "results"]:
    os.makedirs(f"{W}/{d}", exist_ok=True)
found = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*", recursive=True) if os.path.isfile(p)}
need_files = ["pretrain.pkl", "indicmaxxer-48k.json", "heldout.json", "kn.bin", "hi.bin", "en.bin", "kanglish.bin", "hinglish.bin"]
missing = [f for f in need_files if f not in found]
print("found:", [f for f in need_files if f in found])
assert not missing, f"MISSING input files: {missing}  (upload them to the input dataset)"'''

CONFIG = r'''# 3) Config
import numpy as np, jax.numpy as jnp
import imx_core as imx
importlib.reload(imx)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
CFG = imx.Cfg()
CONT_MIN = 90                  # training minutes (total run ~2 h: 15 min extra data + ~5 min tokenize + 90 train + eval)
EXTRA_MIN, EXTRA_MB = 15, 3000   # fresh, strictly cleaned Kannada to download (per source, 2 sources in parallel)
N = jax.device_count()
MICRO_B, ACCUM = 8 * N, 1      # 8 sequences per core x 1024 tokens, 64 x 1024 = 65K tokens/step on v3-8 (same as run 1)
PEAK_LR = 1e-3                 # re-warm lower than run 1 (2e-3): we continue an already-trained model
MIX = {"kn": 0.45, "hi": 0.25, "en": 0.18, "kanglish": 0.06, "hinglish": 0.06}
mesh = Mesh(np.array(jax.devices()), ("data",))
REPL, BATCH = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "data", None))
print(f"{CFG.n_params() / 1e6:.1f}M params | {N} cores | {MICRO_B * ACCUM * CFG.ctx // 1000}K tokens/step | {CONT_MIN} min")'''

LOAD = r'''# 4) Load tokenizer, data, checkpoint (replicated on all cores)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(found["indicmaxxer-48k.json"])
held = json.load(open(found["heldout.json"], encoding="utf-8"))
bins = {l: found[f"{l}.bin"] for l in MIX}
print({l: f"{os.path.getsize(p) / 2e6:.0f}M tokens" for l, p in bins.items()})
CK_OUT, LOG = f"{W}/ckpt/pretrain2.pkl", f"{W}/results/pretrain2_log.jsonl"
resume = os.path.exists(CK_OUT)
params, opt_state, meta = imx.load_ckpt(CK_OUT if resume else found["pretrain.pkl"])
params, opt_state = jax.device_put(params, REPL), jax.device_put(opt_state, REPL)
if not resume:   # new run: keep step/token counters, restart the clock for this run's budget
    meta = {"step": meta.get("step", 0), "tokens": meta.get("tokens", 0), "elapsed": 0.0,
            "run1_tokens": meta.get("tokens", 0)}
print("start from:", "pretrain2.pkl (resume)" if resume else "pretrain.pkl", {k: meta[k] for k in ("step", "tokens", "elapsed")})
tx = imx.make_optimizer(params)
step = imx.make_train_step(CFG, tx, remat=True)'''

MORE = r'''# 4b) MORE KANNADA (Paramanu's lesson: tokens matter most). Two fresh sources in parallel, strict cleaning, ~20 min.
from concurrent.futures import ThreadPoolExecutor
srcs = [("ai4bharat/sangraha", None, "verified/kan", "kn_sangraha"),
        ("ai4bharat/sangraha", None, "unverified/kan", "kn_sangraha_u")]
def fetch(src):
    repo, name, ddir, tag = src
    try:
        st = imx.stream_docs(repo, name, f"/kaggle/working/{tag}.jsonl", EXTRA_MB, "kn", HF_TOKEN,
                             max_minutes=EXTRA_MIN, data_dir=ddir, strict=True)
        return tag, st
    except Exception as e:
        return tag, f"SKIPPED {type(e).__name__}: {str(e)[:150]}"
t0 = time.time()
print("downloading extra Kannada...", flush=True)
with ThreadPoolExecutor(len(srcs)) as ex:
    got = list(ex.map(fetch, srcs))
for tag, st in got:
    print(tag, st, flush=True)
EOS = tok.token_to_id("<|endoftext|>")
for tag, st in got:
    if isinstance(st, dict) and st["docs"] > 1000:
        n = imx.tokenize_to_bin(tok, f"/kaggle/working/{tag}.jsonl", f"/kaggle/working/{tag}.bin", EOS)
        bins[tag] = f"/kaggle/working/{tag}.bin"
        print(f"{tag}: {n / 1e6:.0f}M tokens", flush=True)
# split the Kannada share across all Kannada shards by size (each Kannada token seen about equally often)
kn_bins = [l for l in bins if l.startswith("kn")]
kn_tok = {l: os.path.getsize(bins[l]) / 2 for l in kn_bins}
kn_share = MIX.pop("kn")
for l in kn_bins:
    MIX[l] = kn_share * kn_tok[l] / sum(kn_tok.values())
MIX_HELD = ["kn", "hi", "en", "kanglish", "hinglish"]
print("Kannada tokens:", {l: f"{v / 1e6:.0f}M" for l, v in kn_tok.items()}, f"total {sum(kn_tok.values()) / 1e6:.0f}M")
print("mix:", {l: round(w, 3) for l, w in MIX.items()}, f"| {(time.time() - t0) / 60:.1f} min")'''

HELDLOSS = r'''# 5) Held-out loss per language (BEFORE)  - the number that shows what extra pretraining buys
EOS = tok.token_to_id("<|endoftext|>")
eval_loss = jax.jit(lambda p, t, m: imx.loss_fn(p, t, m, CFG, remat=False))
def held_loss(params, n_docs=64):
    out = {}
    for l in [l for l in ('kn', 'hi', 'en', 'kanglish', 'hinglish') if l in held]:
        ids = []
        for d in held[l][-n_docs:]:
            ids += tok.encode(d).ids + [EOS]
        n = len(ids) // (CFG.ctx + 1)
        T = np.array(ids[:n * (CFG.ctx + 1)], np.int32).reshape(n, CFG.ctx + 1)
        chunk, losses = 2 * N, []                      # small chunks: full-vocab logits are big
        for i in range(0, n - chunk + 1, chunk):
            tb = jax.device_put(T[i:i + chunk], NamedSharding(mesh, P("data", None)))
            losses.append(float(eval_loss(params, tb, jnp.ones(tb.shape, jnp.float32))))
        if losses:
            out[l] = round(float(np.mean(losses)), 4)
    return out
t0 = time.time()
before = held_loss(params)
print("held-out loss BEFORE:", before, f"({time.time() - t0:.0f}s incl. compile)")'''

TRAIN = r'''# 6) Continue pretraining (data-parallel over all TPU cores). Logs every 25 steps, checkpoint every 15 min.
class ShardedMixer(imx.Mixer):
    def __call__(self, A, B):
        t, m = super().__call__(A, B)
        return jax.device_put(t, BATCH), jax.device_put(m, BATCH)
lr_start = meta["step"]
mixer = ShardedMixer(bins, MIX, CFG.ctx, seed=1234 + meta["step"])
t0 = time.time()
print("compiling first step...", flush=True)
params, opt_state, meta, hist = imx.train_loop(params, opt_state, step, mixer, (lambda st, fr, _f=imx.wsd_lr(PEAK_LR, 200, 0.25, floor=0.1 * PEAK_LR), _s0=meta["step"]: _f(st - _s0, fr)),
                                               CONT_MIN * 60, ACCUM, MICRO_B, ckpt_path=CK_OUT, meta=meta,
                                               ckpt_every_s=900, log_every=25, log_path=LOG)
meta["done"] = True
imx.save_ckpt(CK_OUT, params, opt_state, meta)
new_tok = meta["tokens"] - meta.get("run1_tokens", 0)
print(f"done: step {meta['step']}, +{new_tok / 1e6:.0f}M tokens this run, total {meta['tokens'] / 1e6:.0f}M, "
      f"{(time.time() - t0) / 60:.0f} min")'''

AFTER = r'''# 7) AFTER: held-out loss + samples, saved to results/pretrain2_eval.json
after = held_loss(params)
print("held-out loss BEFORE:", before)
print("held-out loss AFTER: ", after)
print("improvement:", {l: round(before[l] - after[l], 4) for l in after if l in before})
single = jax.devices()[0]
p1 = jax.device_put(jax.device_get(params), single)
next_logits = imx.make_next_logits(CFG)
samples = {}
for s in ["ಬೆಂಗಳೂರು ನಗರವು", "ಕನ್ನಡ ಭಾಷೆಯ ಇತಿಹಾಸ", "भारत एक", "The monsoon in India", "naanu ivattu"]:
    samples[s] = imx.generate(p1, next_logits, tok, s, CFG, max_new=50, temp=0.7)
    print(repr(s), "->", samples[s])
json.dump({"before": before, "after": after, "tokens_total": meta["tokens"],
           "tokens_this_run": meta["tokens"] - meta.get("run1_tokens", 0), "steps": meta["step"], "samples": samples},
          open(f"{W}/results/pretrain2_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", CK_OUT)'''

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c

cells = [("markdown", INTRO), ("code", SETUP), ("code", "%%writefile imx_core.py\n" + CORE), ("code", CONFIG),
         ("code", LOAD), ("code", MORE), ("code", HELDLOSS), ("code", TRAIN), ("code", AFTER)]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell(k, s) for k, s in cells]}
dst = os.path.join(os.path.dirname(HERE), "IndicMaxxer_TPU_pretrain.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", dst)
