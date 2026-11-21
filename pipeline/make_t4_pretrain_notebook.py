#!/usr/bin/env python3
"""Build IndicMaxxer_T4_FAST_pretrain.ipynb (Kaggle T4 x2, PyTorch fp16 DDP). Usage: python pipeline/make_t4_pretrain_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = open(os.path.join(HERE, "imx_core.py"), encoding="utf-8").read()
TRAIN = open(os.path.join(HERE, "train_t4.py"), encoding="utf-8").read()

INTRO = """# IndicMaxxer-100M: FAST continued pretraining on Kaggle **T4 x2** (fp16 tensor cores, both GPUs)
**Settings:** Accelerator **GPU T4 x2**, Internet on. Input dataset with `pretrain.pkl`, `indicmaxxer-48k.json`,
`heldout.json` and the 5 shards `kn.bin hi.bin en.bin kanglish.bin hinglish.bin`.
Run with **Save Version → Save & Run All (Commit)** so it survives a closed tab. Output: `IndicMaxxer/ckpt/pretrain2.pkl`
→ add this notebook's output as input to the SFT lean notebook (it picks `pretrain2.pkl` automatically).

Speed tricks: PyTorch fp16 autocast (T4 tensor cores, ~8x fp32), both GPUs (DDP), SDPA attention, fused AdamW,
torch.compile, 512-token windows. Prints held-out loss before/after and tokens/s."""

SETUP = r'''# 1) Setup + inputs
import os, glob, json, subprocess, sys, torch
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 1, "No GPU: set Accelerator to GPU T4 x2"
try:
    from transformers import Qwen3Config
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "regex", "optax"])  # optax: to unpickle the JAX checkpoint
W = "/kaggle/working/IndicMaxxer"
for d in ["ckpt", "results"]:
    os.makedirs(f"{W}/{d}", exist_ok=True)
found = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*", recursive=True) if os.path.isfile(p)}
need = ["pretrain.pkl", "indicmaxxer-48k.json", "heldout.json", "kn.bin", "hi.bin", "en.bin", "kanglish.bin", "hinglish.bin"]
missing = [f for f in need if f not in found]
assert not missing, f"MISSING input files: {missing}"
print("inputs ok")'''

CONFIG = r'''# 4) Config: the only cell you need to touch
MINUTES = 90          # training time (your budget: 1.5-2 h total incl. ~5 min setup/compile + ~3 min eval)
conf = {
    "code_dir": "/kaggle/working", "init_ckpt": found["pretrain.pkl"], "tok": found["indicmaxxer-48k.json"],
    "heldout": found["heldout.json"],
    "bins": {l: found[f"{l}.bin"] for l in ["kn", "hi", "en", "kanglish", "hinglish"]},
    "mix": {"kn": 0.45, "hi": 0.25, "en": 0.18, "kanglish": 0.06, "hinglish": 0.06},
    "seq": 512, "micro_b": 8, "accum": 6,           # 8 x 512 x 6 x 2 GPUs = ~49K tokens / step (fits a 15 GB T4)
    "peak_lr": 8e-4, "warmup": 100, "decay_frac": 0.25,
    "minutes": MINUTES, "log_every": 20, "ckpt_every_min": 15, "seed": 7, "compile": True,
    "out_ckpt": f"{W}/ckpt/pretrain2.pkl", "log": f"{W}/results/pretrain2_log.jsonl",
    "eval_out": f"{W}/results/pretrain2_eval.json",
}
json.dump(conf, open("t4_config.json", "w"), indent=1)
print(json.dumps({k: v for k, v in conf.items() if k not in ("bins",)}, indent=1))'''

RUN = r'''# 5) Train on all GPUs (live log). Out of memory: micro_b 4, accum 12, then rerun cells 4-5.
n = torch.cuda.device_count()
!torchrun --standalone --nproc_per_node={n} train_t4.py t4_config.json'''

RESULT = r'''# 6) Result
r = json.load(open(f"{W}/results/pretrain2_eval.json"))
print(json.dumps(r, indent=1))
print("checkpoint:", f"{W}/ckpt/pretrain2.pkl", round(os.path.getsize(f"{W}/ckpt/pretrain2.pkl") / 1e9, 2), "GB")'''

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c

cells = [("markdown", INTRO), ("code", SETUP), ("code", "%%writefile imx_core.py\n" + CORE),
         ("code", "%%writefile train_t4.py\n" + TRAIN), ("code", CONFIG), ("code", RUN), ("code", RESULT)]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell(k, s) for k, s in cells]}
dst = os.path.join(os.path.dirname(HERE), "IndicMaxxer_T4_FAST_pretrain.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", dst)

# ============================================================================ A100 version (any rented Jupyter GPU box)
A_INTRO = """# IndicMaxxer-100M: continued pretraining on a rented **A100** (bf16, ~200-300K tokens/s expected)
1. Upload into `~/imx_inputs`: `pretrain.pkl`, `indicmaxxer-48k.json`, `heldout.json`, `kn.bin hi.bin en.bin kanglish.bin hinglish.bin`.
2. Run all. Cell 3 pulls fresh, strictly cleaned Kannada (Sangraha) for ~12 min while nothing else runs.
3. Output `~/imx_work/IndicMaxxer/ckpt/pretrain2.pkl` is also copied into `~/imx_inputs`, so `IndicMaxxer_SFT_A100.ipynb`
   (run next, same machine) picks it up automatically. **Download results before you stop the instance.**"""

A_SETUP = r'''# 1) Setup
import os, glob, json, subprocess, sys, getpass, shutil, time
IN, WORKD = os.path.expanduser("~/imx_inputs"), os.path.expanduser("~/imx_work")
W = f"{WORKD}/IndicMaxxer"
for d in ["ckpt", "results"]:
    os.makedirs(f"{W}/{d}", exist_ok=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "tokenizers", "regex",
                       "optax", "datasets", "indic-transliteration"])
import torch
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
HF_TOKEN = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face READ token (hidden): ")
found = {os.path.basename(p): p for p in glob.glob(f"{IN}/**/*", recursive=True) if os.path.isfile(p)}
need = ["pretrain.pkl", "indicmaxxer-48k.json", "heldout.json", "kn.bin", "hi.bin", "en.bin", "kanglish.bin", "hinglish.bin"]
missing = [f for f in need if f not in found]
assert not missing, f"MISSING in {IN}: {missing}"
print("inputs ok")'''

A_MORE = r'''# 3) More Kannada (Paramanu lesson: tokens matter most): Sangraha, 2 streams in parallel, strict cleaning, ~12 min
sys.path.insert(0, os.getcwd())
import importlib, imx_core as imx
importlib.reload(imx)
from tokenizers import Tokenizer
from concurrent.futures import ThreadPoolExecutor
tok = Tokenizer.from_file(found["indicmaxxer-48k.json"])
EOS = tok.token_to_id("<|endoftext|>")
srcs = [("verified/kan", "kn_sangraha"), ("unverified/kan", "kn_sangraha_u")]
def fetch(s):
    ddir, tag = s
    try:
        return tag, imx.stream_docs("ai4bharat/sangraha", None, f"{WORKD}/{tag}.jsonl", 3000, "kn", HF_TOKEN,
                                    max_minutes=12, data_dir=ddir, strict=True)
    except Exception as e:
        return tag, f"SKIPPED {type(e).__name__}: {str(e)[:150]}"
t0 = time.time()
with ThreadPoolExecutor(2) as ex:
    got = list(ex.map(fetch, srcs))
extra = {}
for tag, st in got:
    print(tag, st, flush=True)
    if isinstance(st, dict) and st["docs"] > 1000:
        n = imx.tokenize_to_bin(tok, f"{WORKD}/{tag}.jsonl", f"{WORKD}/{tag}.bin", EOS)
        extra[tag] = f"{WORKD}/{tag}.bin"
        print(f"{tag}: {n / 1e6:.0f}M tokens", flush=True)
print(f"{(time.time() - t0) / 60:.1f} min")'''

A_CONFIG = r'''# 4) Config (the only cell to touch)
MINUTES = 100
bins = {l: found[f"{l}.bin"] for l in ["kn", "hi", "en", "kanglish", "hinglish"]}
bins.update(extra)
mix = {"hi": 0.25, "en": 0.18, "kanglish": 0.06, "hinglish": 0.06}
kn = {l: os.path.getsize(p) / 2 for l, p in bins.items() if l.startswith("kn")}
for l, n in kn.items():
    mix[l] = 0.45 * n / sum(kn.values())                       # Kannada share spread over all Kannada data
conf = {"code_dir": os.getcwd(), "init_ckpt": found["pretrain.pkl"], "tok": found["indicmaxxer-48k.json"],
        "heldout": found["heldout.json"], "bins": bins, "mix": mix, "dtype": "bf16",
        "seq": 1024, "micro_b": 32, "accum": 2,                # 32 x 1024 x 2 = 65K tokens/step (same as run 1)
        "peak_lr": 8e-4, "warmup": 100, "decay_frac": 0.25, "minutes": MINUTES, "log_every": 25,
        "ckpt_every_min": 15, "seed": 11, "compile": True,
        "out_ckpt": f"{W}/ckpt/pretrain2.pkl", "log": f"{W}/results/pretrain2_log.jsonl",
        "eval_out": f"{W}/results/pretrain2_eval.json"}
json.dump(conf, open("a100_config.json", "w"), indent=1)
print("Kannada tokens:", {l: f"{n / 1e6:.0f}M" for l, n in kn.items()}, "| mix:", {l: round(v, 3) for l, v in mix.items()})'''

A_RUN = r'''# 5) Train (live log). Out of memory -> micro_b 16, accum 4.
!torchrun --standalone --nproc_per_node=1 train_t4.py a100_config.json'''

A_RESULT = r'''# 6) Result + hand-off to the SFT notebook
r = json.load(open(f"{W}/results/pretrain2_eval.json"))
print(json.dumps(r, indent=1))
shutil.copy(f"{W}/ckpt/pretrain2.pkl", f"{IN}/pretrain2.pkl")
print("copied pretrain2.pkl into", IN, "-> now run IndicMaxxer_SFT_A100.ipynb")'''

a_cells = [("markdown", A_INTRO), ("code", A_SETUP), ("code", "%%writefile imx_core.py\n" + CORE),
           ("code", "%%writefile train_t4.py\n" + TRAIN), ("code", A_MORE), ("code", A_CONFIG), ("code", A_RUN),
           ("code", A_RESULT)]
nba = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
       "cells": [cell(k, s) for k, s in a_cells]}
dsta = os.path.join(os.path.dirname(HERE), "IndicMaxxer_A100_pretrain.ipynb")
with open(dsta, "w", encoding="utf-8") as f:
    json.dump(nba, f, ensure_ascii=False, indent=1)
print("wrote", dsta)
