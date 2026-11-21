#!/usr/bin/env python3
"""FAST continued pretraining of IndicMaxxer-100M on Kaggle T4 x2 (PyTorch, fp16 tensor cores, DDP, SDPA, fused AdamW).

Why PyTorch here: the T4 has 65 TFLOPs of fp16 tensor cores but only 8 TFLOPs fp32, and JAX on a T4 runs fp32
(no bf16). fp16 autocast + GradScaler uses the tensor cores. Same architecture (Qwen3), weights convert 1:1 with JAX.

Launch:  torchrun --nproc_per_node=2 train_t4.py t4_config.json
Reads    pretrain.pkl (JAX params), token shards *.bin, heldout.json, tokenizer json
Writes   pretrain2.pkl (same JAX format: the SFT notebook loads it directly), log + before/after held-out loss.
"""
import json, math, os, pickle, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

C = json.load(open(sys.argv[1]))
WORLD, RANK = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
if WORLD > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(LOCAL)
DEV = torch.device("cuda", LOCAL)
MAIN = RANK == 0
DT = torch.bfloat16 if C.get("dtype", "fp16") == "bf16" else torch.float16   # bf16 on A100/H100, fp16 on T4
def log(*a):
    if MAIN:
        print(*a, flush=True)

sys.path.insert(0, C["code_dir"])
import imx_core as imx
from transformers import Qwen3Config, Qwen3ForCausalLM
cfg = imx.Cfg()

# ---------------------------------------------------------------- JAX params <-> HF Qwen3 state dict
PAIRS = [("wq", "self_attn.q_proj"), ("wk", "self_attn.k_proj"), ("wv", "self_attn.v_proj"), ("wo", "self_attn.o_proj"),
         ("w_gate", "mlp.gate_proj"), ("w_up", "mlp.up_proj"), ("w_down", "mlp.down_proj")]
NORMS = [("attn_norm", "input_layernorm"), ("mlp_norm", "post_attention_layernorm"),
         ("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm")]

def to_hf(P):
    L = P["layers"]
    sd = {"model.embed_tokens.weight": P["embed"], "lm_head.weight": P["embed"], "model.norm.weight": P["final_norm"]}
    for i in range(cfg.layers):
        pre = f"model.layers.{i}."
        for ours, theirs in NORMS:
            sd[pre + theirs + ".weight"] = L[ours][i]
        for ours, theirs in PAIRS:
            sd[pre + theirs + ".weight"] = np.ascontiguousarray(L[ours][i].T)
    return {k: torch.tensor(np.asarray(v, dtype=np.float32)) for k, v in sd.items()}

def from_hf(model):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    g = lambda i, n: sd[f"model.layers.{i}.{n}.weight"]
    layers = {ours: np.stack([g(i, theirs) for i in range(cfg.layers)]) for ours, theirs in NORMS}
    for ours, theirs in PAIRS:
        layers[ours] = np.stack([g(i, theirs).T for i in range(cfg.layers)])
    return {"embed": sd["model.embed_tokens.weight"], "final_norm": sd["model.norm.weight"], "layers": layers}

def save_jax_ckpt(path, model, meta):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"params": from_hf(model), "opt": None, "meta": meta}, f)
    os.replace(tmp, path)

# ---------------------------------------------------------------- model
ck = pickle.load(open(C["init_ckpt"], "rb"))
meta = dict(ck.get("meta", {}))
run1_tokens = meta.get("tokens", 0)
hcfg = Qwen3Config(vocab_size=cfg.vocab, hidden_size=cfg.d, intermediate_size=cfg.ffn, num_hidden_layers=cfg.layers,
                   num_attention_heads=cfg.heads, num_key_value_heads=cfg.kv_heads, head_dim=cfg.head_dim,
                   max_position_embeddings=cfg.ctx, rope_theta=cfg.rope_theta, rms_norm_eps=cfg.eps,
                   tie_word_embeddings=True, attention_bias=False)
hcfg._attn_implementation = "sdpa"
model = Qwen3ForCausalLM(hcfg)
missing, unexpected = model.load_state_dict(to_hf(ck["params"]), strict=False)
assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
del ck
model.to(DEV).train()
raw = model

# ---------------------------------------------------------------- data
SEQ = C["seq"]
mm = {l: np.memmap(p, dtype=np.uint16, mode="r") for l, p in C["bins"].items()}
langs = list(mm)
w = np.array([C["mix"][l] for l in langs], dtype=np.float64)
w /= w.sum()
rng = np.random.default_rng(C["seed"] + 1000 * RANK)
def batch(B):
    out = np.empty((B, SEQ + 1), dtype=np.int64)
    for i, li in enumerate(rng.choice(len(langs), B, p=w)):
        a = mm[langs[li]]
        s = int(rng.integers(0, len(a) - SEQ - 1))
        out[i] = a[s:s + SEQ + 1]
    t = torch.from_numpy(out).to(DEV, non_blocking=True)
    return t[:, :-1], t[:, 1:]

# ---------------------------------------------------------------- held-out loss (rank 0)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(C["tok"])
EOS = tok.token_to_id("<|endoftext|>")
held = json.load(open(C["heldout"], encoding="utf-8"))
@torch.no_grad()
def held_loss(n_docs=48):
    raw.eval()
    out = {}
    for l in ["kn", "hi", "en", "kanglish", "hinglish"]:
        ids = []
        for d in held.get(l, [])[-n_docs:]:
            ids += tok.encode(d).ids + [EOS]
        n = len(ids) // (SEQ + 1)
        if n == 0:
            continue
        T = torch.tensor(ids[:n * (SEQ + 1)], dtype=torch.long, device=DEV).view(n, SEQ + 1)
        ls = []
        for i in range(0, n, 8):
            x, y = T[i:i + 8, :-1], T[i:i + 8, 1:]
            with torch.autocast("cuda", dtype=DT):
                logits = raw(input_ids=x).logits
            ls.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item())
        out[l] = round(float(np.mean(ls)), 4)
    raw.train()
    return out

before = held_loss() if MAIN else None
log("held-out loss BEFORE:", before)

# ---------------------------------------------------------------- optimizer / speed-ups
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=C["peak_lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)
scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 14, enabled=DT == torch.float16)
if DT == torch.bfloat16:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if C.get("compile", True):
    try:
        model = torch.compile(model)
        log("torch.compile on")
    except Exception as e:
        log("torch.compile skipped:", e)
if WORLD > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL], gradient_as_bucket_view=True)

def lr_at(step, frac):
    """WSD, decays to 10% of peak (Paramanu-style floor)."""
    peak, floor = C["peak_lr"], 0.1 * C["peak_lr"]
    warm = min(1.0, (step + 1) / C["warmup"])
    d = 1.0 if frac < 1 - C["decay_frac"] else max(0.0, (1 - frac) / C["decay_frac"])
    return floor + (peak - floor) * warm * d

# ---------------------------------------------------------------- train
B, A, budget = C["micro_b"], C["accum"], C["minutes"] * 60
tok_per_step = B * A * SEQ * WORLD
log(f"{WORLD} GPU(s) | micro {B} x accum {A} x seq {SEQ} x {WORLD} = {tok_per_step // 1000}K tokens/step | {C['minutes']} min")
step, t0, last, last_ck, tokens = 0, time.time(), time.time(), time.time(), 0
logf = open(C["log"], "a") if MAIN else None
while True:
    el = time.time() - t0
    stop = torch.tensor([1.0 if el >= budget else 0.0], device=DEV)
    if WORLD > 1:
        dist.all_reduce(stop)                      # all ranks stop together
    if stop.item() > 0:
        break
    lr = lr_at(step, el / budget)
    for g in opt.param_groups:
        g["lr"] = lr
    for a in range(A):
        x, y = batch(B)
        sync = a == A - 1 or WORLD == 1
        ctx = model.no_sync() if (WORLD > 1 and not sync) else torch.enable_grad()
        with ctx:
            with torch.autocast("cuda", dtype=DT):
                logits = model(input_ids=x).logits
                # autocast runs cross_entropy in fp32 internally: no extra full-vocab fp32 copy of the logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)) / A
            scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    step += 1
    tokens += tok_per_step
    if step == 1:
        log(f"first step (incl. compile) {time.time() - t0:.0f}s")
    if step % C["log_every"] == 0 and MAIN:
        now = time.time()
        rec = {"step": step, "loss": round(loss.item() * A, 4), "grad_norm": round(float(gn), 3), "lr": round(lr, 6),
               "elapsed_min": round((now - t0) / 60, 1), "tokens_M": round(tokens / 1e6, 1),
               "tok_per_s": int(C["log_every"] * tok_per_step / (now - last))}
        last = now
        print(rec, flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
    if MAIN and time.time() - last_ck > C["ckpt_every_min"] * 60:
        save_jax_ckpt(C["out_ckpt"], raw, {**meta, "step": meta.get("step", 0) + step, "tokens": run1_tokens + tokens,
                                          "elapsed": 0.0, "run1_tokens": run1_tokens})
        last_ck = time.time()
        log(f"checkpoint saved at step {step}")

if MAIN:
    meta_out = {**meta, "step": meta.get("step", 0) + step, "tokens": run1_tokens + tokens, "elapsed": 0.0,
                "run1_tokens": run1_tokens, "done": True}
    save_jax_ckpt(C["out_ckpt"], raw, meta_out)
    after = held_loss()
    res = {"before": before, "after": after, "improvement": {l: round(before[l] - after[l], 4) for l in after if l in before},
           "tokens_this_run": tokens, "tokens_total": run1_tokens + tokens, "steps": step,
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(C["eval_out"], "w"), indent=1)
    print("held-out loss BEFORE:", before)
    print("held-out loss AFTER: ", after)
    print(f"done: +{tokens / 1e6:.0f}M tokens in {res['minutes']} min -> {C['out_ckpt']}")
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
