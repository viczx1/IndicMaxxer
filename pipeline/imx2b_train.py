#!/usr/bin/env python3
"""IndicMaxxer-2B step 2: LoRA fine-tune of the fused Sarvam-1 (fp16 autocast on T4).

Trainable: LoRA on every attention/MLP projection + the full embedding (and output head if untied) in fp32, so fused
Kanglish/Hinglish tokens and chat/think tokens learn. Loss only on the assistant reply.
Saves ONLY the trainable tensors (LoRA + embeddings) to `state_path` every `ckpt_every_min` and at the end, so a crash
still leaves a usable checkpoint. Merging into a full model is a separate step (imx2b_merge.py).

Usage: python imx2b_train.py train_config.json                                  (1 GPU, default, most robust)
       torchrun --standalone --nproc_per_node=2 imx2b_train.py train_config.json (2 GPUs)
"""
import json, math, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

C = json.load(open(sys.argv[1]))
WORLD, RANK = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
if WORLD > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(LOCAL)
DEV = torch.device("cuda", LOCAL)
MAIN = RANK == 0
def log(*a):
    if MAIN:
        print(*a, flush=True)

def load_model(path):
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
class NewRows(torch.nn.Module):
    """Trainable fp32 rows ONLY for the tokens we added (fused words + chat/think tokens).
    The original 68K embedding/head rows stay frozen in fp16: saves ~5 GB of optimizer memory on a T4."""
    def __init__(self, model, old_n, new_n):
        super().__init__()
        e, h = model.get_input_embeddings(), model.get_output_embeddings()
        self.old_n, self.new_n = old_n, new_n
        self.tied = h is None or h.weight is e.weight
        self.e = torch.nn.Parameter(e.weight.data[old_n:new_n].float().clone())
        self.h = None if self.tied else torch.nn.Parameter(h.weight.data[old_n:new_n].float().clone())
        e.register_forward_hook(self._emb_hook)
        if h is not None:
            h.register_forward_hook(self._head_hook)
    def _emb_hook(self, mod, inp, out):
        ids = inp[0]
        m = ids >= self.old_n
        if not m.any():
            return out
        out = out.clone()
        out[m] = self.e[(ids[m] - self.old_n).clamp(max=self.new_n - self.old_n - 1)].to(out.dtype)
        return out
    def _head_hook(self, mod, inp, out):
        w = self.e if self.tied else self.h
        new = (inp[0].to(w.dtype) @ w.t()).to(out.dtype)
        return torch.cat([out[..., :self.old_n], new, out[..., self.new_n:]], dim=-1)

def build(path, r, old_n, new_n):
    """Fused base (frozen fp16) + LoRA (fp32) + trainable new-token rows (fp32)."""
    m = load_model(path)
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                     target_modules=LORA_TARGETS))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    rows = NewRows(m, old_n, new_n)
    return m, rows

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(C["model_dir"])
    OLD_N, NEW_N = C["old_vocab"], len(tok)
    model, rows = build(C["model_dir"], C["lora_r"], OLD_N, NEW_N)
    tied = rows.tied
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(DEV); rows.to(DEV)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in rows.parameters())
    log(f"trainable {n_train / 1e6:.0f}M params (LoRA + {NEW_N - OLD_N} new token rows) | GPUs {WORLD}")

    # ------------------------------------------------------------ data
    data_rows = [json.loads(l) for l in open(C["data"], encoding="utf-8")]
    random.Random(0).shuffle(data_rows)
    data_rows = data_rows[RANK::WORLD]
    L, ex = C["max_len"], []
    for r in data_rows:
        p = tok.encode(r["prompt"], add_special_tokens=False)
        c = tok.encode(r["completion"], add_special_tokens=False)
        if not c or len(c) >= L:
            continue
        p = p[-(L - len(c)):]
        ex.append((p + c, [-100] * len(p) + c))
    log(f"examples per GPU: {len(ex)} | avg length {sum(len(e[0]) for e in ex) / max(len(ex), 1):.0f} tokens")
    B = C["micro_b"]
    def batches(epoch):
        rng = random.Random(epoch * 1000 + RANK)
        idx = list(range(len(ex)))
        rng.shuffle(idx)
        groups = [sorted(idx[i:i + B * 64], key=lambda j: len(ex[j][0])) for i in range(0, len(idx), B * 64)]
        bs = [g[i:i + B] for g in groups for i in range(0, len(g), B)]
        rng.shuffle(bs)
        for b in bs:
            m = max(len(ex[j][0]) for j in b)
            x = torch.full((len(b), m), tok.pad_token_id, dtype=torch.long)
            y = torch.full((len(b), m), -100, dtype=torch.long)
            a = torch.zeros((len(b), m), dtype=torch.long)
            for k, j in enumerate(b):
                ids, lab = ex[j]
                x[k, :len(ids)] = torch.tensor(ids); y[k, :len(lab)] = torch.tensor(lab); a[k, :len(ids)] = 1
            yield x.to(DEV), y.to(DEV), a.to(DEV)

    # ------------------------------------------------------------ optimiser
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad]
    emb_p = list(rows.parameters())
    opt = torch.optim.AdamW([{"params": lora_p, "lr": C["lr"], "weight_decay": 0.0},
                             {"params": emb_p, "lr": C["emb_lr"], "weight_decay": 0.0}], betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda")
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL], static_graph=True) if WORLD > 1 else model

    def save_state(tag):
        if not MAIN:
            return
        sd = {n: p.detach().to("cpu", torch.float16) for n, p in model.named_parameters() if p.requires_grad}
        nr = {k: v.detach().to("cpu", torch.float32) for k, v in rows.state_dict().items()}
        tmp = C["state_path"] + ".tmp"
        torch.save({"state": sd, "new_rows": nr, "old_n": OLD_N, "new_n": NEW_N, "lora_r": C["lora_r"],
                    "tied": tied, "step": step, "tokens": toks}, tmp)
        os.replace(tmp, C["state_path"])
        log(f"saved trainable state ({tag}) at step {step}")

    A, budget = C["accum"], C["minutes"] * 60
    step, t0, toks, epoch, nan_skips, last_ck = 0, time.time(), 0, 0, 0, time.time()
    it = batches(epoch)
    log(f"micro {B} x accum {A} x {WORLD} GPU(s), max_len {L}, budget {C['minutes']} min. first step compiles kernels...")
    while True:
        el = time.time() - t0
        stop = torch.tensor([1.0 if el >= budget else 0.0], device=DEV)
        if WORLD > 1:
            dist.all_reduce(stop)
        if stop.item() > 0:
            break
        frac = el / budget
        mult = min(1.0, (step + 1) / max(1, C["warmup"])) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * mult
        tot = 0.0
        for a in range(A):
            try:
                x, y, am = next(it)
            except StopIteration:
                epoch += 1
                log(f"--- epoch {epoch} complete ---")
                it = batches(epoch)
                x, y, am = next(it)
            ctx = net.no_sync() if (WORLD > 1 and a < A - 1) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = net(input_ids=x, attention_mask=am).logits
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1),
                                           ignore_index=-100) / A
                scaler.scale(loss).backward()
            tot += float(loss.detach())
            toks += int(am.sum()) * WORLD
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(lora_p + emb_p, 1.0)
        if not math.isfinite(tot):                     # fp16 overflow: skip this step, keep going
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            scaler.update(scaler.get_scale() / 2)
            if nan_skips > 50:
                log("too many non-finite losses: stopping early (try lower lr)")
                break
            continue
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        if step == 1:
            log(f"first step done in {time.time() - t0:.0f}s")
        if step % C["log_every"] == 0 and MAIN:
            now = time.time()
            print({"step": step, "loss": round(tot, 4), "grad_norm": round(float(gn), 3), "lr_mult": round(mult, 3),
                   "epoch": epoch, "min": round((now - t0) / 60, 1), "tokens_M": round(toks / 1e6, 2),
                   "tok_per_s": int(toks / (now - t0)), "nan_skips": nan_skips}, flush=True)
        if time.time() - last_ck > C["ckpt_every_min"] * 60:
            save_state("periodic")
            last_ck = time.time()
    save_state("final")
    log(f"training done: {step} steps, {toks / 1e6:.2f}M tokens, {(time.time() - t0) / 60:.1f} min, epochs {epoch}")
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()
