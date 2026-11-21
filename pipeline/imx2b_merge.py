#!/usr/bin/env python3
"""IndicMaxxer-2B step 3: merge LoRA + trained new-token rows into the fused base -> full fp16 model.
Usage: python imx2b_merge.py train_config.json"""
import json, os, sys
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imx2b_train import load_model, LORA_TARGETS

C = json.load(open(sys.argv[1]))
ck = torch.load(C["state_path"], map_location="cpu")
old_n, new_n, r = ck["old_n"], ck["new_n"], ck["lora_r"]
base = load_model(C["model_dir"])
with torch.no_grad():                                    # write trained rows into the real embedding / head
    e, h = base.get_input_embeddings(), base.get_output_embeddings()
    e.weight[old_n:new_n] = ck["new_rows"]["e"].to(e.weight.dtype)
    if "h" in ck["new_rows"] and h is not None and h.weight is not e.weight:
        h.weight[old_n:new_n] = ck["new_rows"]["h"].to(h.weight.dtype)
model = get_peft_model(base, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                        target_modules=LORA_TARGETS))
params = dict(model.named_parameters())
missing = [n for n in ck["state"] if n not in params]
assert not missing, f"LoRA names not found: {missing[:5]}"
with torch.no_grad():
    for n, t in ck["state"].items():
        params[n].copy_(t.to(params[n].dtype))
merged = model.merge_and_unload().half()
merged.config.use_cache = True
os.makedirs(C["out_dir"], exist_ok=True)
merged.save_pretrained(C["out_dir"], safe_serialization=True)
AutoTokenizer.from_pretrained(C["model_dir"]).save_pretrained(C["out_dir"])
json.dump({"step": ck["step"], "tokens": ck["tokens"]}, open(os.path.join(C["out_dir"], "train_stats.json"), "w"))
print(f"IndicMaxxer-2B saved to {C['out_dir']} (step {ck['step']}, {ck['tokens'] / 1e6:.2f}M tokens)")
