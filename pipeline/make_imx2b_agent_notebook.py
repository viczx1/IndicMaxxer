#!/usr/bin/env python3
"""Build IndicMaxxer_2B_agent.ipynb (Kaggle T4): agent runtime (rule-based tool router + native multi-pass reasoning)
+ scripted demo + Gradio app.  Usage: python pipeline/make_imx2b_agent_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()

INTRO = """# IndicMaxxer-2B Agent: /instruct tool router + /reasoning in your own language + live token-tax meter
**Kaggle settings:** Accelerator **GPU T4** (one is enough), Internet **on**, Secret `HF_TOKEN` attached.
**Input:** add the output of the *IndicMaxxer-2B build* notebook (Add Input → Your Work → that notebook) so
`IndicMaxxer-2B/` is under `/kaggle/input`. Or set `IMX_MODEL` to a folder / HF repo id in cell 1.

What it does on every message:
1. **Style lock**: detects Kannada / Hindi / English / Kanglish / Hinglish and answers in the same one (one retry if it drifts).
2. **/instruct**: a rule-based router (no extra model call) picks tools: OCR for images (Tesseract kan+hin+eng),
   doc_read for PDF/DOCX/TXT (long documents are chunked, only relevant chunks enter the 8K window),
   web_search (Wikipedia kn/hi/en) for facts/news, doc_write (answer saved as .docx).
3. **/reasoning**: k native-language thoughts in ONE batched generate, then a vote (majority on the number, else consensus).
4. A trace panel shows tools, latency and **tokens: IndicMaxxer-2B vs Sarvam-1 vs GPT-4 cl100k**, for the same text.

Run all; the last cell prints a public `gradio.live` link (valid while the notebook runs)."""

SETUP = r'''# 1) Setup: OCR engine with Kannada + Hindi, Python deps, find the model
import os, sys, glob, json, subprocess
subprocess.run("apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-kan tesseract-ocr-hin "
               "poppler-utils > /dev/null", shell=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "regex",
                       "gradio", "pytesseract", "pypdf", "python-docx", "pdf2image", "tiktoken", "ddgs"])
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (enable GPU T4)")
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass                                    # only needed for the Sarvam-1 tokenizer in the token-tax meter
WORK = os.environ.setdefault("IMX_WORK", "/kaggle/working")
IMX_MODEL = ""                              # optional: a folder path or HF repo id; leave "" to auto-find
def is_model(d):
    return os.path.exists(f"{d}/config.json") and glob.glob(f"{d}/*.safetensors") and \
           os.path.exists(f"{d}/tokenizer_config.json")
cands = []
for root in ["/kaggle/input", WORK]:
    for cfg in glob.glob(f"{root}/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if is_model(d) and "sarvam_fused" not in d:
            cands.append(d)
cands.sort(key=lambda d: ("IndicMaxxer-2B" not in d, d))   # prefer the merged IndicMaxxer-2B folder
MODEL = IMX_MODEL or os.environ.get("IMX_MODEL") or (cands[0] if cands else None)
if not MODEL:
    print("No model folder found. What /kaggle/input contains (3 levels):")
    for dp, dn, fn in os.walk("/kaggle/input"):
        if dp.count("/") <= 5:
            print(" ", dp, fn[:6])
    raise SystemExit("IndicMaxxer-2B not found. The build notebook must be SAVED AS A VERSION (Save & Run All) and "
                     "finish the merge cell; then Add Input -> Your Work -> that notebook. Or set IMX_MODEL above.")
print("model:", MODEL, "| other candidates:", cands[1:4])
print(subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout)'''

LOAD = r'''# 3) Load the agent (~1 min)
sys.path.insert(0, os.getcwd())
import importlib, imx_agent
importlib.reload(imx_agent)
from imx_agent import Agent, trace_md, route, detect_style
agent = Agent(MODEL, hf_token=os.environ.get("HF_TOKEN"))
print("loaded | token-tax meter compares against:", list(agent.ref))'''

DEMO = r'''# 4) Scripted demo: one case per capability (also saved to reports/agent_demo.json for the deck)
import re
os.makedirs(f"{WORK}/reports", exist_ok=True)
import docx                                             # a small Kannada document to read back
d = docx.Document()
for p in ["ಕಂಪನಿ ರಜೆ ನೀತಿ 2026", "ಪ್ರತಿ ಉದ್ಯೋಗಿಗೆ ವರ್ಷಕ್ಕೆ 18 ದಿನ ವೇತನ ಸಹಿತ ರಜೆ ಇದೆ.",
          "ಅನಾರೋಗ್ಯ ರಜೆ 10 ದಿನ. ಮೂರು ದಿನಕ್ಕಿಂತ ಹೆಚ್ಚು ಅನಾರೋಗ್ಯ ರಜೆಗೆ ವೈದ್ಯರ ಪ್ರಮಾಣಪತ್ರ ಬೇಕು.",
          "ಬಳಸದ ರಜೆಯಲ್ಲಿ ಗರಿಷ್ಠ 6 ದಿನ ಮುಂದಿನ ವರ್ಷಕ್ಕೆ ಮುಂದುವರಿಯುತ್ತದೆ."]:
    d.add_paragraph(p)
POLICY = f"{WORK}/leave_policy_kn.docx"; d.save(POLICY)
cases = [
    ("hii, who are you?", "instruct", []),
    ("naale client meeting ide, enu prepare maadli?", "instruct", []),
    ("kal interview hai, kuch tips do", "instruct", []),
    ("ಮೈಸೂರು ಅರಮನೆ ಬಗ್ಗೆ ಹುಡುಕಿ ಹೇಳು", "instruct", []),
    ("इसरो के बारे में खोजो", "instruct", []),
    ("ಅನಾರೋಗ್ಯ ರಜೆಗೆ ಯಾವಾಗ ಪ್ರಮಾಣಪತ್ರ ಬೇಕು?", "instruct", [POLICY]),
    ("Write a short leave request letter to my manager for 2 days and save it as a docx", "instruct", []),
    ("ರವಿ ಬಳಿ 500 ರೂಪಾಯಿ ಇತ್ತು. ಅವನು 120 ರೂಪಾಯಿಗೆ ಪುಸ್ತಕ ಮತ್ತು 80 ರೂಪಾಯಿಗೆ ಪೆನ್ನು ಖರೀದಿಸಿದನು. ಎಷ್ಟು ಉಳಿಯಿತು?", "reasoning", []),
]
log = []
for q, mode, files in cases:
    ans, tr, f = agent.chat(q, mode=mode, k=3, files=files)
    print("=" * 100, f"\nUSER [{mode}]: {q}\nINDICMAXXER: {ans[:600]}")
    print(re.sub(r"<details>.*?</details>", "", trace_md(tr), flags=re.S).strip())
    if f:
        print("wrote", f)
    log.append({"q": q, "mode": mode, "answer": ans, "trace": {k: v for k, v in tr.items() if k != "passes"}, "file": f})
json.dump(log, open(f"{WORK}/reports/agent_demo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)'''

APP = r'''# 5) Gradio app: public link below (keep this cell running during the demo)
import gradio as gr
def respond(msg, history, mode, k, force, files):
    history = list(history or [])
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
    if not (msg or "").strip() and not paths:
        return history, "", "type something or attach a file", None
    try:
        ans, tr, out = agent.chat(msg or "", history, mode, int(k), force, paths)
        md = trace_md(tr)
    except Exception as e:
        ans, md, out = f"(error: {e})", f"**error:** `{e!r}`", None
    shown = (msg or "") + (("  📎 " + ", ".join(os.path.basename(p) for p in paths)) if paths else "")
    history += [{"role": "user", "content": shown}, {"role": "assistant", "content": ans}]
    return history, "", md, out

with gr.Blocks(title="IndicMaxxer-2B") as demo:
    gr.Markdown("## IndicMaxxer-2B · Kannada · ಕನ್ನಡ · हिंदी · English · Kanglish · Hinglish\n"
                "Local 2B model. `/instruct` = direct answer + tools · `/reasoning` = thinks in your language, k times, then votes.")
    with gr.Row():
        with gr.Column(scale=3):
            try:
                chat = gr.Chatbot(type="messages", height=480)
            except TypeError:
                chat = gr.Chatbot(height=480)
            box = gr.Textbox(placeholder="ಏನು ಬೇಕು? / kya chahiye? / ask anything… (append /reasoning or /instruct)",
                             show_label=False)
            files = gr.File(file_count="multiple", type="filepath",
                            label="Attach image (OCR) or PDF / DOCX / TXT (doc_read)")
        with gr.Column(scale=2):
            mode = gr.Radio(["instruct", "reasoning"], value="instruct", label="mode")
            k = gr.Slider(1, 5, value=3, step=1, label="/reasoning passes (k)")
            force = gr.Checkbox(False, label="force web search")
            trace = gr.Markdown("trace appears here")
            out_file = gr.File(label="doc_write output")
    box.submit(respond, [box, chat, mode, k, force, files], [chat, box, trace, out_file])
    gr.Button("clear").click(lambda: ([], "", "", None), None, [chat, box, trace, out_file])
demo.queue().launch(share=True, debug=False)'''

def cell(kind, s):
    c = {"cell_type": kind, "metadata": {}, "source": s.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c

cells = [("markdown", INTRO), ("code", SETUP), ("code", "%%writefile imx_agent.py\n" + src("imx_agent.py")),
         ("code", LOAD), ("code", DEMO), ("code", APP)]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell(k, s) for k, s in cells]}
dst = os.path.join(os.path.dirname(HERE), "IndicMaxxer_2B_agent.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", dst)
