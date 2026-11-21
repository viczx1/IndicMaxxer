#!/usr/bin/env python3
"""IndicMaxxer-2B local backend: streaming chat (tools + /reasoning) and Indic TTS for the voice orb.

Run:  .venv\\Scripts\\python.exe -m uvicorn server:app --port 8000
Env:  IMX_MODEL  model folder (default ../model/IndicMaxxer-2B)
      IMX_4BIT   1 = NF4 4-bit (default, fits a 4 GB GPU); 0 = fp16
"""
import asyncio, io, json, os, re, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("IMX_WORK", os.path.join(HERE, "work"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # Xet downloads stalled twice on this network: use plain HTTP
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "pipeline")))

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

import imx_agent as ia

MODEL = os.environ.get("IMX_MODEL", os.path.join(HERE, "..", "model", "IndicMaxxer-2B"))
STATE = {"agent": None, "status": "starting", "error": None, "lock": threading.Lock()}

def load():
    if not os.path.exists(os.path.join(MODEL, "config.json")):
        STATE.update(status="no_model", error=f"model not found at {os.path.abspath(MODEL)}")
        return
    try:
        STATE["status"] = "loading"
        STATE["agent"] = ia.Agent(MODEL, hf_token=os.environ.get("HF_TOKEN"),
                                  load_4bit=os.environ.get("IMX_4BIT", "1") == "1" and torch.cuda.is_available())
        STATE["status"] = "ready"
    except Exception as e:
        STATE.update(status="error", error=repr(e))

app = FastAPI(title="IndicMaxxer-2B")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TOKS = {}                                     # tokenizers only (no model needed): live token-tax playground

def load_tokenizers():
    from transformers import AutoTokenizer
    try:
        TOKS["IndicMaxxer-2B"] = AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        pass
    try:
        TOKS["Sarvam-1"] = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
    except Exception:
        pass

@app.on_event("startup")
def _start():
    threading.Thread(target=load_tokenizers, daemon=True).start()

    def load_all():
        load()          # LLM first: its loader changes torch's default dtype while it runs
        mithi()
        load_compressor()
    threading.Thread(target=load_all, daemon=True).start()

# ------------------------------------------------------------------ token compressor: LLMLingua-2 (multilingual, no training)
COMP = {"pc": None, "status": "off"}
_comp_lock = threading.Lock()

def load_compressor():
    try:
        COMP["status"] = "loading"
        from llmlingua import PromptCompressor
        COMP["pc"] = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                                      use_llmlingua2=True, device_map="cpu")
        COMP["status"] = "ready"
        ia.Agent.compressor = staticmethod(compress)
    except Exception as e:
        COMP["status"] = f"error: {e!r}"[:200]

def compress(text, rate=0.5):
    """Drop the words that carry least information (token classifier), keep punctuation, line breaks and digits."""
    with _comp_lock:
        return COMP["pc"].compress_prompt(text, rate=rate, force_tokens=["\n", ".", "?", "!", "।", ","],
                                          force_reserve_digit=True)["compressed_prompt"]

class TaxReq(BaseModel):
    text: str
    compress: bool = False

@app.post("/api/tax")
def tax(req: TaxReq):
    """Token count + the actual pieces each tokenizer produces, for the landing-page playground."""
    text = req.text[:4000]
    words = max(1, len(ia.WORD.findall(text)))
    out = []
    for name, t in TOKS.items():
        ids = t.encode(text, add_special_tokens=False)
        pieces = [p.replace("▁", " ") for p in t.convert_ids_to_tokens(ids[:160])]
        out.append({"name": name, "tokens": len(ids), "per_word": round(len(ids) / words, 2), "pieces": pieces})
    short = None
    if req.compress and COMP["pc"] is not None and "IndicMaxxer-2B" in TOKS and text.strip():
        short = compress(text, 0.5)
        t = TOKS["IndicMaxxer-2B"]
        ids = t.encode(short, add_special_tokens=False)
        out.append({"name": "IndicMaxxer-2B + compressor", "tokens": len(ids), "per_word": round(len(ids) / words, 2),
                    "pieces": [p.replace("▁", " ") for p in t.convert_ids_to_tokens(ids[:160])]})
    return {"words": words, "style": ia.detect_style(text) if text.strip() else None, "results": out,
            "compressed": short, "compressor": COMP["status"]}

@app.get("/api/health")
def health():
    a = STATE["agent"]
    return {"status": STATE["status"], "error": STATE["error"], "model": os.path.abspath(MODEL),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "quant": "nf4" if a is not None and getattr(a.lm, "is_loaded_in_4bit", False) else "fp16",
            "tax_refs": list(a.ref) if a else [], "compressor": COMP["status"]}

def sse(kind, **data):
    return f"data: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\n\n"

def _stream_one(agent, msgs, max_new=384):
    """Yield (kind, text) where kind is 'think' or 'token', from a single streamed generation."""
    text = agent.tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = agent.tk(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -7000:].to(agent.lm.device)
    st = TextIteratorStreamer(agent.tk, skip_prompt=True, skip_special_tokens=False, timeout=120)
    kw = dict(input_ids=ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=6, streamer=st,
              eos_token_id=agent.eos, pad_token_id=agent.tk.pad_token_id)
    th = threading.Thread(target=lambda: agent.lm.generate(**kw), daemon=True)
    th.start()
    full, sent_t, sent_a = "", 0, 0
    for piece in st:
        full += piece
        for sp in ["<|im_end|>", agent.tk.eos_token or "\0"]:
            full = full.replace(sp, "")
        head = full.lstrip()
        if "</think>" in full:
            think, ans = full.split("</think>", 1)
            think, ans = think.replace("<think>", ""), ans.lstrip("\n ")
        elif head.startswith("<think>") or "<think>".startswith(head):
            think, ans = head.replace("<think>", ""), ""          # still inside the think block
        else:
            think, ans = "", full                                  # model skipped the think block
        if len(think) > sent_t:
            yield "think", think[sent_t:]; sent_t = len(think)
        if len(ans) > sent_a:
            yield "token", ans[sent_a:]; sent_a = len(ans)
    th.join()
    yield "n_in", str(ids.shape[1])

@app.post("/api/chat")
async def chat(message: str = Form(""), history: str = Form("[]"), mode: str = Form("instruct"), k: int = Form(3),
               force_search: bool = Form(False), compress: bool = Form(True), docs: str = Form("[]"), style: str = Form(""), voice: bool = Form(False),
               files: list[UploadFile] = File(default=[])):
    agent = STATE["agent"]
    paths = []
    for f in files or []:
        d = tempfile.mkdtemp(prefix="imx_")
        p = os.path.join(d, os.path.basename(f.filename or "upload"))
        with open(p, "wb") as out:
            out.write(await f.read())
        paths.append(p)
    hist = json.loads(history or "[]")
    pick = style or None                                  # language chosen in the UI (voice page)

    async def gen():
        if agent is None:
            yield sse("error", message=f"Model not ready ({STATE['status']}). {STATE['error'] or ''}")
            return
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def work():
            put = lambda kind, **d: loop.call_soon_threadsafe(q.put_nowait, sse(kind, **d))
            try:
                with STATE["lock"]:                   # one generation at a time on a laptop GPU
                    t0 = time.time()
                    put("status", text="routing")
                    text, style, md, msgs, trace = agent.prepare(message, hist, mode, force_search, paths, compress, json.loads(docs or "[]")[-3:], pick)
                    if voice:                                  # spoken turn: short, chatty, fast to first audio
                        msgs[0]["content"] += {"kn": " ಇದು ಮಾತುಕತೆ: 1-2 ಚಿಕ್ಕ ವಾಕ್ಯಗಳಲ್ಲಿ ಸ್ನೇಹಪೂರ್ವಕವಾಗಿ ಉತ್ತರಿಸು.",
                                               "hi": " यह बातचीत है: 1-2 छोटे वाक्यों में दोस्ताना जवाब दो।"}.get(
                            {"kanglish": "kn", "hinglish": "hi"}.get(style, style), " This is a spoken chat: reply in 1-2 short, friendly sentences.")
                    put("meta", style=style, mode=md, tools=trace["tools"], sources=trace["sources"])
                    if trace.get("file_error") and not trace.get("file_ok"):     # say so instead of inventing content
                        f = trace["file_error"]
                        msg = {"kn": f"ಕ್ಷಮಿಸಿ, {f} ಅನ್ನು ಓದಲು ಆಗಲಿಲ್ಲ. ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರ ಅಥವಾ PDF ಕಳುಹಿಸಿ.",
                               "hi": f"माफ़ कीजिए, मैं {f} पढ़ नहीं पाया। कृपया साफ़ तस्वीर या PDF भेजें।",
                               "kanglish": f"Sorry, {f} odoke aagilla. Clear aagiro photo athva PDF kalsi.",
                               "hinglish": f"Sorry, main {f} padh nahi paaya. Clear photo ya PDF bhejo."}.get(
                            style, f"Sorry, I couldn't read {f}. Please send a clearer image or a PDF.")
                        put("token", text=msg)
                        put("done", answer=msg, trace={**trace, "seconds": round(time.time() - t0, 1)}, file=None)
                        return
                    if md == "reasoning":
                        put("status", text=f"thinking natively x{max(1, k)}")
                        ans, tr, out_file = agent.finish(text, style, md, msgs, trace, k, t0)
                        if tr.get("passes"):
                            put("passes", passes=[{"think": th, "answer": a} for th, a in tr["passes"]], vote=tr["vote"])
                        put("think", text=tr.get("think", ""))
                        for i in range(0, len(ans), 6):
                            put("token", text=ans[i:i + 6])
                    else:
                        put("status", text="generating")
                        ans, think, n_in, n_out = "", "", 0, 0
                        for kind, piece in _stream_one(agent, msgs, max_new=110 if voice else 384):
                            if kind == "n_in":
                                n_in = int(piece)
                            elif kind == "think":
                                think += piece; put("think", text=piece)
                            else:
                                ans += piece; n_out += 1; put("token", text=piece)
                        if "</think>" in ans:                     # a second think block leaked: keep the final answer
                            extra, ans = ans.rsplit("</think>", 1)
                            think += "\n" + extra
                        ans = ans.strip()
                        n_out = agent.ntok(think + ans)
                        if not ia.style_ok(style, ans):     # style lock: regenerate once and replace
                            msgs[0]["content"] += f" Reply ONLY in {ia.STYLE_NAME[style]}."
                            (_, a2, n2), = agent.generate(msgs, max_new=384, temp=0.3)[0]
                            if ia.style_ok(style, a2):
                                ans, n_out = a2, n_out + n2
                                trace["tools"].append("style_lock: regenerated once")
                                put("replace", text=ans)
                        out_file = None
                        if any(t == "doc_write" for t, _ in ia.route(text, (), False)):
                            out_file = ia.doc_write(text, ans)
                            trace["tools"].append(f"doc_write -> {os.path.basename(out_file)}")
                        dt = time.time() - t0
                        trace.update(think=think, prompt_tokens=n_in, new_tokens=n_out, seconds=round(dt, 1),
                                     tok_per_s=round(n_out / max(dt, 1e-6), 1), tax=agent.tax(text + "\n" + ans))
                        tr = trace
                    tr = {k2: v for k2, v in tr.items() if k2 != "passes"}
                    g = agent.ref.get("Sarvam-1")
                    if g is not None:      # everything processed this turn vs stock Sarvam-1 (its tokenizer, no compressor)
                        prompt = agent.tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                        base = len(g.encode(prompt, add_special_tokens=False)) + tr.get("compress", {}).get("base_removed", 0)
                        tr["processed"] = {"IndicMaxxer-2B": int(tr.get("prompt_tokens") or 0) + agent.ntok(ans),
                                           "Sarvam-1": base + len(g.encode(ans, add_special_tokens=False))}
                    put("done", answer=ans, trace={k2: v for k2, v in tr.items() if k2 != "docs"}, docs=tr.get("docs") or [],
                        file=f"/api/files/{os.path.basename(out_file)}" if out_file else None)
            except Exception as e:
                put("error", message=repr(e))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/files/{name}")
def files(name: str):
    p = os.path.join(ia.OUT_DIR, os.path.basename(name))
    return FileResponse(p, filename=os.path.basename(p)) if os.path.exists(p) else JSONResponse({"error": "not found"}, 404)

# ------------------------------------------------------------------ TTS: Mithi v1 Flash (99M, Apache-2.0, Kokoro-based)
MITHI_DIR = os.path.abspath(os.path.join(HERE, "..", "tts", "mithi-v1-flash"))
_tts = {"m": None}
_tts_lock = threading.Lock()

def mithi():
    with _tts_lock:
        if _tts["m"] is None:
            sys.path.insert(0, MITHI_DIR)
            import mithi_tts.frontend as fe
            from misaki import espeak
            fe._short_espeak_path()
            for l in ("kn", "hi"):
                fe._g2p[l] = espeak.EspeakG2P(language=l)
            fe._g2p["en"] = espeak.EspeakG2P(language="en-us")   # espeak for English: spaCy is blocked by Smart App Control
            fe._init_g2p = lambda lang: None
            from mithi_tts import Mithi
            torch.set_num_threads(max(4, (os.cpu_count() or 8) // 2))   # physical cores; uvicorn threads can lower it
            print("TTS cpu threads:", torch.get_num_threads(), flush=True)
            m = Mithi(MITHI_DIR, device=os.environ.get("IMX_TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
            for mod in (m.kokoro, m.writer, m.text_enc):     # stay fp32 even if another thread changed the default dtype
                mod.float()
            m.packs = {k: v.float() for k, v in m.packs.items()}
            _tts["m"] = m
        return _tts["m"]

def to_script(text, style):
    """Kanglish / Hinglish are read by the Kannada / Hindi voice after transliteration (rough, but pronounceable)."""
    if style not in ("kanglish", "hinglish"):
        return text
    from indic_transliteration import sanscript
    target = sanscript.KANNADA if style == "kanglish" else sanscript.DEVANAGARI
    return sanscript.transliterate(text.lower(), sanscript.ITRANS, target)

class TTSReq(BaseModel):
    text: str
    style: str | None = None
    voice: str = "charvi"

EMOJI = re.compile(r"[🀀-🫿☀-➿]")

@app.post("/api/tts")
def tts(req: TTSReq):
    style = req.style or ia.detect_style(req.text)
    lang = {"kanglish": "kn", "hinglish": "hi"}.get(style, style if style in ("kn", "hi", "en") else "en")
    text = EMOJI.sub(" ", re.sub(r"[*_`#>\[\]]", " ", to_script(req.text, style))).strip()[:800]
    if not re.search(r"\w", text):
        return Response(status_code=204)
    t0 = time.time()
    m = mithi()
    t1 = time.time()
    with _tts_lock:
        t2 = time.time()
        wav = m.synthesize(text, lang=lang, voice=req.voice if req.voice in m.voices else "charvi")
    print(f"tts timing: get={t1-t0:.2f}s lockwait={t2-t1:.2f}s synth={time.time()-t2:.2f}s", flush=True)
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), 24000, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav", headers={"X-Lang": lang})

# ------------------------------------------------------------------ STT: local Whisper (works in any browser, incl. Brave)
STT_ID = {"kn": "vasista22/whisper-kannada-small", "hi": "vasista22/whisper-hindi-small", "en": "openai/whisper-small"}
STT_DIR = os.path.abspath(os.path.join(HERE, "..", "stt"))
for _l, _d in {"hi": "whisper-hindi-small", "en": "whisper-small", "kn": "whisper-kannada-small"}.items():
    if os.path.exists(os.path.join(STT_DIR, _d, "config.json")):      # local copies (fetch_repo.py) win over the hub
        STT_ID[_l] = os.path.join(STT_DIR, _d)
STT_LANG = {"kn": "kannada", "hi": "hindi", "en": "english"}
_stt = {"lang": None, "proc": None, "model": None}
_stt_lock = threading.Lock()

def stt_model(lang):
    """One Whisper on the GPU at a time (4 GB card): switching language swaps it (~3 s)."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    if _stt["lang"] != lang:
        _stt.update(lang=None, proc=None, model=None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = WhisperProcessor.from_pretrained(STT_ID[lang])
        model = WhisperForConditionalGeneration.from_pretrained(
            STT_ID[lang], dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        _stt.update(lang=lang, proc=proc, model=model)
    return _stt["proc"], _stt["model"]

@app.post("/api/stt")
async def stt(audio: UploadFile = File(...), lang: str = Form("en"), verify: bool = Form(True)):
    lang = lang if lang in STT_ID else "en"
    wav = read_wav16k(await audio.read())
    if len(wav) < 16000 * 0.3:
        return {"text": "", "lang": lang}
    if verify:                                        # only-my-voice: ignore other speakers / TV / background talk
        sim = await asyncio.get_running_loop().run_in_executor(None, voice_match, wav)
        if sim is not None and sim < SPK_THR:
            return {"text": "", "lang": lang, "rejected": True, "similarity": round(sim, 3)}

    def run():
        with _stt_lock:
            t0 = time.time()
            proc, model = stt_model(lang)
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(model.device, model.dtype)
            with torch.no_grad():
                try:
                    ids = model.generate(feats, language=STT_LANG[lang], task="transcribe", max_new_tokens=220)
                except Exception:                     # older fine-tunes without language tokens in their config
                    ids = model.generate(feats, max_new_tokens=220)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0]
            text = re.sub(r"<\|[^|>]*\|>", "", text).strip()      # the fine-tunes leak <|kn|><|transcribe|>… tags
            return text, round(time.time() - t0, 2)
    text, secs = await asyncio.get_running_loop().run_in_executor(None, run)
    return {"text": text, "lang": lang, "seconds": secs, "audio_seconds": round(len(wav) / 16000, 2)}

@app.post("/api/stt/warm")
def stt_warm(lang: str = Form("en")):
    """Load the recogniser for a language before the user speaks."""
    with _stt_lock:
        stt_model(lang if lang in STT_ID else "en")
    return {"ok": True}

# ------------------------------------------------------------------ only-my-voice: WavLM speaker verification (CPU, ~95M)
SPK_DIR = os.path.join(STT_DIR, "wavlm-base-plus-sv")
SPK_FILE = os.path.join(os.environ["IMX_WORK"], "voiceprint.npy")
SPK_THR = float(os.environ.get("IMX_SPK_THR", "0.83"))   # measured: same voice 0.92-0.97, other voice 0.77
_spk = {"fe": None, "m": None}
_spk_lock = threading.Lock()

def spk_embed(wav16k):
    with _spk_lock:
        if _spk["m"] is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            src = SPK_DIR if os.path.exists(os.path.join(SPK_DIR, "config.json")) else "microsoft/wavlm-base-plus-sv"
            _spk["fe"] = AutoFeatureExtractor.from_pretrained(src)
            _spk["m"] = WavLMForXVector.from_pretrained(src, dtype=torch.float32).eval()
        x = _spk["fe"](wav16k, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = _spk["m"](**x).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).numpy()

def read_wav16k(data):
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        idx = np.linspace(0, len(wav) - 1, int(len(wav) * 16000 / sr))
        wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
    return wav

def voice_match(wav16k):
    """None if no voiceprint is enrolled, else cosine similarity to the enrolled speaker."""
    if not os.path.exists(SPK_FILE) or len(wav16k) < 16000 * 0.6:
        return None
    ref = np.load(SPK_FILE)
    return float(np.dot(spk_embed(wav16k), ref / np.linalg.norm(ref)))

@app.post("/api/voiceprint")
async def voiceprint_enroll(audio: UploadFile = File(...), reset: bool = Form(False)):
    wav = read_wav16k(await audio.read())
    if len(wav) < 16000 * 1.5:
        return JSONResponse({"ok": False, "error": "speak for at least 2 seconds"}, 400)
    emb = await asyncio.get_running_loop().run_in_executor(None, spk_embed, wav)
    os.makedirs(os.path.dirname(SPK_FILE), exist_ok=True)
    if not reset and os.path.exists(SPK_FILE):          # several samples average into a steadier voiceprint
        old = np.load(SPK_FILE); emb = old + emb
    np.save(SPK_FILE, emb)
    return {"ok": True, "seconds": round(len(wav) / 16000, 1)}

@app.get("/api/voiceprint")
def voiceprint_status():
    return {"enrolled": os.path.exists(SPK_FILE), "threshold": SPK_THR}

@app.delete("/api/voiceprint")
def voiceprint_reset():
    if os.path.exists(SPK_FILE):
        os.remove(SPK_FILE)
    return {"enrolled": False}
