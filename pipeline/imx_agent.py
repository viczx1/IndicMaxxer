#!/usr/bin/env python3
"""IndicMaxxer-2B agent runtime: style lock + rule-based tool router + native multi-pass reasoning.

  /instruct  -> router decides tools from the message + attachments (no model call needed to pick them):
                OCR (Tesseract kan+hin+eng) | doc_read (pdf/docx/txt, long docs are chunked + retrieved)
                | web_search (Wikipedia kn/hi/en, optional DuckDuckGo) | doc_write (answer saved as .docx)
  /reasoning -> same tools, then k sampled native-language thoughts in ONE batched generate, then a vote
                (majority on the final number, else the answer that agrees most with the others).
Every reply returns a trace: style, mode, tools, tokens (IndicMaxxer vs Sarvam-1 vs cl100k), latency.
"""
import collections, json, os, re, time, urllib.parse, urllib.request

import regex
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ.get("IMX_WORK", "/kaggle/working")
OUT_DIR = os.path.join(WORK, "agent_outputs")
UA = {"User-Agent": "IndicMaxxer-demo/0.1 (research prototype)"}

SYS = {
    "kn": "ನೀನು IndicMaxxer, Sarvam-1 ಆಧಾರಿತ ಭಾರತೀಯ ಭಾಷಾ ಸಹಾಯಕ: ವೃತ್ತಿಪರ, ಆತ್ಮೀಯ, ನೇರ, ಸ್ವಲ್ಪ ಹಾಸ್ಯ. ಕನ್ನಡ ಲಿಪಿಯಲ್ಲೇ ಉತ್ತರಿಸು.",
    "hi": "तुम IndicMaxxer हो, Sarvam-1 पर बना भारतीय भाषा सहायक: पेशेवर, गर्मजोशी भरे, सटीक, हल्का मज़ाक। देवनागरी हिंदी में ही जवाब दो।",
    "en": "You are IndicMaxxer, an Indian-language assistant built on Sarvam-1: professional, warm, crisp, light dry humour. Reply in English.",
    "kanglish": "You are IndicMaxxer, an Indian-language assistant built on Sarvam-1. The user writes Kanglish (Kannada in English letters): reply in Kanglish only.",
    "hinglish": "You are IndicMaxxer, an Indian-language assistant built on Sarvam-1. The user writes Hinglish (Hindi in English letters): reply in Hinglish only.",
}
STYLE_NAME = {"kn": "Kannada script", "hi": "Devanagari Hindi", "en": "English",
              "kanglish": "Kanglish (Kannada in English letters)", "hinglish": "Hinglish (Hindi in English letters)"}
CTX_INTRO = {"kn": "ಕೆಳಗಿನ ಮಾಹಿತಿಯನ್ನು ಬಳಸಿ ಉತ್ತರಿಸು:", "hi": "नीचे दी गई जानकारी से जवाब दो:",
             "en": "Use this information to answer:", "kanglish": "Ee information use maadi answer maadi:",
             "hinglish": "Is information ka use karke jawab do:"}
CTX_ASK = {"kn": "ಮೇಲಿನ ಮಾಹಿತಿಯಿಂದ ಮಾತ್ರ ಚಿಕ್ಕದಾಗಿ ಉತ್ತರಿಸು. ಹೆಸರು, ಸಂಖ್ಯೆಗಳನ್ನು ಇದ್ದಂತೆಯೇ ಬರೆ. ಪ್ರಶ್ನೆ:",
           "hi": "सिर्फ़ ऊपर दी गई जानकारी से छोटा जवाब दो। नाम और संख्याएँ जैसी हैं वैसी ही लिखो। सवाल:",
           "en": "Answer in 1-3 short sentences using only the information above. Copy names and numbers exactly. Question:",
           "kanglish": "Mele iro information inda maatra chikkadaagi answer maadi. Hesaru, numbers iddange bareyiri. Question:",
           "hinglish": "Sirf upar ki information se chhota jawab do. Naam aur numbers jaise hain waise likho. Question:"}
BRIEF = {"kn": " ಚಿಕ್ಕದಾಗಿ, ಸ್ಪಷ್ಟವಾಗಿ ಉತ್ತರಿಸು.", "hi": " छोटा और साफ़ जवाब दो।", "en": " Keep replies short and clear.",
         "kanglish": " Chikkadaagi clear aagi answer maadi.", "hinglish": " Chhota aur clear jawab do."}
CTX_SUM = {"kn": "ಮೇಲಿನ ದಾಖಲೆಯ ಸಾರಾಂಶವನ್ನು ಕನ್ನಡದಲ್ಲಿ 4-6 ಬಿಂದುಗಳಲ್ಲಿ ಬರೆ. ಹೆಸರುಗಳು, ದಿನಾಂಕಗಳು ಮತ್ತು ಸಂಖ್ಯೆಗಳನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಇರುವಂತೆಯೇ ಬರೆ.",
           "hi": "ऊपर दिए दस्तावेज़ का सारांश हिंदी में 4-6 बिंदुओं में लिखो। नाम, तारीखें और संख्याएँ दस्तावेज़ जैसी ही लिखो।",
           "en": "Summarise the document above in English in 4-6 bullet points. Copy names, dates and numbers exactly as they appear in the document. Do not add anything that is not in it.",
           "kanglish": "Mele iro document na Kanglish alli 4-6 points alli summary maadi. Hesaru, dates, numbers document alli iro haage bareyiri.",
           "hinglish": "Upar wale document ka Hinglish mein 4-6 points mein summary do. Naam, dates aur numbers document jaise hi likho."}
SEARCH_VERB = re.compile(r"\b(search( for)?|google|look\s*up|find|hudku\w*|huduki\w*|khoj\w*)\b|ಹುಡುಕ\S*|खोज\S*|ढूंढ\S*", re.I)

# ------------------------------------------------------------------ style detection (kn / hi / en / Kanglish / Hinglish)
KANNADA, DEVANAGARI = regex.compile(r"\p{Kannada}"), regex.compile(r"\p{Devanagari}")
KANGLISH = set("""naanu neenu nanna ninna avanu avalu yaaru yenu enu enri illa ide ide alla beku beda maadi maadu
maadtini maadtiya maadbeku hogu hogi banni baa bartini gottu gottilla swalpa tumba chennagi chennagide yaake yelli elli
hege heg aaytu aagide aagilla kodi kodu helu heli sari guru maga namaskara oota nodi nodu idiya idini nimma namma
haudu houdu ivattu naale ninne iga alli illi aa ee jothe saaku bega dodda chikka yeshtu eshtu""".split())
HINGLISH = set("""hai hain ho hoon hu tha thi kya kyun kaise kaisa kaun kab kahan kitna nahi nahin mein mujhe tujhe tum
aap hum main mera meri tera tumhara aapka yaar bhai accha acha theek thik karo karna kar raha rahi kuch bahut abhi aaj
kal bata batao chahiye sakte sakta wala wali matlab samajh dekho chalo haan bilkul zyada kam jaldi phir""".split())

def detect_style(text):
    kn, hi = len(KANNADA.findall(text)), len(DEVANAGARI.findall(text))
    if kn or hi:
        return "kn" if kn >= hi else "hi"
    words = re.findall(r"[a-z]+", text.lower())
    k, h = sum(w in KANGLISH for w in words), sum(w in HINGLISH for w in words)
    if max(k, h) == 0 or max(k, h) / max(len(words), 1) < 0.12:
        return "en"
    return "kanglish" if k > h else "hinglish"

def style_ok(style, text):
    letters = regex.findall(r"\p{L}", text)
    if len(letters) < 8:
        return True
    kn = sum(bool(KANNADA.match(c)) for c in letters) / len(letters)
    hi = sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters)
    return {"kn": kn > 0.4, "hi": hi > 0.4}.get(style, kn + hi < 0.1)

# ------------------------------------------------------------------ rule-based router
SEARCH_RE = re.compile(
    r"\b(search|google|look\s*up|latest|news|today|current|recent|price of|who won|weather|wiki\w*|who (is|was)|"
    r"when (is|was|did)|where is|capital of|population of|founded|hudku\w*|ivattu|suddi|khoj\w*|aaj|taaza|khabar)\b"
    r"|ಹುಡುಕ|ಇತ್ತೀಚ|ಸುದ್ದಿ|ಇಂದು|ಈಗಿನ|ಯಾರು|ಯಾವಾಗ|ಎಲ್ಲಿದೆ|ರಾಜಧಾನಿ|खोज|ताज़ा|ताजा|खबर|आज|कौन (है|था|थे)|कब|राजधानी",
    re.I)
SELF_RE = re.compile(r"\b(you|your|my|mine|neenu|ninna|nanna|tum|tumhara|aap|aapka|mera|meri|mujhe)\b"
                     r"|ನೀನು|ನಿನ್ನ|ನೀವು|ನನ್ನ|तुम|आप|मेरा|मेरी|मुझे", re.I)   # personal / about-me: no web search
WRITE_RE = re.compile(
    r"\b(write|draft|create|make|save|prepare)\b.{0,40}\b(doc|docx|document|file|letter|report|notice|email|mail)\b"
    r"|\b(as|to) (a )?(docx|word|doc|file)\b|\b(bareda|baredu|bari|likh\w*)\b.{0,30}\b(doc|file|letter|patra)\b"
    r"|ಪತ್ರ ಬರೆ|ಡಾಕ್ಯುಮೆಂಟ್|ಫೈಲ್|ದಾಖಲೆ ಬರೆ|पत्र लिख|दस्तावेज़|डॉक्यूमेंट|फ़ाइल|फाइल", re.I)
SUMMARY_RE = re.compile(r"summar|saaramsha|saransh|ಸಾರಾಂಶ|सारांश|overview|gist|tl;?dr", re.I)
IMG_EXT, DOC_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}, {".pdf", ".docx", ".txt", ".md"}
FLAG_RE = re.compile(r"\s*/(reasoning|instruct)\s*$", re.I)

def route(text, files=(), force_search=False):
    tools = []
    for f in files or []:
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXT:
            tools.append(("ocr", f))
        elif ext in DOC_EXT:
            tools.append(("doc_read", f))
    if force_search or (SEARCH_RE.search(text) and not SELF_RE.search(text) and not files):
        tools.append(("web_search", text))
    if WRITE_RE.search(text):
        tools.append(("doc_write", None))
    return tools

# ------------------------------------------------------------------ tools (plain Python)
_READERS = {}

def _easy_lines(res):
    """EasyOCR boxes -> text lines in reading order (group by vertical position, then left to right)."""
    res = sorted(res, key=lambda r: sum(p[1] for p in r[0]) / 4)
    lines, cur, cy = [], [], None
    for box, txt, _ in res:
        y = sum(p[1] for p in box) / 4
        h = abs(box[2][1] - box[0][1]) or 10
        if cy is None or abs(y - cy) < h * 0.6:
            cur.append((box[0][0], txt)); cy = y if cy is None else (cy + y) / 2
        else:
            lines.append(" ".join(t for _, t in sorted(cur))); cur, cy = [(box[0][0], txt)], y
    if cur:
        lines.append(" ".join(t for _, t in sorted(cur)))
    return "\n".join(lines)

def ocr(path, style="en"):
    """EasyOCR (pure PyTorch, no Tesseract needed). Kannada and Devanagari need separate readers: try the one that
    matches the user's language first, keep the other's result if it is more confident. Tesseract is the fallback."""
    try:
        import easyocr, torch
        order = ["kn", "hi"] if style in ("kn", "kanglish") else ["hi", "kn"]
        best = None
        for script in order:
            if script not in _READERS:
                _READERS[script] = easyocr.Reader([script, "en"], gpu=torch.cuda.is_available(), verbose=False)
            res = _READERS[script].readtext(path, detail=1)
            conf = sum(r[2] for r in res) / max(len(res), 1)
            if best is None or conf > best[0]:
                best = (conf, res)
            if conf >= 0.55 and res:
                break
        return _easy_lines(best[1]).strip()
    except ImportError:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path), lang="kan+hin+eng").strip()

def doc_read(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        if len(text.strip()) < 50:                         # scanned PDF: OCR the first pages
            try:
                from pdf2image import convert_from_path
                import pytesseract
                text = "\n".join(pytesseract.image_to_string(im, lang="kan+hin+eng")
                                 for im in convert_from_path(path, dpi=200, first_page=1, last_page=6))
            except Exception as e:
                text += f"\n[scanned PDF, OCR failed: {e}]"
        return text.strip()
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs).strip()
    return open(path, encoding="utf-8", errors="ignore").read().strip()

def _get(url, timeout=8):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

STOP = re.compile(r"\b(please|pls|search|google|look\s*up|tell me|about|the|latest|news|today|current|what|who|is|was|"
                  r"are|of|in|on|for|a|an|hudku\w*|khoj\w*|batao|bata|helu|heli|enu|kya|hai)\b|[?!.,]"
                  r"|ಹುಡುಕ\S*|ಬಗ್ಗೆ|ಹೇಳ\S*|ಏನು|खोज\S*|बता\S*|के बारे में|क्या|है", re.I)

NEWS_RE = re.compile(r"\b(latest|news|today|current|recent|price|weather|who won|ivattu|aaj|taaza|khabar)\b"
                     r"|ಇತ್ತೀಚ|ಸುದ್ದಿ|ಇಂದು|ಈಗಿನ|ताज़ा|ताजा|खबर|आज", re.I)

def _wiki(q, lang, n):
    api = f"https://{lang}.wikipedia.org/w/api.php"
    for attempt in range(2):                               # one polite retry on HTTP 429
        try:
            hits = _get(f"{api}?action=query&list=search&format=json&srlimit={n}&srsearch={urllib.parse.quote(q)}")
            ids = [str(h["pageid"]) for h in hits["query"]["search"]]
            if not ids:
                return []
            pages = _get(f"{api}?action=query&prop=extracts&exintro=1&explaintext=1&format=json&pageids={'|'.join(ids)}")
            return [(p["title"], p.get("extract", "")[:700], f"https://{lang}.wikipedia.org/?curid={p['pageid']}")
                    for p in pages["query"]["pages"].values() if p.get("extract")]
        except Exception:
            time.sleep(1.5)
    return []

def _ddg(q, n, news=False):
    try:
        from ddgs import DDGS
        if news:
            return [(f"{r['title']} ({r.get('date', '')[:10]})", r["body"][:500], r["url"])
                    for r in DDGS().news(q, max_results=n)]
        return [(r["title"], r["body"][:500], r["href"]) for r in DDGS().text(q, max_results=n)]
    except Exception:
        return []

def web_search(text, style, n=3):
    """News-like -> DuckDuckGo first. Else Wikipedia in the user's language -> DuckDuckGo -> English Wikipedia."""
    q = re.sub(r"\s+", " ", STOP.sub(" ", FLAG_RE.sub("", text))).strip() or text
    native = {"kn": "kn", "hi": "hi"}.get(style)
    order = [lambda: _ddg(q, n, news=True)] if NEWS_RE.search(text) else []
    if native:
        order.append(lambda: _wiki(q, native, n))
    order += [lambda: _ddg(q, n), lambda: _wiki(q, "en", n)]
    for fn in order:
        hits = fn()
        if hits:
            return q, hits
    return q, []

def doc_write(title, body):
    import docx
    os.makedirs(OUT_DIR, exist_ok=True)
    d = docx.Document()
    d.add_heading(title[:80] or "IndicMaxxer", level=1)
    for para in body.split("\n"):
        if para.strip():
            d.add_paragraph(para.strip())
    safe = re.sub(r"[^\w]+", "_", title[:40], flags=re.U).strip("_") or "document"
    path = os.path.join(OUT_DIR, f"{safe}_{int(time.time())}.docx")
    d.save(path)
    return path

WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
def pick_chunks(text, question, count_tokens, budget):
    """Context tax: only the chunks that matter go into the 8K window (keyword overlap), in document order."""
    words = WORD.findall(text)
    chunks = [" ".join(words[i:i + 120]) for i in range(0, len(words), 120)]
    if count_tokens(text) <= budget:
        return text, len(chunks), len(chunks)
    if SUMMARY_RE.search(question):
        order = list(range(len(chunks)))
    else:
        q = {w.lower() for w in WORD.findall(question) if len(w) > 2}
        order = sorted(range(len(chunks)), key=lambda i: -len(q & {w.lower() for w in WORD.findall(chunks[i])}))
    keep, used = [], 0
    for i in order:
        t = count_tokens(chunks[i])
        if used + t > budget:
            break
        keep.append(i); used += t
    return "\n...\n".join(chunks[i] for i in sorted(keep)), len(keep), len(chunks)

# ------------------------------------------------------------------ answer selection for /reasoning
def last_num(s):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None

def vote(answers):
    nums = [last_num(a) for a in answers]
    c = collections.Counter(n for n in nums if n)
    if c and c.most_common(1)[0][1] >= 2:
        best = c.most_common(1)[0][0]
        return nums.index(best), f"majority on {best} ({c[best]}/{len(answers)})"
    sets = [set(w.lower() for w in WORD.findall(a)) for a in answers]
    score = [sum(len(s & t) / max(len(s | t), 1) for j, t in enumerate(sets) if j != i) for i, s in enumerate(sets)]
    i = max(range(len(answers)), key=lambda j: score[j])
    return i, "consensus (most agreement with the other passes)"

# ------------------------------------------------------------------ the agent
class Agent:
    def __init__(self, path, device="cuda:0", hf_token=None, load_4bit=False):
        """load_4bit=True: NF4 weights (~1.8 GB VRAM) so the 2B model fits a 4 GB laptop GPU."""
        self.dev = device
        self.tk = AutoTokenizer.from_pretrained(path)
        kw = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=torch.float16),
                  "device_map": {"": 0}}
        try:
            self.lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, **kw)
        except TypeError:
            self.lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, **kw)
        if not getattr(self.lm, "is_loaded_in_4bit", False):
            self.lm.to(device)
        self.lm.eval()
        self.eos = [self.tk.convert_tokens_to_ids("<|im_end|>"), self.tk.eos_token_id]
        self.ref = {}                                      # tokenizers to show the token tax live
        try:
            self.ref["Sarvam-1"] = AutoTokenizer.from_pretrained("sarvamai/sarvam-1", token=hf_token)
        except Exception:
            pass
        try:
            import tiktoken
            self.ref["GPT-4 cl100k"] = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    def ntok(self, text):
        return len(self.tk.encode(text, add_special_tokens=False))

    def tax(self, text):
        row = {"IndicMaxxer-2B": self.ntok(text)}
        for name, t in self.ref.items():
            row[name] = len(t.encode(text)) if name.startswith("GPT") else len(t.encode(text, add_special_tokens=False))
        return row

    @torch.no_grad()
    def generate(self, msgs, max_new=384, temp=0.0, k=1):
        text = self.tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = self.tk(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -7000:].to(self.dev)
        sample = temp > 0 or k > 1
        out = self.lm.generate(ids, max_new_tokens=max_new, do_sample=sample, temperature=max(temp, 0.7) if sample else None,
                               top_p=0.9 if sample else None, num_return_sequences=k, repetition_penalty=1.0, no_repeat_ngram_size=6,   # a penalty would punish Indic suffixes that legitimately repeat
                               eos_token_id=self.eos, pad_token_id=self.tk.pad_token_id)
        res = []
        for row in out:
            gen = row[ids.shape[1]:]
            n = int((~torch.isin(gen, torch.tensor(self.eos + [self.tk.pad_token_id], device=gen.device))).sum())
            t = self.tk.decode(gen, skip_special_tokens=False)
            for sp in ["<|im_end|>", self.tk.eos_token or "", self.tk.pad_token or ""]:
                if sp:
                    t = t.replace(sp, "")
            think, ans = t.rsplit("</think>", 1) if "</think>" in t else ("", t)   # last </think> wins
            res.append((think.replace("<think>", "").replace("</think>", "\n").strip(), ans.strip(), n))
        return res, ids.shape[1]

    def chat(self, message, history=(), mode=None, k=3, force_search=False, files=()):
        t0 = time.time()
        text, style, mode, msgs, trace = self.prepare(message, history, mode, force_search, files)
        return self.finish(text, style, mode, msgs, trace, k, t0)

    compressor = None     # optional callable(text, rate) -> shorter text (LLMLingua-2), set by the server

    def _squeeze(self, text, rate, trace):
        """Compress context/history with the token compressor and keep score of the saving."""
        if not (self.compressor and trace.get("compress_on")) or self.ntok(text) < 60:
            return text
        if trace.get("style") in ("kn", "hi"):     # Indic scripts are already dense: keep more words (fewer lost facts)
            rate = min(0.75, rate + 0.1)
        try:
            short = self.compressor(text, rate)
        except Exception as e:
            trace["tools"].append(f"compressor failed: {e!r}"[:120])
            return text
        c = trace.setdefault("compress", {"before": 0, "after": 0, "gpt_removed": 0})
        c["before"] += self.ntok(text); c["after"] += self.ntok(short)
        g = self.ref.get("GPT-4 cl100k")
        if g is not None:                       # what the same text would have cost GPT-4 had it been sent uncompressed
            c["gpt_removed"] += len(g.encode(text)) - len(g.encode(short))
        return short

    def prepare(self, message, history=(), mode=None, force_search=False, files=(), compress=True, docs=()):
        """Style detection + rule-based tool calls + prompt. Returns (text, style, mode, msgs, trace)."""
        m = FLAG_RE.search(message)
        mode = (m.group(1).lower() if m else None) or mode or "instruct"
        text = FLAG_RE.sub("", message).strip()
        style = detect_style(text) if text else "en"
        trace = {"style": style, "mode": mode, "tools": [], "sources": [], "compress_on": bool(compress and self.compressor)}
        ctx = [f"[document]\n{d}" for d in docs if d and d.strip()]   # documents read earlier in this chat

        for tool, arg in route(text, files, force_search):          # ---- rule-based tool calls
            if tool == "web_search" and docs and not force_search:     # a question about the open document
                continue
            t1 = time.time()
            if tool in ("ocr", "doc_read"):
                try:
                    raw = ocr(arg, style) if tool == "ocr" else doc_read(arg)
                    if not raw.strip():
                        raise ValueError("no text found")
                except Exception as e:                        # never let the model guess what an unread file says
                    trace["tools"].append(f"{tool}({os.path.basename(arg)}) failed: {e}")
                    trace["file_error"] = os.path.basename(arg)
                    continue
                # with the compressor on, twice as much of the document fits in the same context window
                kept, nk, nall = pick_chunks(raw, text, self.ntok, budget=5000 if trace["compress_on"] else 2500)
                # short documents/forms go in verbatim (compressing labels like "blood group" breaks them);
                # only long ones are squeezed so more of them fits
                ctx.append(f"[{os.path.basename(arg)}]\n{self._squeeze(kept, 0.5, trace) if self.ntok(kept) > 800 else kept}")
                trace["file_ok"] = True
                trace.setdefault("docs", []).append(kept)              # sent back so follow-up turns keep the document
                trace["tools"].append(f"{tool}({os.path.basename(arg)}): {len(raw)} chars, kept {nk}/{nall} chunks"
                                      f", {time.time() - t1:.1f}s")
                trace["doc_tax"] = self.tax(raw)
                if not text:
                    text = {"kn": "ಈ ದಾಖಲೆಯ ಸಾರಾಂಶ ಕೊಡು.", "hi": "इस दस्तावेज़ का सारांश दो।"}.get(style, "Summarise this document.")
            elif tool == "web_search":
                q, hits = web_search(text, style)
                ctx += [f"[{t}] {self._squeeze(s, 0.4, trace)}" for t, s, _ in hits]
                trace["sources"] += [(t, u) for t, _, u in hits]
                trace["tools"].append(f"web_search('{q}'): {len(hits)} results, {time.time() - t1:.1f}s")

        msgs = [{"role": "system", "content": SYS[style] + BRIEF[style]}]
        # token diet: only the last 2 exchanges; older turns squeezed hard; the last reply lightly; your last message verbatim
        hist = [h for h in list(history)[-4:] if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)]
        for j, h in enumerate(hist):
            c = h["content"].split("\n\n---\n")[0][:1200]
            if j < len(hist) - 2:
                c = self._squeeze(c, 0.35, trace)
            elif h["role"] == "assistant":
                c = self._squeeze(c, 0.55, trace)
            msgs.append({"role": h["role"], "content": c})
        if ctx:   # the model was not trained on RAG: drop "search for" verbs so it answers instead of telling you to google
            q = re.sub(r"\s+", " ", SEARCH_VERB.sub(" ", text)).strip() or text
            ask = CTX_SUM[style] if (SUMMARY_RE.search(q) or (files and len(q.split()) < 4)) else f"{CTX_ASK[style]} {q}"
            user = f"{CTX_INTRO[style]}\n\n" + "\n\n".join(ctx) + f"\n\n{ask}"
        else:
            user = text
        msgs.append({"role": "user", "content": f"{user} /{mode}"})
        if trace.get("compress"):
            c = trace["compress"]
            trace["tools"].append(f"compressor: {c['before']} -> {c['after']} tokens "
                                  f"(-{100 * (c['before'] - c['after']) / max(c['before'], 1):.0f}%)")
        return text, style, mode, msgs, trace

    def finish(self, text, style, mode, msgs, trace, k, t0):
        out_file = None
        if mode == "reasoning":                                       # ---- k native thoughts, one batched call
            runs, n_in = self.generate(msgs, max_new=512, temp=0.7, k=max(1, k))
            i, how = vote([a for _, a, _ in runs]) if len(runs) > 1 else (0, "single pass")
            think, ans = runs[i][0], runs[i][1]
            trace["passes"] = [(th, a) for th, a, _ in runs]
            trace["vote"] = how
            n_out = sum(n for _, _, n in runs)
        else:
            runs, n_in = self.generate(msgs, max_new=384)
            think, ans, n_out = runs[0]
        if not style_ok(style, ans):                                  # ---- style lock: one retry
            msgs[0]["content"] += f" Reply ONLY in {STYLE_NAME[style]}."
            _, a2, n2 = self.generate(msgs, max_new=384, temp=0.3)[0][0]
            if style_ok(style, a2):
                ans, n_out = a2, n_out + n2
                trace["tools"].append("style_lock: regenerated once")

        if any(t == "doc_write" for t, _ in route(text, (), False)):
            out_file = doc_write(text, ans)
            trace["tools"].append(f"doc_write -> {os.path.basename(out_file)}")
        dt = time.time() - t0
        trace.update(think=think, prompt_tokens=n_in, new_tokens=n_out, seconds=round(dt, 1),
                     tok_per_s=round(n_out / max(dt, 1e-6), 1), tax=self.tax(text + "\n" + ans))
        return ans, trace, out_file

def trace_md(tr):
    lines = [f"**style** `{tr['style']}` · **mode** `/{tr['mode']}` · **{tr['seconds']} s** · "
             f"{tr['new_tokens']} new tokens ({tr['tok_per_s']} tok/s) · prompt {tr['prompt_tokens']} tokens"]
    lines.append("**tools:** " + ("; ".join(tr["tools"]) if tr["tools"] else "none (model only)"))
    if tr.get("vote"):
        lines.append(f"**/reasoning:** {len(tr['passes'])} native passes, picked by {tr['vote']}")
    tax = tr["tax"]
    base = tax["IndicMaxxer-2B"]
    lines.append("**token tax (this turn):** " + " · ".join(
        f"{k} {v}" + ("" if k == "IndicMaxxer-2B" else f" ({v / max(base, 1):.1f}x)") for k, v in tax.items()))
    if tr.get("doc_tax"):
        lines.append("**document tokens:** " + " · ".join(f"{k} {v}" for k, v in tr["doc_tax"].items()))
    if tr["sources"]:
        lines.append("**sources:** " + " · ".join(f"[{t}]({u})" for t, u in tr["sources"]))
    if tr.get("think"):
        lines.append(f"<details><summary>thinking</summary>\n\n{tr['think'][:3000]}\n\n</details>")
    if tr.get("passes") and len(tr["passes"]) > 1:
        for j, (th, a) in enumerate(tr["passes"]):
            lines.append(f"<details><summary>pass {j + 1}: {a[:80]}</summary>\n\n{th[:1500]}\n\n**→** {a[:600]}\n\n</details>")
    return "\n\n".join(lines)
