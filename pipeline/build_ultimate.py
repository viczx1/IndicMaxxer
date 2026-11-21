#!/usr/bin/env python3
"""IndicMaxxer ULTIMATE chat dataset -> ONE jsonl.

Pulls the best open chat/instruction data for Kannada, Hindi, English, Hinglish, Kanglish, cleans it, balances it,
adds our own Codex reasoning + hand-written conversations, and writes a single file:
    data/ultimate/indicmaxxer_ultimate.jsonl
Each line: {id, lang, source, task, mode, messages:[{role,content}], prompt, completion}
  - messages   : canonical chat (last message = assistant target; may carry "think")
  - prompt/completion : already rendered in IndicMaxxer chat format (drop-in for the SFT cell, loss on completion)

Run where `datasets` works (Kaggle / Colab cell):  !python pipeline/build_ultimate.py --hf-token $HF_TOKEN
Every source is optional: a failing source is skipped with a message, never fatal.
"""
import argparse, glob, hashlib, json, os, random, re, time, unicodedata
from collections import Counter, defaultdict

SYSTEM = {
    "kn": "ನೀನು IndicMaxxer: ವೃತ್ತಿಪರ, ಆತ್ಮೀಯ, ನೇರ ಮಾತು. ಸ್ವಲ್ಪ ಹಾಸ್ಯ, ಅತಿಯಾಗಿ ಅಲ್ಲ. ಬಳಕೆದಾರರು ಕನ್ನಡದಲ್ಲಿ ಬರೆದಿದ್ದಾರೆ: ಕನ್ನಡ ಲಿಪಿಯಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸು.",
    "hi": "तुम IndicMaxxer हो: पेशेवर, गर्मजोशी भरे, सटीक। हल्का सा मज़ाक, ज़्यादा नहीं। उपयोगकर्ता ने हिंदी में लिखा है: सिर्फ़ देवनागरी हिंदी में जवाब दो।",
    "en": "You are IndicMaxxer: professional, warm, crisp. Light dry humour, never overdone. "
          "The user wrote in English: reply in English. Tools: web_search, doc_read, doc_write.",
    "kanglish": "You are IndicMaxxer: professional, warm, crisp. Light dry humour, never overdone. "
                "The user wrote Kanglish (Kannada in English letters): reply in Kanglish only, "
                "not in Kannada script, not in Hindi. Tools: web_search, doc_read, doc_write.",
    "hinglish": "You are IndicMaxxer: professional, warm, crisp. Light dry humour, never overdone. "
                "The user wrote Hinglish (Hindi in English letters): reply in Hinglish only, "
                "not in Devanagari, not in Kannada. Tools: web_search, doc_read, doc_write.",
    "default": "You are IndicMaxxer: professional, warm, crisp. Light dry humour, never overdone. "
               "Reply in the user's language and script. Tools: web_search, doc_read, doc_write.",
}
# target share of the final file (kanglish is scarce online: filled by our data + romanised Kannada)
TARGET = {"kn": 0.30, "hi": 0.20, "en": 0.20, "hinglish": 0.15, "kanglish": 0.15}

# (hf repo, config, split, expected lang or None = detect, cap, task)
HF_SOURCES = [
    ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", "en", 3000, "conversation"),
    ("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", "en", 9000, "instruction"),
    ("CohereLabs/aya_dataset", None, "train", None, 30000, "instruction"),          # kn / hi / en rows by language
    ("rishiraj/samvaad-hi-v1", None, "train", None, 8000, "conversation"),
    ("ai4bharat/indic-align", "Anudesh", "train", None, 12000, "instruction"),
    ("ai4bharat/indic-align", "Indic_ShareLlama", "train", None, 12000, "instruction"),
    ("Tensoic/gooftagoo", None, "train", None, 8000, "conversation"),
    ("DSMJ910/hinglish-instruct-10k", None, "train", "hinglish", 8000, "instruction"),
]

BAD = re.compile(r"as an ai language model|i am an ai developed by|openai|chatgpt|मैं एक एआई भाषा मॉडल|"
                 r"llama|meta ai|i cannot browse|knowledge cutoff", re.I)
LANG_NAMES = {"kannada": "kn", "kan": "kn", "kn": "kn", "hindi": "hi", "hin": "hi", "hi": "hi",
              "english": "en", "eng": "en", "en": "en", "hinglish": "hinglish"}
HINGLISH_W = set("hai hain kya nahi nahin mein main kaise kyun bhai yaar toh aur bahut accha achha karo karna "
                 "kar raha rahi tha thi ho hoon batao chahiye kuch abhi sab kal aaj".split())
KANGLISH_W = set("illa ide idhe enu yenu hege heLi heli maadi maadu guru maga beku beda gottu nanna ninna "
                 "nimma aagutte aaytu swalpa thumba tumba olle chennagide yaaru elli yaavaga".split())

def nfc(s):
    return unicodedata.normalize("NFC", (s or "").replace("\r\n", "\n")).strip()

def style(text):
    """kn / hi / en / kanglish / hinglish from script + romanised markers."""
    kn = sum(0x0C80 <= ord(c) <= 0x0CFF for c in text)
    hi = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    lat = sum(c.isascii() and c.isalpha() for c in text)
    tot = kn + hi + lat or 1
    if kn / tot > 0.4:
        return "kn"
    if hi / tot > 0.4:
        return "hi"
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    k, h = len(words & KANGLISH_W), len(words & HINGLISH_W)
    if k >= 2 and k > h:
        return "kanglish"
    if h >= 2:
        return "hinglish"
    return "en"

# ------------------------------------------------------------------ schema-adaptive extraction -> list of messages
def to_messages(ex):
    for key in ("messages", "conversations", "conversation", "chat", "dialogue"):
        v = ex.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out = []
            for m in v:
                role = (m.get("role") or m.get("from") or m.get("speaker") or "").lower()
                text = m.get("content") or m.get("value") or m.get("text") or ""
                role = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}.get(role, role)
                if role in ("user", "assistant", "system"):
                    out.append({"role": role, "content": nfc(text)})
            return out
    v = ex.get("interactions")                         # indic-align style: [[user, assistant], ...]
    if isinstance(v, list) and v:
        out = []
        for pair in v:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                out += [{"role": "user", "content": nfc(pair[0])}, {"role": "assistant", "content": nfc(pair[1])}]
        return out
    for q, a in (("inputs", "targets"), ("instruction", "output"), ("prompt", "response"), ("question", "answer"),
                 ("input", "output"), ("prompt", "completion"), ("query", "response")):
        if ex.get(q) and ex.get(a):
            user = nfc(ex[q])
            if q == "instruction" and ex.get("input"):
                user += "\n\n" + nfc(ex["input"])
            return [{"role": "user", "content": user}, {"role": "assistant", "content": nfc(ex[a])}]
    return []

def clean_dialog(msgs, max_turns=6):
    msgs = [m for m in msgs if m["role"] != "system" and m["content"]]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    out = []
    for m in msgs:                                      # enforce strict user/assistant alternation
        want = "user" if len(out) % 2 == 0 else "assistant"
        if m["role"] != want:
            break
        out.append(m)
    if len(out) % 2:
        out = out[:-1]
    out = out[-2 * max_turns:]
    return out if len(out) >= 2 else []

def ok(msgs, lang):
    for m in msgs:
        c = m["content"]
        if len(c) > (1800 if m["role"] == "assistant" else 1500) or BAD.search(c):
            return False
        if m["role"] == "assistant" and len(c) < 2:
            return False
        if re.search(r"(.{6,}?)\1{4,}", c):              # degenerate repetition
            return False
    last_user, last_bot = msgs[-2]["content"], msgs[-1]["content"]
    su, sb = style(last_user), style(last_bot)
    if lang in ("kn", "hi"):
        return su == lang and sb == lang                # same script both sides
    if lang == "en":
        return su == "en" and sb == "en"
    return su == lang and sb == lang                    # romanised: reply must stay romanised, same style

# ------------------------------------------------------------------ rendering (IndicMaxxer chat format)
def render(msgs, lang, rng, mode=None, think=""):
    mode = mode or "no_think"
    flag = rng.choices([mode, "auto"], [0.93, 0.07])[0]
    parts = [f"<|im_start|>system\n{SYSTEM.get(lang, SYSTEM['default'])}<|im_end|>\n"] if rng.random() < 0.7 else []
    for i, m in enumerate(msgs[:-1]):
        c = m["content"]
        if i == len(msgs) - 2 and flag != "auto":
            c += " /think" if mode == "think" else " /no_think"
        parts.append(f"<|im_start|>{m['role']}\n{c}<|im_end|>\n")
    prompt = "".join(parts) + "<|im_start|>assistant\n"
    completion = f"<think>\n{think}\n</think>\n\n{msgs[-1]['content']}<|im_end|>\n" if think \
        else f"<think>\n\n</think>\n\n{msgs[-1]['content']}<|im_end|>\n"
    return prompt, completion

# ------------------------------------------------------------------ Kanglish from Kannada (casual romanisation)
def kanglish_of(text):
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return None
    r = transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
    r = re.sub(r"[\^~]", "", r).replace("RRi", "ru").replace("RRI", "ru")
    r = r.replace("aa", "aa").replace("A", "aa").replace("I", "ii").replace("U", "uu").replace("E", "e").replace("O", "o")
    r = re.sub(r"[.]{1}N", "n", r).replace(".a", "").replace("M", "m").replace("H", "h")
    return r.lower()

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--out", default="data/ultimate/indicmaxxer_ultimate.jsonl")
    ap.add_argument("--total", type=int, default=60000, help="approx rows in the final file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=2.5, help="time cap per source")
    ap.add_argument("--own", nargs="*", default=[], help="existing ultimate jsonl(s) to reuse as our own data")
    ap.add_argument("--target", default=None, help='JSON language shares, e.g. {"en":0.5,"kn":0.2,...}')
    args = ap.parse_args()
    if not args.hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            args.hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if args.target:
        TARGET.clear(); TARGET.update(json.loads(args.target))
    rng = random.Random(args.seed)
    pool = defaultdict(list)                            # lang -> [record]
    seen, report = set(), Counter()

    def add(msgs, lang, source, task, mode="no_think", think=""):
        key = hashlib.md5((lang + mode + msgs[-2]["content"][:300]).encode()).hexdigest()
        if key in seen:
            report["dup"] += 1
            return
        seen.add(key)
        pool[lang].append({"lang": lang, "source": source, "task": task, "mode": mode, "messages": msgs, "think": think})

    # 1) our own data (highest priority, always kept)
    own = 0
    for path in sorted(glob.glob("data/sft/*train*.jsonl")) + sorted(glob.glob("data/convo/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:                         # hand-written convos: one record per assistant turn
                ms = r["messages"]
                for i in range(1, len(ms), 2):
                    add([{"role": m["role"], "content": m["content"]} for m in ms[:i + 1]], r["lang"],
                        "indicmaxxer/convo", "conversation")
                    own += 1
            elif "prompt" in r and r.get("task") == "reasoning":   # codex reasoning (already rendered)
                user = r["prompt"].rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0]
                user = user.replace(" /think", "").replace(" /no_think", "")
                comp = r["completion"]
                think = comp.split("<think>\n", 1)[1].split("\n</think>")[0].strip() if "<think>" in comp else ""
                ans = comp.split("</think>\n\n", 1)[-1].replace("<|im_end|>\n", "").strip()
                add([{"role": "user", "content": user}, {"role": "assistant", "content": ans}], r["lang"],
                    "indicmaxxer/codex", "reasoning", r["mode"], think if r["mode"] == "think" else "")
                own += 1
    for path in args.own:                               # a previous ultimate file: reuse its own-data rows
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("source", "").startswith("indicmaxxer"):
                last = r["messages"][-1]
                ms = r["messages"][:-1] + [{"role": "assistant", "content": last["content"]}]
                add(ms, r["lang"], r["source"], r["task"], r["mode"], last.get("think", ""))
                own += 1
    print("own records:", own)

    # 2) Hugging Face sources
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
        print("`datasets` not installed: only our own data will be written")
    for repo, cfg, split, lang_fixed, cap, task in (HF_SOURCES if load_dataset else []):
        name = f"{repo}" + (f"/{cfg}" if cfg else "")
        kept = scanned = 0
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True, token=args.hf_token)
            t_src = time.time()
            for ex in ds:
                scanned += 1
                if kept >= cap or scanned > cap * 4 or time.time() - t_src > args.max_minutes * 60:
                    break
                if scanned % 5000 == 0:
                    print(f"  {name}: scanned {scanned}, kept {kept}", flush=True)
                lang_col = str(ex.get("language") or ex.get("lang") or ex.get("language_code") or "").lower()
                msgs = clean_dialog(to_messages(ex))
                if not msgs:
                    report[f"{name}:unparsed"] += 1
                    continue
                lang = lang_fixed or LANG_NAMES.get(lang_col) or style(msgs[-2]["content"])
                if lang not in TARGET:
                    continue
                if not ok(msgs, lang):
                    report[f"{name}:filtered"] += 1
                    continue
                add(msgs, lang, name, task)
                kept += 1
            print(f"{name}: kept {kept} / scanned {scanned}")
        except Exception as e:
            print(f"{name}: SKIPPED ({type(e).__name__}: {str(e)[:120]})")

    # 3) Kanglish top-up: romanise Kannada conversations (Kanglish is scarce online)
    need = int(args.total * TARGET["kanglish"]) - len(pool["kanglish"])
    if need > 0 and pool["kn"]:
        made = 0
        for r in rng.sample(pool["kn"], min(len(pool["kn"]), need * 2)):
            if made >= need or r["task"] == "reasoning":
                continue
            ms = [{"role": m["role"], "content": kanglish_of(m["content"])} for m in r["messages"]]
            if any(m["content"] is None or len(m["content"]) < 2 for m in ms):
                continue
            add(ms, "kanglish", r["source"] + "+romanised", r["task"])
            made += 1
        print("kanglish romanised from kannada:", made)

    # 4) balance to target shares (own data always kept first)
    final = []
    for lang, share in TARGET.items():
        recs = pool[lang]
        mine = [r for r in recs if r["source"].startswith("indicmaxxer")]
        rest = [r for r in recs if not r["source"].startswith("indicmaxxer")]
        rng.shuffle(rest)
        quota = max(int(args.total * share) - len(mine), 0)
        final += mine + rest[:quota]
    rng.shuffle(final)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stats = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(final):
            prompt, completion = render(r["messages"], r["lang"], rng, r["mode"], r["think"])
            msgs = r["messages"][:-1] + [dict(r["messages"][-1], **({"think": r["think"]} if r["think"] else {}))]
            f.write(json.dumps({"id": f"imx-{i:06d}", "lang": r["lang"], "source": r["source"], "task": r["task"],
                                "mode": r["mode"], "messages": msgs, "prompt": prompt, "completion": completion},
                               ensure_ascii=False) + "\n")
            stats[(r["lang"], r["source"])] += 1
    by_lang = Counter(r["lang"] for r in final)
    print(f"\nwrote {len(final)} rows -> {args.out}")
    print("by language:", dict(by_lang))
    print("by source:")
    for (lang, src), n in sorted(stats.items()):
        print(f"  {lang:9s} {src:45s} {n}")
    print("dropped:", dict(report.most_common(12)))

if __name__ == "__main__":
    main()
