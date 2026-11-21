#!/usr/bin/env python3
"""Clean, label and deduplicate Codex-generated distillation data (standard library only).

Input : data/distill/*.jsonl  (one JSON object per line; a file holding one JSON array also works)
        Accepted record shapes:
          A) {"user": "...", "think": "...", "answer": "...", "lang": "...", "task": "...", "mode": "..."}
          B) {"messages": [{"role": "user", "content": "..."},
                           {"role": "assistant", "content": "...", "think": "..."}, ...],
              "lang": "...", "task": "..."}
Output: data/clean/clean.jsonl    kept records (canonical, labelled)
        data/clean/rejects.jsonl  dropped records + reason
        data/clean/report.json    counts per language / task / mode / reject reason

Usage:  python -X utf8 pipeline/clean_distill.py [--in "data/distill/*.jsonl"] [--out data/clean]
"""
import argparse, glob, hashlib, json, os, random, re, unicodedata
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------- text normalisation
KN_RANGE, HI_RANGE = (0x0C80, 0x0CFF), (0x0900, 0x097F)
ZW_KEEP = {"‌", "‍"}                      # ZWNJ / ZWJ matter in Indic scripts
DIGIT_TABLE = str.maketrans({**{chr(0x0CE6 + i): str(i) for i in range(10)},   # Kannada digits
                             **{chr(0x0966 + i): str(i) for i in range(10)}})  # Devanagari digits

def norm_text(s):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch in "\n\t" or ch in ZW_KEEP:
            out.append(ch)
        elif cat in ("Cc", "Cf", "Co", "Cs"):      # control / invisible format chars
            continue
        else:
            out.append(ch)
    s = "".join(out).replace("\t", " ")
    s = re.sub(r"[  ]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"^\s*(example|ex|sample)\s*#?\d+\s*[:.)-]\s*", "", s, flags=re.I)  # "Example 12:" leftovers
    return s.strip()

# ----------------------------------------------------------------------------- language / script labelling
KANGLISH = set("""ide idhe illa illla maadi madi maadu madu hegide hegiddira hegidiya nanu naanu nange nanage nimma
nimge nimage beku beda aagide agide aaytu aytu eshtu yeshtu yaake yake houdu howdu hoon swalpa banni hogi kodi
heli helu helti yenu enu yen guru sari ella barutte baratte bartini maga namma nam avru avaru alli illi idu adu
yavaga yelli elli hege nodi nodu bandu hogbeku kelsa kelasa duddu ondu eradu""".split())
HINGLISH = set("""hai hain nahi nahin kya kar karo karna karke mein main mera meri mere aap tum hoga hogi tha thi
the bhai kyun kyu kaise kab acha accha achha theek thik raha rahe rahi gaya gayi jayega jaayega chahiye bata batao
bhi abhi yaar kuch sab woh wo yeh ye apna apni hum humein mujhe tumhe paisa paise kaam kal aaj""".split())
CANON = {"kn": "kn", "kannada": "kn", "hi": "hi", "hindi": "hi", "en": "en", "english": "en",
         "kanglish": "kanglish", "hinglish": "hinglish"}
ROMAN = {"en", "kanglish", "hinglish"}

def script_counts(s):
    kn = hi = lat = 0
    for ch in s:
        o = ord(ch)
        if KN_RANGE[0] <= o <= KN_RANGE[1]:
            kn += 1
        elif HI_RANGE[0] <= o <= HI_RANGE[1]:
            hi += 1
        elif ch.isascii() and ch.isalpha():
            lat += 1
    return kn, hi, lat

def detect_style(s):
    """kn | hi | en | kanglish | hinglish | mixed | unknown"""
    kn, hi, lat = script_counts(s)
    tot = kn + hi + lat
    if tot == 0:
        return "unknown"
    if kn / tot >= 0.5:
        return "kn"
    if hi / tot >= 0.5:
        return "hi"
    if lat / tot >= 0.5:
        words = re.findall(r"[a-z]+", s.lower())
        k = sum(w in KANGLISH for w in words)
        h = sum(w in HINGLISH for w in words)
        need = 1 if len(words) < 12 else max(2, round(0.04 * len(words)))
        if max(k, h) >= need and k != h:
            return "kanglish" if k > h else "hinglish"
        return "en"
    return "mixed"

def claimed_style(raw):
    raw = (raw or "").lower()
    for key in ("kanglish", "hinglish", "kannada", "hindi", "english"):
        if key in raw:
            return CANON[key]
    return CANON.get(raw.strip(), "unknown")

def styles_compatible(user_style, ans_style):
    if user_style == ans_style:
        return True
    # romanized detection is noisy on short answers: accept en <-> kanglish/hinglish, flag later
    return user_style in ROMAN and ans_style in ROMAN

# ----------------------------------------------------------------------------- task labelling
TASK_RULES = [("grounded_qa", r"faq|document|context|policy|passage-based|given text"),
              ("conversation", r"dialog|conversation|customer|chat|support"),
              ("reasoning", r"reason|math|word problem|logic|arithmetic|aptitude"),
              ("grammar", r"grammar|vibhakti|tense|case|morpholog"),
              ("comprehension", r"comprehension|reading|summar"),
              ("nlu", r"intent|extract|slot|entity|classif|sentiment")]

def canon_task(raw):
    raw = (raw or "").lower()
    for name, pat in TASK_RULES:
        if re.search(pat, raw):
            return name
    return "general"

# ----------------------------------------------------------------------------- quality checks
JUNK = re.compile(r"as an ai (language )?model|here are \d+ (training )?examples|i cannot (generate|help)|"
                  r"```|\bjsonl?\b\s*:|lorem ipsum", re.I)
EXPR = re.compile(r"(\d+(?:\.\d+)?(?:\s*[-+×x*/÷]\s*\d+(?:\.\d+)?)+)\s*=\s*(\d+(?:\.\d+)?)")

def check_math(text):
    """Return (#equations checked, #wrong). Only plain 'a op b ... = c' chains are checked."""
    t = text.translate(DIGIT_TABLE)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)        # 1,000 -> 1000 (but keep list commas)
    checked = wrong = 0
    for m in EXPR.finditer(t):
        expr = m.group(1).replace("×", "*").replace("x", "*").replace("÷", "/")
        try:
            val = eval(expr, {"__builtins__": {}}, {})   # regex guarantees digits/operators only
        except Exception:
            continue
        checked += 1
        if abs(val - float(m.group(2))) > max(0.051, 0.005 * abs(val)):
            wrong += 1
    return checked, wrong

def degenerate(s):
    words = s.split()
    if len(words) < 30:
        return False
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(grams)) / len(grams) < 0.3

# ----------------------------------------------------------------------------- loading
def iter_raw(paths):
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        stripped = text.strip()
        if stripped.startswith("["):                      # whole file is one JSON array
            try:
                for i, obj in enumerate(json.loads(stripped)):
                    yield path, i, obj, None
                continue
            except json.JSONDecodeError:
                pass
        for i, line in enumerate(text.splitlines()):
            line = line.strip().rstrip(",")
            if not line or line.startswith("```"):
                continue
            try:
                yield path, i, json.loads(line), None
            except json.JSONDecodeError as e:
                yield path, i, line[:300], f"bad_json: {e.msg}"

def to_messages(obj):
    """Normalise both record shapes to [{'role','content', ('think')}], user/assistant alternating."""
    if isinstance(obj.get("messages"), list):
        msgs = []
        for m in obj["messages"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).lower()
            role = {"human": "user", "customer": "user", "bot": "assistant", "agent": "assistant",
                    "model": "assistant"}.get(role, role)
            if role not in ("user", "assistant", "system"):
                continue
            msgs.append({"role": role, "content": norm_text(m.get("content")), "think": norm_text(m.get("think"))})
        msgs = [m for m in msgs if m["role"] != "system"]
    else:
        msgs = [{"role": "user", "content": norm_text(obj.get("user") or obj.get("question") or obj.get("input")),
                 "think": ""},
                {"role": "assistant", "content": norm_text(obj.get("answer") or obj.get("output") or obj.get("response")),
                 "think": norm_text(obj.get("think") or obj.get("reasoning") or obj.get("explanation"))}]
    return msgs

# ----------------------------------------------------------------------------- near-duplicate detection
MASK = (1 << 61) - 1
_rng = random.Random(0)
PERMS = [(_rng.randrange(1, MASK), _rng.randrange(0, MASK)) for _ in range(16)]

def shingles(s, k=5):
    s = re.sub(r"\s+", " ", s.lower()).strip()[:600]
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}

def minhash(sh):
    hs = [int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") for x in sh]
    return [min((a * h + b) & MASK for h in hs) for a, b in PERMS]

def skeleton(s):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", s.lower())).strip()

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/distill/*.jsonl")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--near-dup", type=float, default=0.8, help="Jaccard threshold on user text")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inp))
    if not paths:
        raise SystemExit(f"no input files match {args.inp}")
    os.makedirs(args.out, exist_ok=True)

    kept, rejects = [], []
    reasons = Counter()

    def reject(reason, src, payload):
        reasons[reason.split(":")[0]] += 1
        rejects.append({"reason": reason, "source": src, "record": payload})

    seen_exact = set()
    for path, idx, obj, err in iter_raw(paths):
        src = f"{os.path.basename(path)}:{idx}"
        if err:
            reject(err, src, obj); continue
        if not isinstance(obj, dict):
            reject("not_object", src, obj); continue

        msgs = to_messages(obj)
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            reject("bad_turn_structure", src, obj); continue
        if any(msgs[i]["role"] == msgs[i + 1]["role"] for i in range(len(msgs) - 1)):
            reject("turns_not_alternating", src, obj); continue
        if any(not m["content"] for m in msgs):
            reject("empty_turn", src, obj); continue

        last_user, final = msgs[-2], msgs[-1]
        think = final.get("think", "")
        all_text = "\n".join(m["content"] + "\n" + m.get("think", "") for m in msgs)

        # length limits
        if not (2 <= len(last_user["content"]) <= 6000):
            reject("user_length", src, obj); continue
        if not (1 <= len(final["content"]) <= 3000) or len(think) > 5000:
            reject("answer_or_think_length", src, obj); continue
        if JUNK.search(all_text):
            reject("junk_or_meta_text", src, obj); continue
        if degenerate(final["content"]) or degenerate(think):
            reject("repetitive", src, obj); continue

        # language / script fidelity (judge user by its tail: FAQ docs come first, question last)
        u_style = detect_style(last_user["content"][-400:])
        a_style = detect_style(final["content"])
        if u_style in ("unknown", "mixed"):
            reject(f"user_style_{u_style}", src, obj); continue
        if not styles_compatible(u_style, a_style):
            reject(f"script_mismatch:{u_style}->{a_style}", src, obj); continue
        if think:
            t_style = detect_style(think)
            if t_style not in ("unknown",) and not styles_compatible(u_style, t_style):
                reject(f"think_script_mismatch:{u_style}->{t_style}", src, obj); continue

        # math verification
        checked, wrong = check_math(think + "\n" + final["content"])
        if wrong:
            reject(f"math_wrong:{wrong}/{checked}", src, obj); continue

        # exact + skeleton dedupe (numbers/whitespace normalised)
        key = hashlib.sha1((skeleton(last_user["content"]) + "\x00" + skeleton(final["content"])).encode()).hexdigest()
        if key in seen_exact:
            reject("duplicate_exact", src, obj); continue
        seen_exact.add(key)

        mode_raw = str(obj.get("mode", "")).lower().replace("-", "_")
        mode = "think" if think else "no_think"
        steps = len([l for l in think.splitlines() if l.strip()])
        claimed = claimed_style(obj.get("lang"))
        kept.append({
            "id": hashlib.sha1(src.encode()).hexdigest()[:12],
            "source": src,
            "lang": u_style,
            "lang_claimed": claimed,
            "lang_agrees": claimed in ("unknown", u_style),
            "romanized_answer_is_en": u_style in ("kanglish", "hinglish") and a_style == "en",
            "task": canon_task(obj.get("task")),
            "mode": mode,
            "mode_relabelled": bool(mode_raw) and mode_raw != mode,
            "turns": sum(m["role"] == "user" for m in msgs),
            "think_steps": steps,
            "difficulty": "none" if steps == 0 else ("easy" if steps <= 2 else ("medium" if steps <= 5 else "hard")),
            "math_checked": checked,
            "messages": msgs,
        })

    # near-duplicates on the final user turn, within each language (MinHash LSH, 4 bands x 4 rows)
    buckets = defaultdict(list)
    drop = set()
    sigs, shs = {}, {}
    for i, r in enumerate(kept):
        sh = shingles(r["messages"][-2]["content"])
        shs[i], sigs[i] = sh, minhash(sh)
        for b in range(4):
            buckets[(r["lang"], b, tuple(sigs[i][b * 4:(b + 1) * 4]))].append(i)
    for members in buckets.values():
        for x in range(len(members)):
            i = members[x]
            if i in drop:
                continue
            for j in members[x + 1:]:
                if j in drop:
                    continue
                inter = len(shs[i] & shs[j])
                if inter / (len(shs[i] | shs[j]) or 1) >= args.near_dup:
                    drop.add(j)
    for j in sorted(drop):
        reject("duplicate_near", kept[j]["source"], kept[j]["messages"])
    kept = [r for i, r in enumerate(kept) if i not in drop]

    with open(os.path.join(args.out, "clean.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "rejects.jsonl"), "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(kept) + len(rejects)
    report = {
        "input_files": len(paths), "records_seen": total, "kept": len(kept), "rejected": len(rejects),
        "keep_rate": round(len(kept) / total, 3) if total else 0,
        "reject_reasons": dict(reasons.most_common()),
        "by_lang": dict(Counter(r["lang"] for r in kept).most_common()),
        "by_task": dict(Counter(r["task"] for r in kept).most_common()),
        "by_mode": dict(Counter(r["mode"] for r in kept).most_common()),
        "by_difficulty": dict(Counter(r["difficulty"] for r in kept).most_common()),
        "lang_label_disagreements": sum(not r["lang_agrees"] for r in kept),
        "romanized_question_english_answer": sum(r["romanized_answer_is_en"] for r in kept),
        "math_equations_verified": sum(r["math_checked"] for r in kept),
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
