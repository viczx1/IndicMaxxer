#!/usr/bin/env python3
"""Turn cleaned records into IndicMaxxer chat-format SFT data (standard library only).

Input : data/clean/clean.jsonl   (from clean_distill.py; template-generated reasoning can be appended
                                  in the same canonical format)
Output: data/sft/train.jsonl, data/sft/val.jsonl, data/sft/stats.json
        Each line: {"id","lang","task","mode","flag","prompt","completion","text"}
          prompt     = everything up to and including "<|im_start|>assistant\n"  (no loss)
          completion = "<think>...</think>\n\n<answer><|im_end|>\n"              (loss here)
          text       = prompt + completion

Mode flags (Qwen3-style, one model for both behaviours):
  think    -> last user turn ends with " /think",    completion has the reasoning inside <think>
  no_think -> last user turn ends with " /no_think", completion has an EMPTY <think>\n\n</think>
  ~5% of examples get no flag at all ("auto"): the model learns to choose.

Usage:  python -X utf8 pipeline/build_sft.py [--in data/clean/clean.jsonl] [--out data/sft]
"""
import argparse, hashlib, json, os, random
from collections import Counter

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
TARGET_LANG_SHARE = {"kn": 0.45, "hi": 0.25, "en": 0.10, "kanglish": 0.10, "hinglish": 0.10}

def turn(role, content):
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"

def render(rec, rng, p_system, p_auto):
    msgs = rec["messages"]
    final = msgs[-1]
    flag = "auto" if rng.random() < p_auto else rec["mode"]

    parts = []
    if rng.random() < p_system:
        parts.append(turn("system", SYSTEM.get(rec["lang"], SYSTEM["default"])))
    for i, m in enumerate(msgs[:-1]):
        content = m["content"]
        if i == len(msgs) - 2 and flag != "auto":          # flag goes on the last user turn
            content += " /think" if flag == "think" else " /no_think"
        # earlier assistant turns are shown without their reasoning (as Qwen3 does in history)
        parts.append(turn(m["role"], content))
    prompt = "".join(parts) + "<|im_start|>assistant\n"
    think = final.get("think", "") if rec["mode"] == "think" else ""
    completion = f"<think>\n{think}\n</think>\n\n{final['content']}<|im_end|>\n" if think \
        else f"<think>\n\n</think>\n\n{final['content']}<|im_end|>\n"
    return flag, prompt, completion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean/clean.jsonl")
    ap.add_argument("--extra", nargs="*", default=[], help="more canonical jsonl files (e.g. template reasoning)")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--p-system", type=float, default=0.7)
    ap.add_argument("--p-auto", type=float, default=0.05)
    ap.add_argument("--p-no-think", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = []
    for path in [args.inp, *args.extra]:
        with open(path, encoding="utf-8") as f:
            recs.extend(json.loads(l) for l in f if l.strip())
    if not recs:
        raise SystemExit("no records")

    os.makedirs(args.out, exist_ok=True)
    out = {"train": open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8"),
           "val": open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8")}
    counts = {"train": Counter(), "val": Counter()}
    flags = Counter()
    # also teach direct answers: a copy of some think items in no_think mode (answer only)
    extra = [dict(r, id=r["id"] + "#nt", mode="no_think") for r in recs
             if r["mode"] == "think" and rng.random() < args.p_no_think]
    recs = recs + extra
    for rec in recs:
        flag, prompt, completion = render(rec, rng, args.p_system, args.p_auto)
        # deterministic split by id so re-runs never leak val into train
        h = int(hashlib.sha1(rec["id"].split("#")[0].encode()).hexdigest(), 16) % 10_000
        split = "val" if h < args.val_frac * 10_000 else "train"
        row = {"id": rec["id"], "lang": rec["lang"], "task": rec["task"], "mode": rec["mode"], "flag": flag,
               "prompt": prompt, "completion": completion, "text": prompt + completion}
        out[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split][rec["lang"]] += 1
        flags[flag] += 1
    for f in out.values():
        f.close()

    n = sum(counts["train"].values()) + sum(counts["val"].values())
    lang_total = counts["train"] + counts["val"]
    share = {k: round(v / n, 3) for k, v in lang_total.most_common()}
    gaps = {k: f"{share.get(k, 0):.2f} vs target {t:.2f}" for k, t in TARGET_LANG_SHARE.items()
            if abs(share.get(k, 0) - t) > 0.05}
    stats = {"records": n, "train": sum(counts["train"].values()), "val": sum(counts["val"].values()),
             "lang_share": share, "flags": dict(flags.most_common()),
             "tasks": dict(Counter(r["task"] for r in recs).most_common()),
             "lang_mix_warnings": gaps or "none"}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if gaps:
        print("\nLanguage mix is off target: generate more Codex batches for the under-filled languages.")

if __name__ == "__main__":
    main()
