#!/usr/bin/env python3
"""Turn data/convo/*.jsonl multi-turn chats into SFT rows (one row per assistant turn, earlier turns as history).
Output: data/sft/convo_train.jsonl  ({id, lang, task, mode, flag, prompt, completion}); stdlib only.
Usage: python -X utf8 pipeline/build_convo.py"""
import glob, json, os, random

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

def turn(role, content):
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"

def main():
    rng, out, n_conv = random.Random(0), [], 0
    for path in sorted(glob.glob("data/convo/*.jsonl")):
        for ln, line in enumerate(open(path, encoding="utf-8")):
            if not line.strip():
                continue
            c = json.loads(line)
            msgs, lang = c["messages"], c["lang"]
            assert all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(msgs)), (path, ln)
            n_conv += 1
            for i in range(1, len(msgs), 2):
                flag = rng.choices(["no_think", "think", "auto"], [0.8, 0.12, 0.08])[0]
                parts = [turn("system", SYSTEM.get(lang, SYSTEM["default"]))] if rng.random() < 0.7 else []
                for j, m in enumerate(msgs[:i]):
                    content = m["content"]
                    if j == i - 1 and flag != "auto":
                        content += " /think" if flag == "think" else " /no_think"
                    parts.append(turn(m["role"], content))
                prompt = "".join(parts) + "<|im_start|>assistant\n"
                completion = f"<think>\n\n</think>\n\n{msgs[i]['content']}<|im_end|>\n"
                out.append({"id": f"{os.path.basename(path)}:{ln}:{i}", "lang": lang, "task": "conversation",
                            "mode": "no_think", "flag": flag, "prompt": prompt, "completion": completion})
    os.makedirs("data/sft", exist_ok=True)
    with open("data/sft/convo_train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in out:
        by[r["lang"]] = by.get(r["lang"], 0) + 1
    print(f"{n_conv} conversations -> {len(out)} training rows", by)

if __name__ == "__main__":
    main()
