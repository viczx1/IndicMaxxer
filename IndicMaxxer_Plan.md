# IndicMaxxer: 12-Hour Hackathon Plan
### Trained from scratch · instruct + reasoning in one model · Kannada, Hindi, English, Kanglish, Hinglish

**Challenge:** The Big Hack 2.0, AI Research R3, "The token tax". Reduce the cost or quality gap for Indian languages in language models, and measure exactly what the improvement costs.

**Our answer:** **IndicMaxxer**, a small multilingual model **trained from scratch**, with **Kannada first**, plus Hindi and English. It also handles the way people actually type: **Kanglish** and **Hinglish** (Kannada or Hindi written in English letters, mixed with English words). It uses **our own tokenizer** and is **one model with two modes**:
- **Instruct mode** (`/no_think`): fast, direct answers.
- **Reasoning mode** (`/think`): works step by step **in the user's own language** inside `<think>…</think>`, then answers.

It runs on a 4 GB laptop GPU, with no API cost and no data leaving the device.

> This file is the single source of truth. `DatasetCollectionReport.md` and `architecture_designReport.md` are reference material only.

---

## 1. Vision

1. **Our own Indic tokenizer.** Kannada and Hindi should cost about as many tokens as English.
2. **Native reasoning.** It reasons in Kannada or Hindi, not through English internally.
3. **The languages people actually use.** Native script, romanized text, and code-mixing like "naale meeting ide, reschedule maadi" or "kal meeting hai, reschedule kar do".
4. **Instruct and reasoning in one model,** switched per request.
5. **Small, local and enterprise-ready.** Privacy by default, no per-token bill, voice (speech-to-text and text-to-speech) plugs in and out, more text fits in the context window.

---

## 2. Facts we will lean on

### 2.1 We tested this ourselves: GPT-4 and Qwen tokenizers break Indic words apart

Before BPE runs, tokenizers split text into chunks with a regular expression ("regex"), and BPE never merges across chunk boundaries. The GPT-4 (cl100k) and Qwen2-style regexes treat **vowel signs and the virama** as punctuation, because Unicode classes them as "marks" (Mn/Mc), not letters. Every Kannada word, and every Hindi word with a vowel sign (ि, ा, ्), is cut apart before BPE starts. Llama 3 uses the same pattern as cl100k.

Sentence: `ಕನ್ನಡ ನಮ್ಮ ಭಾಷೆ. ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿಯಾಗಿದೆ.`

| Pre-tokenizer | Chunks | Result |
|---|---|---|
| cl100k (GPT-4), Qwen2-style | **22** | `ಕನ` `್ನಡ` ` ನಮ` `್ಮ` ` ಭ` `ಾಷ` `ೆ.` … |
| o200k (GPT-4o) | **8** | `ಕನ್ನಡ` ` ನಮ್ಮ` ` ಭಾಷೆ` `.` … |

Hindi: `भारत की राजधानी दिल्ली है। हिंदी हमारी भाषा है।` becomes **26 chunks** under cl100k (`भ` `ारत` ` क` `ी` …) vs **11** under o200k (`भारत` ` की` ` राजधानी` …).

o200k fixed this by adding `\p{M}` (Unicode marks) to its letter classes; ours does too. *(Both tested with the published regex patterns using Python's `regex` module. Confirm Qwen3's exact pattern in its `tokenizer.json`.)*

### 2.2 How many tokens a word costs today (Kannada, from published reports)

| Tokenizer | Kannada tokens/word | Source |
|---|---|---|
| mGPT | 16.39 | Dravidian study (arXiv 2608.07727) |
| Llama-3.1 | 14.95 | Sarvam-1 model card |
| Gemma-2 | 5.55 | Sarvam-1 model card |
| GPT-4o | 3.29 | Sarvam-1 model card |
| Sarvam-1 | 2.37 | Sarvam-1 model card |
| PARAM-1 | 2.17 | PARAM-1 paper |
| Dedicated 32K Kannada BPE | 1.59 | Dravidian study |

The challenge brief puts the tax at Kannada 12×, Tamil 9.9×, Hindi 4.1× (cl100k vs English).

**Targets:** Kannada ≤ 1.8 tokens/word, Hindi ≤ 1.5, English ≤ 1.5, with all five language styles sharing one 48K vocabulary. We measure the Hindi baselines ourselves (§6.1).

### 2.3 Models do lean on English internally
- Schut et al. 2025 ([arXiv 2502.15603](https://arxiv.org/abs/2502.15603)): multilingual models make key decisions in a space closest to English.
- Zhong et al. 2024 ([arXiv 2408.10811](https://arxiv.org/abs/2408.10811)): models trained heavily on Japanese develop a Japanese internal language as well.
- We show it with a **logit-lens** comparison (decoding each layer back into words): Qwen3-0.6B vs IndicMaxxer, on Kannada and Hindi prompts.

### 2.4 Who else is in this space, and where our opening is
Kannada: Ambari 7B, Kan-LLaMA 7B. Multilingual Indic: Sarvam-1 2B (base model), PARAM-1 2.9B, Krutrim. **Our opening:** a **sub-150M model trained from scratch** on its own Indic tokenizer, with **romanized and code-mixed support** and **instruct and native reasoning in one model**, that runs on a laptop.

### 2.5 What we borrow
- **Qwen3 (April 2025):** one hybrid model with `/think` and `/no_think`. Qwen later split it into separate Instruct and Thinking models (July 2025, "2507"). We stay with one model and measure what the hybrid costs. Architecture building blocks: GQA, QK-Norm, RoPE, SwiGLU, RMSNorm, tied embeddings.
- **Karpathy's nanochat:** the staged recipe tokenizer → pretraining → mid-training → SFT → optional RL in a few hours. *(Its repo couldn't be fetched from here: if you fork it, check its license and change its split pattern.)*

---

## 3. Compute: three Google accounts plus a laptop

### 3.1 What we have

| Account | Colab allowance | Notes |
|---|---|---|
| **Pro A** (you) | ~200 compute units/month, A100/L4 priority | Google AI Pro bundles Colab benefits since Sept 2026 |
| **Pro B** (friend) | ~200 units/month | Same |
| **Plus C** (you) | ~50 units/month | Smaller allowance |
| **Laptop** | RTX 3050 4 GB, Ryzen 7 5800H, 15 GB RAM | Runs the demo; backup training |

Check that the units are active in **Colab → Settings → Subscription** on each account (trials don't count). Each person runs **their own** account.

**Important:** Colab gives each session **one GPU**, and runs can't be combined across accounts. **Compute units aren't our limit; the 12-hour clock is.** One A100 for about 5.5 hours of pretraining caps the main model at around **135M parameters**. The extra accounts buy **parallel experiments**, and those experiments are what answer "measure what it costs."

### 3.2 Who runs what

| Account | GPU | Job | Units |
|---|---|---|---|
| **Pro A** | A100 (or L4) | **Main run: IndicMaxxer-135M**, multilingual: pretraining → mid-training → hybrid SFT | ~100 |
| **Pro B** | L4 / A100 | **Ablations at XS size (~38M), each ~1–1.5 h:** ① **Kannada-only vs multilingual** (the cost of adding Hindi and English to Kannada quality), ② **our regex vs GPT-4's regex** (the tokenizer's effect on its own), ③ 32K vs 48K vocabulary (if time allows) | ~60–100 |
| **Plus C** | CPU, then T4/L4 | **Hours 0–2:** download and tokenize all data into the shared Drive. **Later:** evaluation runs, Gemini judge scoring, backup | ~30–50 |
| **Laptop** | RTX 3050 | Tokenizer benchmark, demo app, XS backup model | — |

**Shared data:** one Drive folder (`IndicMaxxer/`) shared with the team. Each account adds a shortcut in My Drive so Colab can mount it. Tokenized `uint16` shards are written to Drive once and copied to Colab's local `/content` disk at the start of each session.

**Colab rules:** use a **resumable** training script. Save the model, optimizer, step and data position to Drive every 20–30 minutes, and resume automatically. Keep tabs open and laptops awake.

**Gemini credits (~$10/month on each Pro, via the Google Developer Program; the API needs a $5 prepayment):** use them for grammar-checking templates, translating evaluation prompts, judge scoring, and Gemini CLI for writing code. **Don't use Gemini outputs as training data:** Google's terms restrict using outputs to build competing models.

### 3.3 Laptop gotchas
- **Set `HF_HOME=D:\hf_cache`**, since C: has only 12 GB free.
- Windows Application Control blocked an unsigned Python extension DLL in this session. **Test PyTorch with CUDA in a virtual environment on D: early.** Fall back to WSL2 or Colab if it's blocked.
- No `torch.compile` or flash-attn on native Windows; plain PyTorch SDPA with bf16 is fine. Keep it plugged in, in Performance mode.

### 3.4 Model configurations (Qwen3-style, vocabulary 49,152 = "48K", tied embeddings)

| Config | d_model | Layers | Q / KV heads (head size 64) | FFN | Context | Params | Where |
|---|---|---|---|---|---|---|---|
| Ablation-Kn | 256 | 8 | 4 / 2 | 704 | 1024 | ~14M (32K vocabulary) | Pro B, tokenizer ablation |
| **XS** | 384 | 12 | 6 / 2 | 1024 | 1024 | **~38M** | Pro B ablations · laptop backup |
| S– | 512 | 16 | 8 / 2 | 1408 | 1024 | ~70M | Main run if only an L4 is available |
| **S** | 576 | 30 | 9 / 3 | 1536 | 2048 | **~135M** | **Main run on an A100** |
| L | 1024 | 28 | 16 / 4 | 3072 | 2048 | ~390M | Only on 8× H100 (future work) |

At 135M parameters, an A100 processes about 150K tokens/s, so 5.5 hours ≈ **3B tokens**, about the compute-optimal amount (~20 tokens per parameter).

```python
from transformers import Qwen3Config, Qwen3ForCausalLM
cfg = Qwen3Config(                                   # config S
    vocab_size=49152, hidden_size=576, intermediate_size=1536, num_hidden_layers=30,
    num_attention_heads=9, num_key_value_heads=3, head_dim=64,
    max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
    tie_word_embeddings=True, attention_bias=False,
    bos_token_id=None, eos_token_id=IM_END_ID, pad_token_id=PAD_ID)
model = Qwen3ForCausalLM(cfg)                        # random init = trained from scratch
```

---

## 4. Tokenizer: one 48K vocabulary for five language styles

### 4.1 Specification
- **Type:** byte-level BPE, **49,152** entries. Still fits in `uint16`. The extra vocabulary covers Devanagari and romanized words without starving Kannada.
- **Normalization:** NFC only. **Never strip diacritics** (they're the vowel signs). Keep ZWJ/ZWNJ (zero-width joiner and non-joiner characters).
- **Splitting regex:** Qwen-style with `\p{M}` grouped with letters. Digits split one at a time.
- **Training sample, by characters (~400 MB):** Kannada 40% · Hindi 25% · English 15% · Kanglish 8% · Hinglish 7% · code/numbers 5%.
- **Special tokens:** `<|endoftext|>` `<|im_start|>` `<|im_end|>` `<think>` `</think>` `<|pad|>` plus 10 `<|reserved_i|>` (tool calls, voice tags, language tags later).
- **Ship gate:** Kannada fertility must stay ≤ 1.8. If Hindi pushes it higher, raise Kannada's share of the tokenizer sample. Kannada is first.

### 4.2 Training script

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers, Regex

PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}"
       r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

tok = Tokenizer(models.BPE())
tok.normalizer = normalizers.NFC()
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(PAT), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|pad|>"] \
           + [f"<|reserved_{i}|>" for i in range(10)]
trainer = trainers.BpeTrainer(vocab_size=49152, min_frequency=2, special_tokens=specials,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train(["data/tok_train.txt"], trainer)
tok.save("tokenizer/indicmaxxer-48k.json")
```

For the ablation, train two **Kannada-only 32K** tokenizers on the same text: one with `PAT`, and one with the GPT-4 regex (no `\p{M}`).

---

## 5. Data and training

### 5.1 Sources

| Style | Pretraining sources | Notes |
|---|---|---|
| **Kannada** | `HuggingFaceFW/fineweb-2` (`kan_Knda`, ~13.9B bytes), `ai4bharat/sangraha` (kn), `ai4bharat/IndicCorpV2` (kn), Kannada Wikipedia | ~1–2B unique tokens: the limiting resource |
| **Hindi** | `HuggingFaceFW/fineweb-2` (`hin_Deva`), `ai4bharat/sangraha` (hi), `ai4bharat/IndicCorpV2` (hi), Hindi Wikipedia | Plenty; take only what the mix needs |
| **English** | `HuggingFaceFW/fineweb-edu` sample | Knowledge plus technical terms |
| **Hinglish** | Search HF for L3Cube **HingCorpus** (romanized Hindi–English, tweets) and other Hinglish corpora | Real code-mixed text |
| **Kanglish** | Real data is scarce: use **DravidianCodeMix (Kannada)** comments for evaluation/SFT flavor; **for volume, romanize Kannada sentences word by word using `ai4bharat/Aksharantar` transliteration pairs** (spellings people actually type), and randomly swap ~20% of words for English equivalents | Synthetic, so say so in the pitch |
| **Reasoning text** | Our template generator (§5.3) in Kannada, Hindi and romanized versions | Correct by construction |

**Cleaning:** NFC; a script-ratio filter per language (≥ 60% of letters in the target script for native-script data); exact deduplication by hash; drop documents under 200 characters. **Keep** punctuation, Latin letters and newlines.

### 5.2 Mixes

| Stage | Kannada | Hindi | English | Kanglish | Hinglish | Reasoning (templates) |
|---|---|---|---|---|---|---|
| **Pretraining** (~3B tokens) | 42% | 23% | 15% | 6% | 6% | 8% |
| **Mid-training** (decay phase, last ~15%) | high-quality Wikipedia slices + **chat-formatted** instructions + reasoning traces, in all five styles | | | | | |
| **Hybrid SFT** | ~60% instruct `/no_think` + ~40% reasoning `/think`; language split about 45 / 25 / 10 / 10 / 10 | | | | | |

**SFT instruct sources:** `Cognitive-Lab/Kannada-Instruct-dataset`, `CohereForAI/aya_dataset` (kn, hi), `ai4bharat/indic-align` (kn, hi), plus Hinglish/Kanglish versions made by romanizing a slice with Aksharantar.

**Rule: reply in the user's language and script.** A Kannada question gets a Kannada answer, a Kanglish question gets a Kanglish answer. Do the same inside `<think>`. That is what "native reasoning" means for every language style.

Pre-tokenize into `uint16` shards per style, so the mix can be adjusted while training runs.

### 5.3 Reasoning data generator (correct by construction, native in every language)

```python
import random
KN_NAMES = ["ರವಿ", "ಅನಿಲ್", "ಸುರೇಶ್", "ಮಂಜು", "ಕಿರಣ್"]
HI_NAMES = ["राहुल", "अमित", "सुरेश", "विकास", "रोहन"]

def shopping(lang):
    a = random.randint(100, 900); b = random.randint(10, a // 3); c = random.randint(10, a // 3)
    if lang == "kn":
        n = random.choice(KN_NAMES)
        q = (f"{n} ಬಳಿ {a} ರೂಪಾಯಿ ಇತ್ತು. ಅವನು {b} ರೂಪಾಯಿಗೆ ಪುಸ್ತಕ ಮತ್ತು {c} ರೂಪಾಯಿಗೆ "
             f"ಪೆನ್ನು ಖರೀದಿಸಿದನು. ಈಗ ಅವನ ಬಳಿ ಎಷ್ಟು ರೂಪಾಯಿ ಉಳಿದಿದೆ?")
        t = f"ಒಟ್ಟು ಖರ್ಚು = {b} + {c} = {b+c} ರೂಪಾಯಿ.\nಉಳಿದ ಹಣ = {a} - {b+c} = {a-b-c} ರೂಪಾಯಿ."
        ans = f"{n} ಬಳಿ {a-b-c} ರೂಪಾಯಿ ಉಳಿದಿದೆ."
    else:  # "hi"
        n = random.choice(HI_NAMES)
        q = f"{n} के पास {a} रुपये थे। उसने {b} रुपये की किताब और {c} रुपये का पेन खरीदा। अब उसके पास कितने रुपये बचे?"
        t = f"कुल खर्च = {b} + {c} = {b+c} रुपये।\nबचे हुए पैसे = {a} - {b+c} = {a-b-c} रुपये।"
        ans = f"{n} के पास {a-b-c} रुपये बचे हैं।"
    return q, t, ans, a - b - c
# Romanized versions: pass q/t/ans through the Aksharantar romanizer (+ English word swaps).
# Templates to add: speed-distance, percentage/discount, profit-loss, ratio, ages, time-and-work,
# averages, day-of-week, ordering logic; plus multi-step chains of 2–4 templates.
```

- **Native speakers check every template** (one Kannada speaker, one Hindi speaker; 10 minutes each).
- Generate 40–60K examples from 15–25 templates. **Hold out 3–5 templates** entirely for evaluation.

### 5.4 Chat format (one model, two modes)

```
<|im_start|>system
ನೀನು IndicMaxxer. ಬಳಕೆದಾರರ ಭಾಷೆಯಲ್ಲೇ ಉತ್ತರಿಸು.<|im_end|>
<|im_start|>user
ರವಿ ಬಳಿ 250 ರೂಪಾಯಿ ಇತ್ತು ... ಎಷ್ಟು ಉಳಿದಿದೆ? /think<|im_end|>
<|im_start|>assistant
<think>
ಒಟ್ಟು ಖರ್ಚು = 80 + 45 = 125 ರೂಪಾಯಿ.
ಉಳಿದ ಹಣ = 250 - 125 = 125 ರೂಪಾಯಿ.
</think>
ರವಿ ಬಳಿ 125 ರೂಪಾಯಿ ಉಳಿದಿದೆ.<|im_end|>
```

- `/no_think` (the default) produces an empty `<think>\n\n</think>` block, so the output format is identical in both modes (Qwen3's trick).
- Loss only on assistant tokens. About 5% of examples have no flag, so the model learns a small "auto" mode.

### 5.5 Hyperparameters
- bf16 autocast (fp16 on a T4), SDPA attention, `torch.compile` on Colab only, sequences packed.
- AdamW (β = 0.9, 0.95), weight decay 0.1, gradient clipping at 1.0. Peak learning rate 3e-3 (XS/S–/S); QK-Norm keeps it stable.
- **WSD schedule (warmup–stable–decay):** 2% warmup, a flat stage, then decay to 0 over the last 15% (= mid-training). You can start the decay early if the clock runs short.
- Batch ~0.5M tokens per step (gradient accumulation). Checkpoint to Drive every 20–30 minutes.
- Hybrid SFT: learning rate 1e-4, 2–3 epochs.
- **Stretch:** one round of rejection-sampling fine-tuning on the template maths. Sample 4 answers, keep only the correct ones, fine-tune once.

---

## 6. Evaluation: "measure exactly what it costs"

| # | Metric | How | What it shows |
|---|---|---|---|
| 1 | **Tokens per word in all 5 styles** | Ours vs cl100k, o200k, Qwen3, Gemma, Llama, Sarvam | The size of the token-tax cut, per language |
| 2 | **Cost of multilinguality** (Pro B) | XS Kannada-only vs XS multilingual, equal compute: Kannada bits per byte | **What adding Hindi, English and romanized text costs Kannada** |
| 3 | **Tokenizer ablation** (Pro B) | Our regex vs the GPT-4 regex, same data and compute | The root-cause fix, measured |
| 4 | Bits per byte per language | Held-out sets (FLORES-200 `kan_Knda` / `hin_Deva` / `eng_Latn`, or held-out slices) | Quality measure that doesn't depend on the tokenizer |
| 5 | **Reasoning: `/think` vs `/no_think`** | Held-out templates (kn, hi, romanized) + ~50 hand-written problems per language | **Cost of thinking:** accuracy gained vs extra tokens and seconds |
| 6 | Instruct quality | 30 prompts per language, rated 1–5 by team members blind to the model + Gemini as judge, vs Qwen3-0.6B | Honest quality gap |
| 7 | Script fidelity | Share of answers in the same language and script as the question | Kanglish/Hinglish handling |
| 8 | Logit lens | Middle-layer decodings, Qwen3-0.6B vs ours, on Kannada/Hindi prompts | Native reasoning |
| 9 | Speed / size | Words per second on the laptop GPU and CPU, RAM, file size | On-device story |
| 10 | Effective context | Pages per 2K/4K tokens, ours vs other tokenizers | Long-context story |

**Be honest in the pitch:** a 135M model trained in one day knows less than Qwen3-0.6B. Frame it as cost vs quality.

### 6.1 Tokenizer benchmark

```python
import tiktoken            # run in Colab (its DLL is blocked on this Windows laptop)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def counters():
    c = {n: (lambda e: lambda s: len(e.encode(s)))(tiktoken.get_encoding(n))
         for n in ["cl100k_base", "o200k_base"]}
    for name in ["Qwen/Qwen3-0.6B", "google/gemma-3-1b-it", "meta-llama/Llama-3.2-1B", "sarvamai/sarvam-1"]:
        t = AutoTokenizer.from_pretrained(name)           # gemma/llama are gated: accept on HF first
        c[name] = (lambda t: lambda s: len(t(s, add_special_tokens=False).input_ids))(t)
    ours = PreTrainedTokenizerFast(tokenizer_file="tokenizer/indicmaxxer-48k.json")
    c["IndicMaxxer-48k"] = lambda s: len(ours(s, add_special_tokens=False).input_ids)
    return c

fert = lambda sents, f: sum(map(f, sents)) / sum(len(s.split()) for s in sents)
# sets = {"kn": [...], "hi": [...], "en": [...], "kanglish": [...], "hinglish": [...]}
for n, f in counters().items():
    print(f"{n:24s}", "  ".join(f"{k} {fert(v, f):5.2f}" for k, v in sets.items()))
```

### 6.2 Logit lens

```python
def logit_lens(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    hs = model(ids, output_hidden_states=True).hidden_states
    return [tok.decode(model.lm_head(model.model.norm(h[:, -1])).argmax(-1)) for h in hs]
```

---

## 7. Hour-by-hour plan

**Freeze the model at hour 9, whatever state it's in.**

| Hour | Pro A: main run | Pro B: ablations | Plus C: data + evaluation | Laptop / people |
|---|---|---|---|---|
| 0–1 | Confirm the units are active and A100 availability; write the resumable training script; smoke test | Same smoke test; prepare the ablation configs | **Stream all five language styles to the shared Drive** (Kannada first); build the tokenizer sample | `HF_HOME`; PyTorch test; **native speakers write and check templates (kn + hi)**; Gradio skeleton |
| 1–2 | Train the 48K tokenizer + round-trip checks | Train two 32K Kannada tokenizers (our regex vs GPT-4's) | **Tokenize everything into shards on Drive**; Aksharantar romanizer for Kanglish/Hinglish | Tokenizer benchmark (§6.1) → slides 1–4 |
| 2–7 | **Pretraining S-135M, stable phase** (checkpoint every 30 min) | Ablations ① ② ③ at XS / ~14M size | Build SFT data (instruct + reasoning, five styles); evaluation scripts; logit lens on Qwen3-0.6B | Chat tab with the `/think` toggle; voice stretch |
| 7–7.75 | **Decay / mid-training** | Write up the ablation results + plots | Evaluate base model checkpoints | Live meters (tokens, words/sec, RAM) |
| 7.75–8.5 | **Hybrid SFT** | Fine-tune the XS backup model (instruct + reasoning) | Prepare held-out evaluation sets | — |
| 8.5–9 | Stretch: rejection-sampling fine-tuning. **FREEZE at hour 9.** | — | — | — |
| 9–11 | Evaluation runs | Evaluation runs | Fill the **full results table** | Slides with the real numbers; rehearse 3×; backup video |
| 11–12 | Buffer | Buffer | Buffer | Buffer |

**Fallbacks:**
- If Pro A gets only an L4, the main run becomes S– (70M).
- If the hour-6 checkpoint is weak, start the decay early (safe with WSD).
- If SFT breaks the model, demo the XS backup model.
- **The tokenizer results, the regex finding and the two ablations are a complete research result on their own.**

**If time runs out, cut in this order:** ablation ③ → romanized SFT slices → rejection-sampling fine-tuning → voice. **Never cut** Kannada, the tokenizer benchmark, or ablation ①/②.

---

## 8. Demo script (90 seconds)

1. Paste one message in Kannada, Hindi, Kanglish and Hinglish. The token visualizer shows GPT-4 vs IndicMaxxer token counts for each.
2. The regex root cause: "the token tax starts before BPE even runs."
3. Instruct mode: a fast answer, offline on a 4 GB laptop, with tokens and words/sec on screen.
4. A word problem with `/think` in Kannada, then in Hinglish: the reasoning steps appear **in that language**. Show the accuracy vs extra tokens.
5. Logit lens: Qwen's middle layers decode to English, ours to Kannada/Hindi.
6. **What it costs:** the multilinguality ablation, the thinking cost, and the quality gap vs Qwen3-0.6B. Then the enterprise slide.

## 9. Pitch and leverage

> "IndicMaxxer is a 135M model we trained from scratch in one day on one Colab GPU. It speaks Kannada, Hindi and English, plus the Kanglish and Hinglish people actually type. GPT-style tokenizers break Indic words apart at every vowel sign; ours doesn't, so Kannada costs **X× fewer tokens** and Hindi **Y× fewer**. One model answers instantly, or reasons step by step in your own language with `/think`, for **+A% accuracy at B× the tokens**. It runs offline on a 4 GB laptop. Here is exactly what it costs: adding Hindi and English cost Kannada **C%**, and we trail a 4× bigger model by **D** points."

**Why ours, compared to other models:**
1. A tokenizer built from scratch for Indic scripts *and* romanized text, with a demonstrated root-cause fix.
2. Romanized and code-mixed Kannada and Hindi: how India actually types, and where most models struggle.
3. Native reasoning in each language, with logit-lens evidence.
4. Instruct and reasoning in one small model, with a per-request cost/accuracy switch.
5. A reasoning data generator that is always correct, scales without limit, and needs no teacher model.
6. On-device, private, zero per-token cost, with reserved tokens for tool calls and voice.
7. Open, reproducible benchmark and ablations.

## 10. Future work

Tamil/Telugu/Malayalam/Marathi with the same method; scale to 0.4–1B parameters; hybrid long-context layers; RL on reasoning; speech-style text plus voice; a retrieval memory layer; tool calling; per-domain adapters.

---

## Sources
- Schut et al., *Do Multilingual LLMs Think In English?*: https://arxiv.org/abs/2502.15603
- Zhong et al., *Beyond English-Centric LLMs*: https://arxiv.org/abs/2408.10811
- Sarvam-1: https://huggingface.co/sarvamai/sarvam-1 · https://www.sarvam.ai/blogs/sarvam-1
- PARAM-1: https://arxiv.org/pdf/2507.13390
- Dravidian monolingual vs joint models: https://arxiv.org/abs/2608.07727
- FineWeb2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · byte counts: https://arxiv.org/pdf/2512.03989
- IndicLLMSuite / Sangraha: https://arxiv.org/pdf/2403.06350
- nanochat: https://github.com/karpathy/nanochat
- Google AI plans (Colab units): https://one.google.com/about/google-ai-plans/
- Google Developer Program in AI Pro: https://blog.google/innovation-and-ai/technology/developers-tools/gdp-premium-ai-pro-ultra/
- Challenge source: arXiv 2607.24276
