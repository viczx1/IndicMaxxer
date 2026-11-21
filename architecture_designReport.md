# Kannada-First Hybrid (Mamba+Transformer) Model Plan

This plan evaluates a **Jamba-style hybrid architecture** (mixing Mamba/SSM layers with intermittent Transformer layers) for a Kannada-first small LLM. We emphasize verified facts and clarify tradeoffs: **hybrid models improve memory/throughput for long context but do *not* inherently reduce token counts or API billing**. Token inefficiency (“tokenizer tax”) is a separate issue, solved by tokenizer/vocab design. The hybrid approach’s benefits (and limits) are analyzed below, followed by a detailed implementation roadmap: architecture, tokenizer, data, training, evaluation, and project management. Equations and concrete metrics are provided throughout, with a focus on a compute-conscious solution for a small team. 

## 1. Hybrid Architecture Overview

A **Jamba-style hybrid** interleaves state-space (Mamba) layers with occasional Transformer (attention) layers in each block. In AI21’s Jamba model, each block has *8 layers*: 1 Transformer layer (with multi-head attention) and 7 Mamba (SSM) layers. This 1:7 ratio was chosen by ablation as a strong compute-efficient tradeoff. In total, Jamba’s published model has 4 such blocks (32 layers) with MoE modules on every 2<sup>nd</sup> layer. Figure below illustrates a Jamba block and layer types:

 *Figure: Jamba hybrid block (excerpt). Each block contains 1 Transformer layer (yellow) and 7 Mamba layers (pink). Mamba+MoE layers combine an SSM (“Mamba”) with MoE in the MLP branch. (Transformer layers include Attention+MLP; Mamba layers include SSM+MLP.)*  

- **Transformer layers:** Full self-attention with key/value cache. Good at precise recall and context alignment, but KV memory grows ∝*L* (sequence length).
- **Mamba (SSM) layers:** “Selective State Space” layers that process tokens recurrently (no KV cache), offering **linear-time sequence modeling**. They excel at long-range patterns and high throughput (Jamba reports ≈5× inference throughput vs. same-size Transformer). 
- **Mixture-of-Experts (MoE):** To boost capacity without huge compute, Jamba inserts sparse MoE sub-layers in some MLPs. For example, Jamba-12B used 16 experts (top-2 routing) every 2 layers, so only 12B of 52B total params are active at inference. We plan to include MoE similarly for capacity.

The hybrid design yields *two main benefits* for long-context tasks: **reduced KV memory** and **higher throughput**. In Jamba’s 256K-token tests, its KV cache was ~4 GB vs. 32–128 GB for comparable Transformers (an 8× reduction). This lets very long contexts (≥64K tokens) run on a single GPU (Jamba fits 128K–256K tokens in 80 GB). Throughput graphs show Jamba outperforming LLaMA/Mixtral on ≥8K contexts – e.g., 3× higher tokens/sec than Mixtral-8×7B at long lengths. (We will reproduce such benchmarks on our hardware.) 

**What hybrid does NOT solve:** The Kannada **tokenizer tax** remains. Token count per sentence is dominated by subword vocabulary, not model architecture. Indic languages like Kannada incur ~10–12× tokens vs. English under common BPE tokenizers, limiting effective context to ~12% as much text. A hybrid model *alone* cannot fix that; we must **re-design the tokenizer and vocabulary** for Kannada (see Sec.3). Only then can we truly reduce API costs or fit more Kannada content into a context window. The hybrid architecture is orthogonal: it **allows** larger contexts per GPU (which is useful once we reduce token count per content), but it does not inherently lower token usage.

## 2. Architecture Details

### 2.1 Block and Layer Layout

We adopt Jamba’s **block design**: each block has 8 layers, with 1 Transformer layer and 7 Mamba layers. The full model will stack multiple blocks. This 1:7 ratio is our **default hypothesis** (chosen by Jamba via ablations), but we plan ablation variants (e.g. 1:3, 1:5, 1:9) to verify optimal tradeoffs for our data/scale. Each block layout is: *[Mamba+MoE -> Mamba -> Mamba+MoE -> Mamba -> Transformer -> Mamba+MoE -> Mamba -> Mamba+MoE]*, as shown above (with MoE layers on every 2<sup>nd</sup> MLP).  

- **Residual & Norm:** Each sub-layer (attention or SSM or MLP) has residual connections and layer normalization. Jamba uses **RMSNorm** before each block sub-layer (including Mamba) to stabilize training. We will also use RMSNorm on Mamba layers and possibly LayerNorm on transformers, as in prior art.  
- **MLP and MoE:** Transformer layers have standard two-layer MLP (dense-GELU-dense). Mamba layers likewise include an MLP after the SSM. We will include MoE in *some* MLPs: e.g. every other block as in Jamba. Experts (16 experts, top-2) is a reasonable starting point, but tuning expert count vs. capacity vs. throughput will be part of design.  
- **Activation:** Use SiLU (Swish) in Mamba SSM as in [36][40], and GELU in Transformer MLP.  
- **Positional Encoding:** We will apply **Rotary Position Embeddings (RoPE)** in Transformer layers (common practice). Jamba found that even omitting explicit pos encodings gave similar results, presumably because Mamba’s recurrent state encodes position implicitly. However, to be safe we will include RoPE in the attention sublayers.  

In summary, each Jamba block contains alternating Mamba and Transformer layers with appropriate norms and residuals. The result should capture both fast sequential processing and dense attention as needed. 

### 2.2 Mamba vs. Transformer Roles

- **Mamba (SSM) layers:** Fully-recurrent SSM layers allow processing new tokens in O(1) time (no KV cache) and constant memory per layer. They excel at capturing long-range dependencies and maintaining context without blowing up memory. For our Kannada model, Mamba layers will handle the bulk of “heavy lifting” over long sequences, allowing large context windows.  
- **Transformer layers:** Though sparse, the periodic attention layers let the model “refresh” long-range interactions. Every 8<sup>th</sup> layer, a full self-attention is applied, acting as a checkpoint for detailed context and reasoning. This matches Jamba’s design intent of mixing precise contextual attention with Mamba’s broad context. We avoid characterizing this as a “fact-check” (as some blog phrased it) – in fact the Transformer’s role is to ensure any long-term dependencies or cross-token patterns are captured exactly. 

**Memory tradeoff:** By reducing attention layers, we greatly shrink the KV cache. If $d$ is model dimension and $L$ context length, a pure Transformer’s KV cache per layer is $O(2\,d\,L)$ (keys+values) floats. Replacing 7 out of 8 layers with Mamba cuts total KV storage by ~1/8. For example, Jamba’s 12B/256K configuration needed only ~4 GB of key-value memory vs. 32–128 GB for comparable transformer models. We will similarly benefit, enabling us to use longer contexts or fit model on smaller hardware.

**Throughput advantage:** Empirical results show Mamba layers compute faster per token than attention. Jamba reports ~3× throughput vs. Mixtral on long contexts. Our own benchmarks should confirm this: for example, in prior work Mamba had ~5× faster token generation than Transformers of similar size. We expect a hybrid with few attentions will still be faster than an all-attention model at high sequence lengths.

### 2.3 Architecture Comparison

The hybrid sits between two extremes: 

- A **pure Transformer** (all-attention) with high recall and quality but huge memory cost for long contexts.  
- A **pure Mamba** (all-SSM) that is very efficient but may underperform on tasks needing explicit attention (and pure Mamba’s quality on text was slightly lower than best Transformers). Indeed, Mamba-3B matched a 6B transformer in quality, so we expect some drop if we eliminate *all* attention. 

Hybrid aims to capture best of both: most layers are fast Mamba, but periodic attention maintains top-tier language modeling quality. Prior studies have shown hybrid outperform or match pure variants. We will verify that: as one baseline, we can implement a *pure-Mamba* model (same size, no attention) and a *pure-Transformer* model (same param count) to compare. This will isolate the value of the hybrid mix.

## 3. Tokenizer and Vocabulary Design

**Critical:** A specialized tokenizer is needed to eliminate the Kannada tokenization tax. We will build a **Kannada-English bilingual vocabulary** with morphological awareness:

- **Tokenizer algorithm:** Use a **SentencePiece Unigram** tokenizer (per [32]) rather than BPE, since Unigram often preserves morphemes better. Incorporate script-specific normalization (e.g. unicode normalization, removing diacritics consistently) as recommended. 
- **Vocabulary size:** Start with ~32K subwords (like BHARATI’s v3) and evaluate increasing (e.g. 50K) to see coverage tradeoff. Ensure all major Kannada characters and common words are directly in-vocab. Also include English subwords to enable bilingual text. 
- **Training data:** Train tokenizer on a **mixed corpus**: Kannada monolingual data (news, Wikipedia, crawled web), English corpora, and code-switched examples. We may give higher weight to Kannada during training to encourage native subwords (e.g. 50% Kannada text, 50% English). The BHARATI paper did a balanced mix across Indic+English which reduced token lengths dramatically (7.1→2.6 tokens per Indic technical term). We will similarly tune the mix. 
- **Morphology:** We can incorporate the idea of “morphology-aware” tokenizers: for Kannada’s agglutinative structure, ensure common suffixes/prefixes are tokens (BHARATI used reserved tokens for important terms). Although BHARATI is for classical languages, the strategy of preserving roots and affixes helps. 
- **Subword fertility target:** We aim to reduce Kannada fertility from ~12× down to near-unity. A realistic goal (by analogy to XLM-R or BHARATI) is to cut that by 50–80%. E.g., XLM-R cuts the Indic tax ~73%. If we achieve Kannada fertility ~3× English (vs. 12×), that quadruples effective context. We will measure word/char fertility on a held-out set (FLORES-200) to quantify improvements.

### Tokenizer Experiment Plan

- **Baseline:** off-the-shelf cl100k_base (GPT-4) vs multilingual (mBART/XLM) vs our custom Kannada-Unigram tokenizer. Measure fertility (tokens per Kannada word) and context shrinkage.
- **Ablation:** Test Unigram vs BPE variants, with/without script normalization, with different vocab sizes. Use [32]’s findings: script normalization and Unigram usually win for Indic. 
- **Evaluation:** Plot token length vs English baseline (tax multiplier) and usable characters at fixed token budget (like Figure 2 of [29]). Ensure metrics for budgeting.  

This tokenizer design is **orthogonal to model architecture** but crucial: once we have a smaller token count, the hybrid model’s large context capacity is fully utilized and API cost per text falls. 

## 4. Data Pipeline

We will gather and preprocess data covering Kannada, English, and mixed content:

- **Kannada monolingual:** Scrape Common Crawl for Kannada (using language tags), Kannada Wikipedia, news sites (e.g. Times of India – Kannada, regional news), digital libraries. Clean for encoding, remove boilerplate. Aim for ~50–100B tokens if possible (the more, the better the model). 
- **English monolingual:** Standard large corpora (English Wikipedia, BookCorpus, CC-100 English, OpenWebText, etc.), to give broad English knowledge and code-mixed examples. 
- **Bilingual and code-switched:** Include Kannada-English parallel corpora (FLORES, AI4Bharat corpora, etc.) to help mixed usage. Also gather social media text (Twitter, YouTube captions) with Kannada-English code-switching. For example, scraping Kannada tweets with hashtags or mixing English slang. 
- **Preprocessing:** Normalize, dedupe, and filter out low-quality text (excessive punctuation, etc.). Possibly segment at sentence level. Create mixed-shuffle dataset for training: e.g., 50% Kannada sentences, 30% English, 20% mixed, to enforce Kannada priority. 
- **Language ID tagging:** Optionally prepend a special token `<|kn|>` or `<|en|>` to each sequence to signal language (useful in bilingual models). This helps the model know which language it is generating. 
- **Tokenization:** All text is tokenized with the new SentencePiece model (as in Sec.3). We'll generate sharded binary datasets (e.g., HuggingFace datasets or WebDataset) for efficient loading.  

We will maintain a held-out evaluation set in Kannada (and mixed) to track perplexity and generation quality (see Sec.8). 

## 5. Model Scale and Training

### 5.1 Model Size and Depth

Given a small-team constraint, we recommend a **7–12B parameter active model** (comparable to Jamba’s 12B active, 52B total). For example, a 7B version (like Mixtral 7B) or 12B if resources allow. The MoE design means *total* parameters may be 4–5× larger (inactive experts) but only active params count. We choose base dimension \(d\approx 4096\), intermediate MLP \( \approx 4d\), 32 total layers (4 Jamba blocks). For a 7B active model, this might be ~4096 width, 32k MLP, 12 layers of attention (in 4 blocks), and 20 layers of Mamba+MoE.  

Alternatively, we could start smaller (3–4B) for prototypes. Ultimately, scaling laws suggest training more tokens is often better than a larger model. Using the Chinchilla scaling rule ~20 tokens per param, a 7B model would optimally train on ~140B tokens. We should budget for at least ~150–200B tokens (which implies ~60k H100 GPU-hours at 40% MFU, e.g. 60 GPUs × 1000h ~150k GPUh). A 12B active model (with MoE) might require proportionally more. We will refine after hardware planning. 

### 5.2 Training Schedule

- **Pretraining:** Train from scratch (no existing checkpoint has Mamba layers). Use AdamW with learning rate warmup and cosine (or polynomial) decay. For example, LR ∼1e-4 peak, linear warmup 5000 steps, decay over run.  
- **Batching:** Sequence length initially 1024 (for speed) and gradually increase (see *length warmup* below). We can use gradient accumulation to simulate large batches (e.g. 128k tokens per step).  
- **Sequence length warmup:** Start with short sequences (1k) for faster convergence, then gradually ramp to max context (e.g. 32k or more) over first 10–20% of training. This helps stabilize training for long-range models.  
- **Run length:** If we target 150B tokens, on e.g. 7B model, that’s \(6\times7e9\times150e9 = 6.3\times10^{21}\) FLOPs. Using 80 H100s (396 TFLOPS each @40% MFU), we’d need ~\(\frac{6.3e21}{396e12 \times 80 \times 3600} ≈ 138\,000\) GPU-hours (~5.8k GPU-days). This is large for a small team. We may start with fewer tokens (~50–100B) or smaller model, then consider longer runs later.  
- **Checkpointing:** Save periodic model checkpoints (e.g. every 5B tokens) for resuming or partial evaluation.  

We will monitor **perplexity and loss** on held-out dev sets (Kannada and bilingual). Also track FLOPs/sec and memory to ensure model trains without overflow. Use mixed-precision (BF16/F16) and gradient checkpointing to fit large model on GPUs.

### 5.3 Alternative Approaches

If compute is too limited, we may consider **continued pretraining** from an existing English model by converting layers to Jamba blocks. However, no public checkpoint has Mamba layers. We might repurpose an English GPT-like model by converting some layers to SSM (highly nontrivial). Probably simpler to train from scratch with random init of Mamba. Another idea: train the Mamba backbone first (as in [40]), then insert sparse attention layers and fine-tune. This two-stage training could simplify optimization, but is experimental.

## 6. Implementation and Infrastructure

### 6.1 Codebase and Libraries

- **Framework:** We will use PyTorch (likely with FlashAttention, cutlass kernels). There are open-source Mamba implementations (by Tri Dao’s `state-spaces` library) and AI21 has released Jamba code under Apache-2.0. We can leverage [36]’s code or HuggingFace’s `transformers` if Jamba is integrated. We may modify a `transformers` GPT2 or LLaMA-like codebase to include Mamba layers and MoE (similar to Switch-Transformer code).  
- **Modules:** Implement a `JambaLayer` class wrapping either an `AttentionBlock` or `MambaBlock`. The `MambaBlock` uses the SSM module (implemented via linear scan as in [40]) with RMSNorm and an MLP (with optional MoE). The `TransformerBlock` uses standard multi-head attention + MLP + RMSNorm. The model is a stack of these layers.  
- **Parallelism:** Use DeepSpeed or FairScale for distributed training. MoE needs tensor-model-parallelism support (e.g. DeepSpeed MoE). We assume a multi-GPU machine (like 8×A100 or 4×H100 with NVLink) or a cluster.  
- **Dataset pipeline:** Tokenize on-the-fly or pre-tokenize to LMDB/Parquet. Use efficient sampler (e.g. HuggingFace’s `datasets`) with optional bucketed batches for similar lengths.  

### 6.2 Pseudocode Snippet

Below is a high-level pseudocode outline of the training loop:

```python
model = JambaModel(num_blocks=N, layers_per_block=8, width=d, ...)
optimizer = AdamW(model.params(), lr=1e-4, weight_decay=0.1)
for step, batch in enumerate(data_loader):
    tokens = batch['input_ids']  # (B, L)
    # Forward through model
    outputs = model(tokens)
    loss = outputs.loss  # e.g. cross-entropy
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    model.zero_grad()
    if step % eval_interval == 0:
        eval_ppl = evaluate(model, eval_data)
        log_metrics(step, loss, eval_ppl)
        save_checkpoint(model, optimizer, step)
```

We will use mixed-precision (AMP) and gradient accumulation to reach effective batch sizes of hundreds of thousands of tokens.

## 7. Inference and Serving

- **Generation:** In autoregressive inference, Mamba layers only need to update their hidden state (constant per token). Transformer layers still use KV cache. We implement a hybrid inference: maintain KV caches for attention layers only. For long-context decoding, this yields lower memory per token.  
- **Batching:** We will support batched decoding (standard beam or sampling). Mamba’s recurrent nature allows efficient sequential generation.  
- **Quantization:** To reduce memory and latency, we can apply 4-bit or 8-bit weight quantization (e.g. GPTQ) once model is trained. Because the model has MoE and potentially large dimension, quantization can greatly reduce serving cost. We must ensure quantization works with Mamba layers (likely yes, as they are linear layers).  
- **Streaming:** For applications needing streaming (prefix requests), Mamba’s lack of KV means we only carry its hidden state. The attention layers still use KV, but we can stream up to their layer’s capacity.  

Hardware: The model should run on a single 40–80GB GPU for moderate prompts (up to 64K tokens). Deploy with NVIDIA Triton or HuggingFace Inference endpoints. If multiple GPUs are available, we can shard transformer layers across GPUs.

## 8. Evaluation Plan

We will evaluate on multiple axes:

- **Tokenization metrics:** Measure Kannada and English tokenization `fertility = tokens/word` on held-out text. Compute *tokenizer tax* (Kannada tokens per English token). Validate that our new tokenizer indeed lowers the tax (target ≪12×).  
- **Perplexity:** Compute cross-entropy perplexity on held-out Kannada text (monolingual dev set) and English dev set. Also on mixed text (e.g. code-switched utterances). Compare hybrid vs baseline models.  
- **Generation quality:** Use standard benchmarks adapted to Kannada (if available) or translation proxies. For example, Kannada QA or summarization tasks, or bilingual benchmarks (FLORES can test en-kn translation quality via prompting). Zero/few-shot tasks: apply prompts in Kannada and see accuracy. Also measure English tasks to ensure bilingual capacity.  
- **Long-context retrieval:** Test models on tasks requiring long context. E.g. give 64K or 128K of document and ask for summary or Q&A. Confirm the model uses the entire context coherently.  
- **Throughput & latency:** Measure tokens/sec and latency for generation of a typical prompt at various lengths. Compare hybrid vs pure-transformer runs (if baseline is available).  
- **Memory usage:** Profile GPU memory for a given context length (KV + activations). Verify that hybrid indeed uses less KV as expected (calculate KV memory = $2 \times d \times L \times$ sizeof(float) – should match observations).  
- **Energy/Cost:** Log power usage if possible, or use proxy by GPU utilization and runtime. Estimate inference cost per token for hybrid vs baseline (especially important if deploying as API).  
- **Ablation studies:** Isolate *tokenizer* vs *architecture*. For instance, test: (a) new tokenizer + pure Transformer, (b) old tokenizer + hybrid, (c) new tokenizer + hybrid. This will show how much each contributes to end metrics (perplexity, cost, throughput).  

We will track all metrics in an experiment dashboard (e.g., Weights & Biases), ensuring reproducibility. Provide scripts for automatic evaluation (PPL calc, gen tasks, token counts). 

## 9. Experimental Matrix and Tracking

We plan experiments along two axes: **Architecture** and **Tokenization**:

1. **Arch A:** Pure Transformer (7B) with our tokenizer.  
2. **Arch B:** Pure Mamba (7B) with our tokenizer.  
3. **Arch C:** Hybrid Jamba 1:7 (7B active) with our tokenizer (baseline).  
4. **Arch D:** Hybrid variants (1:3, 1:5, 1:9 ratios) – each at 7B active.  
5. **Arch E:** Smaller/bigger scales (3B, 12B active) to see scaling.  

For each, use **Tok1:** our bilingual Unigram tokenizer, **Tok2:** baseline GPT-3.5 (cl100k), **Tok3:** multilingual baseline (XLM-R). This yields e.g. [A-Tok1, A-Tok2, A-Tok3], etc. Key comparisons: C-Tok1 vs C-Tok2 (showing benefit of our tokenizer on hybrid), A-Tok1 vs C-Tok1 (benefit of architecture), etc.

**Metrics to record:** PPL (kn, en, en+kn), tok-per-second, max feasible context, KV memory usage, generation accuracy on tasks, energy (approx). Also track training loss curves.

Maintain a table of results for each run. Track random seeds and hyperparams. Ensure each experiment is logged with metadata (arch config, tokenizer, compute used).

## 10. Compute, Hardware and Cost

We assume access to GPUs (e.g. a small cluster of A100/H100s). Example estimate for a 7B model on 100B tokens:

- Compute ≈ 6×7e9×1e11 = 4.2e21 FLOPs. 
- On an H100 (396 TFLOPS @40% MFU), 1 GPU-hour does 396e12 * 3600 ≈ 1.43e18 FLOPs. 
- GPU hours ≈ 4.2e21 / 1.43e18 ≈ 2939 GPU-hours (~123 days on 1 GPU). 
- If using 8 GPUs in parallel: ~368 GPU-hours (~15 days).  
- At $3/hr, that’s $1104 of GPU time. Accounting 40% utilization overhead, say $3000.  
- Add 2–3× overhead for inefficiencies (3000 → ~$9000). This is plausible for a small project.  

For context, the ComputingPower.org estimate for 7B on 2T tokens was ~$175k, which is far above a small team budget. Hence we scale down (fewer tokens, maybe using cheaper spot instances). We may also leverage academic grants or cloud credits.

**Memory:** A single 80 GB GPU with 8-bit weights can handle Jamba-12B at 128K context. For 7B/256K, 48–80 GB should suffice with 8-bit quantization or ZeRO-offloading. Training will use multi-GPU (e.g. 4×80GB for optimizer states). Estimate VRAM for training by config + batch size (likely ~40 GB usable per GPU with DeepSpeed ZeRO2/3).

**Energy:** Rough estimate: ~0.5–1 kWh per GPU-hour. Training 3000 GPU-hr → ~1500–3000 kWh, plus overhead. Minor in scope; inference will be more impactful cost-wise via API, which is why reducing tokens (via tokenizer) is crucial.

## 11. Milestones & Timeline

1. **Data & Tokenizer (Month 1–2):** 
   - Collect/clean Kannada and English corpora. 
   - Develop and test tokenizer (Unigram with normalization). Validate on test sets. 
2. **Prototype Model (Month 2–3):** 
   - Implement hybrid architecture code (Jamba blocks). 
   - Run small-scale (3B, short tokens) experiments to debug. 
3. **Full Training Setup (Month 3–4):** 
   - Scale to target size (7–12B active). 
   - Configure distributed training (DeepSpeed) and begin longer runs. 
4. **Ablations & Evaluation (Month 5–6):** 
   - Train alternative configs (different ratios, tokenizers). 
   - Evaluate on metrics (perplexity, throughput, tasks). 
5. **Tuning & Finishing (Month 7–8):** 
   - Refine best model (more tokens or tuning). 
   - Apply SFT/alignment if desired (e.g. instruction tuning on Kannada tasks). 
   - Final evaluation and documentation.

**Deliverables:** Code repo (model, training scripts, tokenizer), trained model checkpoints, evaluation results, and project report (like this plan expanded with results). 

**Risks:** 
- **Data scarcity:** Kannada corpora may be smaller; mitigate by aggressive web scraping and data augmentation. 
- **Training instability:** Novel architecture may require tuning (gradient clipping, LR tuning). 
- **Compute overrun:** If training is too costly, we fallback to smaller model or fewer tokens. Possibly leverage multi-stage training (start with Mamba-only warmup). 
- **Token quality:** If our tokenizer underperforms, alternate strategies (e.g. separate tokenizers by language with soft-mixing) can be tried. 

**Failure criteria:** If hybrid yields no performance gain or training diverges, we’ll revert to a simpler model (e.g. pure Transformer with our tokenizer) as a baseline deliverable. 

## 12. Summary

The proposed Jamba-style hybrid can support much longer Kannada context and faster generation, which indirectly helps our goals. **However**, it does *not* solve the core “token tax” problem by itself. We must pair it with a more efficient Kannada tokenizer. The plan above details the architecture design, tokenizer and vocab strategy, training regimen, and comprehensive evaluation needed to build and assess this Kannada-first bilingual model. We distinguish confirmed facts (from Jamba/Mamba research) from our engineering hypotheses (e.g. 1:7 ratio, model size) and will experimentally verify each. This compute-conscious approach focuses on maximal gains for Kannada language capabilities with practical resource use. 

**Sources:** We cite AI21’s Jamba paper and blog, the Mamba SSM paper, and recent tokenizer research to inform design. All recommended strategies are grounded in these references.  

