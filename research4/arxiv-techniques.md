# arXiv Research: Newest Inference Speedup Techniques for Bandwidth-Bound MoE Decode (2025–2026)

**Scope:** papers from the last 18 months (2025-03 → 2026-09) relevant to Qwen3.8-Flash-Next ~105B A48 (36 GDN + 12 QSA layers, MoE 512 experts top-10 + shared, NVFP4 weights, FP8 KV) on NVIDIA GB10 DGX Spark (sm_121, 128GB unified LPDDR5x, ~273 GB/s effective bus), vLLM 0.1.dev with MTP spec-decode (acceptance ~2.95 @ k=3).
**Goal:** 80 tok/s single-stream prose (from ~50) at ZERO quality loss; faster/stabler C4/C8.
**Method:** arXiv API search across 25 targeted queries (402 raw hits → 334 unique → 296 in-window), abstract triage of 60 shortlisted papers, full-text extraction of 35. Every claim cites its arXiv URL.

---

## 1. Speculative decoding beyond EAGLE-3

### 1.1 Bole: Efficient Tree Speculation for Hybrid-Attention Language Models
- **arXiv:** [2608.01651](https://arxiv.org/abs/2608.01651) — 2026-08-03
- **Mechanism:** Kernel–runtime co-design for tree speculation on hybrid (full-attn + linear-attn) LLMs. Existing tree systems materialize a full recurrent state per draft node; Bole uses *factorized state management* (value-tiled parallel verifier for GDN layers) so verification latency/memory no longer scale with tree size, plus an offline profiler that picks the max verification budget N s.t. T_ver(N) ≤ (1+ε)·T_dec.
- **Measured:** **Benchmarked directly on GB10 DGX Spark (128GB LPDDR5x, 273 GB/s)** with Qwen3-5-9B/27B (GDN hybrids) and native MTP heads as drafters. Peak speedups 3.x on A100, **up to 4.x on GB10** over SGLang-Tree/AdaServe baselines; state materialization was 86% of serial verification time on GB10 (vs 38% on A100), so the gain is *largest* on low-bandwidth unified memory. [Source](https://arxiv.org/html/2608.01651v1)
- **Quality:** Lossless (standard speculative verification, unchanged distributions).
- **Port effort to vLLM fork: M–L.** SGLang-side implementation; need factorized GDN tree-verification kernels + budget profiler ported into vLLM's MTP path.
- **Composes with MTP: YES** — evaluated using each model's native MTP head as the drafter.

### 1.2 TreeWY: Speculative Verification for Gated DeltaNet Hybrids
- **arXiv:** [2608.20961](https://arxiv.org/abs/2608.20961) — 2026-08-21
- **Mechanism:** Replaces per-position recurrent-state snapshots in GDN verification with a tree-structured WY transform of the gated delta rule: every draft node's output computed via one triangular solve; states held once per layer regardless of tree width w.
- **Measured:** On Qwen3.5-35B/397B-class hybrids: cuts speculative recurrent-state memory dramatically at identical acceptance length; freed memory converted to higher concurrency — 1.83x TPOT improvement at 0.96x peak memory vs ReplaySSM; mean end-to-end latency 1.17x better at 1.96x peak usage. Trees "correct but not yet a throughput win" (piecewise, not fused). [Source](https://arxiv.org/html/2608.20961v1)
- **Quality:** Acceptance length statistically identical to snapshot baseline (lossless verification).
- **Port effort: L** (new WY-transform GDN verification kernel; FLA-adjacent).
- **Composes with MTP: YES** (any drafter; it's a verifier-side change).

### 1.3 SpecLA: Efficient Speculative Decoding for Linear-Attention Models
- **arXiv:** [2607.16673](https://arxiv.org/abs/2607.16673) — 2026-07-18
- **Mechanism:** Runtime for stateful linear-attention targets: chain/branch-aware verification, accepted-state recovery via factor-buffer reuse (2.28x latency reduction), fused commit-and-verify (1.44x), delayed state update.
- **Measured:** 1.70x end-to-end over AR on mixed suite/GSM8K/HumanEval (Qwen3-adjacent linear-attn targets); oracle-acceptance projections to 3.62–4.x. [Source](https://arxiv.org/html/2607.16673v1)
- **Quality:** Lossless (reports first-token match).
- **Port effort: M–L.**
- **Composes with MTP: YES** (drafter-agnostic).

### 1.4 EVICT — Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding
- **arXiv:** [2605.00342](https://arxiv.org/abs/2605.00342) — 2026-05-01
- **Mechanism:** For MoE targets, wider draft trees activate the *union* of experts across branches, inflating verification cost. EVICT (training-free, hyperparameter-free) truncates the tree to the cost-effective prefix using drafter confidence signals before verification.
- **Measured:** 1.35x over AR, **1.21x over EAGLE-3** on production MoEs, consistent across temperatures; explicitly CUDA-graph compatible (important for vLLM). [Source](https://arxiv.org/html/2605.00342v1)
- **Quality:** Lossless.
- **Port effort: S–M** (draft-tree selection logic in the MTP verify path; no new kernels).
- **Composes with MTP: YES** — orthogonal to drafter; directly relevant to 512-expert top-10 target.

### 1.5 EcoSpec — Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for MoE
- **arXiv:** [2607.12696](https://arxiv.org/abs/2607.12696) — 2026-07-14
- **Mechanism:** Identifies "expert scattering": confidence-driven draft selection routes tree nodes to disjoint experts. EcoSpec uses a lightweight expert predictor + dynamic expert buffer to prefer draft paths that reuse already-loaded experts; standard verification rule untouched.
- **Measured:** Consistent speedups across production-scale MoEs (up to ~1.2x over confidence-only selection). [Source](https://arxiv.org/html/2607.12696v1)
- **Quality:** Lossless.
- **Port effort: M** (needs a small expert-predictor trained per model).
- **Composes with MTP: YES.**

### 1.6 AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets
- **arXiv:** [2608.02989](https://arxiv.org/abs/2608.02989) — 2026-08-04
- **Mechanism:** Verifier-side expert selector combining target-router scores with offline-estimated commitment probabilities; self-sizes the expert set activated during verification (activates a subset, falls back for correctness).
- **Measured:** Mean 1.x speedup over Standard SD on all 12 model–task pairs; up to 2.x combined. [Source](https://arxiv.org/html/2608.02989v1)
- **Quality:** Lossless (fallback preserves outputs).
- **Port effort: M.**
- **Composes with MTP: YES.**

### 1.7 Osprey: Target-Agnostic Pre-training Makes Stronger Drafters
- **arXiv:** [2609.09338](https://arxiv.org/abs/2609.09338) — 2026-09-08
- **Mechanism:** Pre-trains the drafter once (no target hidden-state/logit distillation) so acceptance doesn't collapse under workload shifts; attaches cheaply to any target.
- **Measured:** Higher acceptance and end-to-end speedup than EAGLE-3 baselines, especially OOD. [Source](https://arxiv.org/html/2609.09338v1)
- **Quality:** Lossless.
- **Port effort: M** (replace/augment MTP drafter; drafter weights must be trained).
- **Composes with MTP: PARTIAL** (alternative drafter to the native MTP head).

### 1.8 Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding
- **arXiv:** [2604.02047](https://arxiv.org/abs/2604.02047) — 2026-04-02
- **Mechanism:** Shapes trees per token-source: n-gram context copies get deep-narrow chains (high acceptance, ~6x median higher than statistical predictions), statistical drafts get breadth; allocates node budget anisotropically.
- **Measured:** 4.3x lossless speedup vs AR (5 models × 5 benchmarks), 12–33% over isotropic training-free trees; matches/exceeds EAGLE-2 wall-clock on 2 of 3 models — training-free. [Source](https://arxiv.org/html/2604.02047v1)
- **Quality:** Lossless.
- **Port effort: S–M** (tree-scheduler change; complements their existing PLE n-gram table!).
- **Composes with MTP: YES** (hybrid drafting sources).

### 1.9 NanoSpec: Minimalist In-Context Vocabularies for the Draft LM-Head
- **arXiv:** [2605.26444](https://arxiv.org/abs/2605.26444) — 2026-04-08
- **Mechanism:** Training-free per-step dynamic draft vocabulary (<3k active tokens, 40x reduction from ~30k static sub-vocab methods) built from temporal locality; GPU-resident state + async gather to make sparse access fast.
- **Measured:** Draft inference latency −51.6%; 1.32x end-to-end over EAGLE-2. [Source](https://arxiv.org/html/2605.26444v1)
- **Quality:** Lossless if fallback on coverage miss (their design includes fallback).
- **Port effort: S–M.** Direct upgrade over their planned draft-vocab reduction (Qwen3.8 ships 47k draft vocab).
- **Composes with MTP: YES.**

### 1.10 SlimSpec: Low-Rank Draft LM-Head
- **arXiv:** [2605.10453](https://arxiv.org/abs/2605.10453) — 2026-05-11
- **Mechanism:** Factorizes the draft LM-head (V×d → V×r · r×d) so draft logits cost r-projection + small GEMM; no vocab curation or special sampling logic.
- **Measured:** 4–5x draft LM-head latency reduction (vs ~60% for VocabTrim/SpecVocab), +8–9% end-to-end speedup over standard EAGLE-3 pipeline; acceptance length ≈ full-vocab baseline. [Source](https://arxiv.org/html/2605.10453v1)
- **Quality:** Lossless verification; drafter acceptance preserved (marginally lower AL but net-positive).
- **Port effort: M** (needs distilled low-rank head weights for the MTP head; kernel trivial).
- **Composes with MTP: YES** — applies to the MTP head's own 47k-vocab projection.

### 1.11 PACER: Blockwise Pre-verification with Adaptive Length
- **arXiv:** [2602.01274](https://arxiv.org/abs/2602.01274) — 2026-02-01
- **Mechanism:** Lightweight trainable pre-verification layer decides draft length per step before target verification (optimal γ varies per step); prunes wasted draft passes.
- **Measured:** vs best fixed-window SD: draft passes −4,837, target passes 3,047→1,150 on their eval; overall 1.x speedup on top. [Source](https://arxiv.org/html/2602.01274v1)
- **Quality:** Lossless.
- **Port effort: M.**
- **Composes with MTP: YES** (dynamic k on top of MTP; their k=3 today is static).

### 1.12 EntMTP: Entropy-Guided Multi-Token Prediction
- **arXiv:** [2606.27550](https://arxiv.org/abs/2606.27550) — 2026-06-25
- **Mechanism:** Replaces static MTP tree topology with entropy-conditioned speculation depth (deep in low-entropy regions, shallow in high-entropy).
- **Measured:** Up to ~3.x speedup vs vanilla, 1.15x vs Hydra (dynamic-depth class). [Source](https://arxiv.org/html/2606.27550v1)
- **Quality:** Lossless (verification unchanged).
- **Port effort: S–M** (scheduler policy on top of MTP head).
- **Composes with MTP: YES** — literally an MTP enhancement.

### 1.13 Windowed-MTP: Removing the Full-Context Draft-KV Tax
- **arXiv:** [2607.21535](https://arxiv.org/abs/2607.21535) — 2026-07-23
- **Mechanism:** Native MTP draft heads run full attention over the entire KV cache each draft step; at long context this dominates. Windowed draft attention caps draft-KV reads; *sharpens* under hybrid/linear-attention targets.
- **Measured:** At 1M ctx (γ=6) the draft phase alone adds +92% to +138% per step — windowing removes this and lifts acceptance; never regresses end-to-end. [Source](https://arxiv.org/html/2607.21535v1)
- **Quality:** Verified output distribution preserved (greedy-equal claims; window is draft-side only).
- **Port effort: S** (attention-mask change on the MTP module).
- **Composes with MTP: YES** — this IS an MTP fix. **Directly relevant to C4/C8 stability.**

### 1.14 JetSpec: Parallel Tree Drafting
- **arXiv:** [2606.18394](https://arxiv.org/abs/2606.18394) — 2026-06-16
- **Mechanism:** JetFlow parallel (non-AR) drafter generates entire tree in one pass, avoiding EAGLE-3's sequential drafting overhead; causal correction preserves path dependence.
- **Measured:** 7–10x vs AR on math/coding (H100), >4x end-to-end; beats EAGLE-3 at every budget. [Source](https://arxiv.org/html/2606.18394v1)
- **Quality:** Lossless.
- **Port effort: L** (new drafter model + integration).
- **Composes with MTP: COMPETES** (replaces MTP head).

### 1.15 CLP: Collocation-Length Prediction (Backbone-as-Draft)
- **arXiv:** [2606.10935](https://arxiv.org/abs/2606.10935) — 2026-06-09
- **Mechanism:** Fixes MTP head/backbone competition (first-token head conflict — root cause of repetitive outputs in MTP acceleration); predicts collocation length for zero-loss multi-token acceptance, 200x fewer parameters than draft models.
- **Measured:** Best speedup 1.x at k=3; zero quality loss, no second model. [Source](https://arxiv.org/html/2606.10935v1)
- **Quality:** **Zero loss by construction** (backbone logits always authoritative).
- **Port effort: M.**
- **Composes with MTP: YES/ALTERNATIVE.**

### 1.16 Other spec-decode notables (abstract-reviewed)
- **Cacheback** [2511.21699](https://arxiv.org/abs/2511.21699) (2025-11-15): pure LRU n-gram cache drafting, training-free, SOTA among model-free. Composes with MTP as a second source. Port S.
- **CATS** [2605.11186](https://arxiv.org/abs/2605.11186) (2026-05-11): cascaded adaptive tree speculation for memory-limited devices (flash-staged weights, Jetson Orin measured); up to 5.08x vs AR. Relevant pattern for PLE/SSD staging. Port M.
- **S2-MoE** [2608.15018](https://arxiv.org/abs/2608.15018) (2026-08-15): routing-aware adaptive speculative expansion + reuse-aware verification for edge MoE; 2.3x on Jetson Orin. Lossless. Port M.
- **DraftExpert** [2607.24434](https://arxiv.org/abs/2607.24434) (2026-07-27): expert-offload-aware self-spec decoding (expert set expansion formalized); for CPU/Flash-staged experts — matches their SSD-mmap PLE setup. Port M.
- **CAS-Spec** [2510.26843](https://arxiv.org/abs/2510.26843) (2025-10-30): cascade of dynamically switchable self-speculative drafters, on-the-fly lossless. Port M.
- **SPD (Speculative Pipeline Decoding)** [2605.30852](https://arxiv.org/abs/2605.30852) (2026-05-29): pipeline-stage drafting hides drafter latency; higher theoretical ceiling than EAGLE-class. Port L.
- **ReTrace** [2608.29748](https://arxiv.org/abs/2608.29748) (2026-08-30): reuses rejected-suffix information to re-draft after rejection. Port M.

---

## 2. Memory-bandwidth-bound batch-1 decode

### 2.1 MonoMoE: Fused Mega-kernel for Quantized MoE Decoding
- **arXiv:** [2609.04244](https://arxiv.org/abs/2609.04244) — 2026-08-19
- **Mechanism:** Token-major grouped GEMMs waste bandwidth at batch-1 (tile padding, short-lived grids, separate quant/act/reduce kernels, inter-kernel idle even under CUDA graphs). MonoMoE: **weight-major persistent megakernel** — one launch per MoE block, continuous weight streaming, auxiliary work overlapped.
- **Measured:** NVIDIA **H200**, block-wise FP8 MoE, B∈{1,2,4,8}: up to **1.84x** vs FlashMoE-FP8 adaptation; up to **18% end-to-end TPOT reduction** in vLLM; speedups over vLLM grouped GEMM 1.x–1.8x across models. [Source](https://arxiv.org/html/2609.04244v1)
- **Quality:** Lossless (same math, different scheduling).
- **Port effort: M** — already integrated into vLLM (fork-friendly); needs NVFP4 path + sm_121 (tcgen05 absent on sm_121? — GB10 is sm_120/121 family, uses different tensor core path than H200's sm_90; kernel rewrite for GB10 tcgen05-lite, hence M not S).
- **Composes with MTP: YES** (orthogonal kernel-layer change; helps verification batch too).

### 2.2 SonicMoE: IO- and Tile-aware MoE Optimization + Token-Rounding Routing
- **arXiv:** [2512.14080](https://arxiv.org/abs/2512.14080) — 2025-12-16
- **Mechanism:** For fine-grained sparse MoEs: memory-efficient dispatch (minimized activation IO), fused gather kernels with ping-pong scheduling, and **token-rounding (TR) routing** that eliminates grouped-GEMM padding without changing expected FLOPs/outputs.
- **Measured:** +43% forward-pass over a highly optimized DeepGEMM baseline on fine-grained 7B MoE; **on Qwen3-Next-80B-A3B (K/E = 10/512 — the exact expert topology of our target)** TR adds **19%+ over top-K token-choice routing**; up to 92–120% over ScatterMoE/MoMoE forward for sparse fine-grained configs. [Source](https://arxiv.org/html/2512.14080v1)
- **Quality:** TR is **quality-neutral in expectation** (rounding preserves model FLOPs; they report no quality loss); kernels lossless.
- **Port effort: M** (open-sourced: github.com/Dao-AILab/sonic-moe; adapt to NVFP4 + vLLM).
- **Composes with MTP: YES.**

### 2.3 Minima — NVFP4 W4A4 for the GDN half of a hybrid (evidence base)
- **arXiv:** [2609.04098](https://arxiv.org/abs/2609.04098) — 2026-09-03
- **Mechanism:** Tests the "recurrent layers need ≥8-bit" intuition on Qwen3.5-27B-style hybrid (48 GDN + 16 attn): NVFP4 W4A4 on **all 496 linear layers including GDN decay/write gates**. Mechanism: NVFP4 block scaling neutralizes residual-stream outliers; gate parameterizations compress quant noise; delta-rule overwrite erases state error faster than decay horizons.
- **Measured:** Matches BF16 within noise on PPL@4K/32K, MMLU-Pro, GSM8K, AIME'25, GPQA-D, LiveCodeBench, RULER@64K. [Source](https://arxiv.org/html/2609.04098v1)
- **Quality:** Near-lossless (PTQ, no retrain).
- **Port effort: L** (extend NVFP4 to the currently-BF16 dense/GDN GEMMs of Flash-Next; the 73% dense-BF16-GEMM bottleneck is a *byte* problem — halving-to-quartering bytes ≈ direct latency win at 180–200/273 GB/s utilization).
- **Composes with MTP: YES.**
- Related evidence: [2603.08747](https://arxiv.org/abs/2603.08747) (2026-03-05) layer/block-wise NVFP4-vs-MXFP4 sensitivity diagnosis.

### 2.4 DeltaLog: Deferred Materialization of Recurrent States
- **arXiv:** [2608.15533](https://arxiv.org/abs/2608.15533) — 2026-08-16
- **Mechanism:** Eager GDN/KDA/RWKV decoding writes the full recurrent state every token (write amplification; 32→44% of decode latency at B=64 on Qwen3.5-class models). DeltaLog keeps a dense base state + compact per-token delta factors, materializing lazily.
- **Measured:** Recurrent-state update kernel **1.69x on H200, 1.86x on RTX 4090**; serving-level gains up to ~1.20x for one-token decode; 2–20x serving speedups over dense-recurrent baselines in batched regimes. [Source](https://arxiv.org/html/2608.15533v1)
- **Quality:** Lossless (semantics unchanged; mixed-precision optionality analyzed).
- **Port effort: M** (FLA kernel layer; model-surgery-free).
- **Composes with MTP: YES** (also reduces state-rollback cost in verification).

### 2.5 DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization
- **arXiv:** [2608.27513](https://arxiv.org/abs/2608.27513) — 2026-08-27
- **Mechanism:** First PTQ study of GDN/KDA recurrent states (commonly FP32): decay-aware mixed-precision (INT8 states where decay forgets error fast). State updates are a bandwidth-bound latency term.
- **Measured:** TPOT improvements vs SGLang FP32-state baseline (≈1.0x KDA at large batch, higher for GDN; recurrent-update operator dominates at 20.3% of decode latency in their B=256 breakdown — at batch-1 the share is smaller but the update is per-step). [Source](https://arxiv.org/html/2608.27513v1)
- **Quality:** PTQ, near-lossless with decay-aware assignment.
- **Port effort: S–M.**
- **Composes with MTP: YES.**

### 2.6 Weight-only quant / dequant-fusion kernel advances
- **CodeGEMM** [2512.17970](https://arxiv.org/abs/2512.17970) (2025-12-19): codebook GEMM *without* dequant — precomputed centroid×activation partial sums ("Psumbook") gathered by index; kills per-element LUT/dequant overhead of 2–3-bit codebook methods. Port M (kernels; applies to dense GEMV share).
- **OASIS** [2507.23035](https://arxiv.org/abs/2507.23035) (2025-07-30): outlier-aware LUT GEMM with dual-side quant; avoids both WOQ dequant cost and INT-WAQ quality loss. GPU-measured. Port M.
- **HBQ** [2609.00450](https://arxiv.org/abs/2609.00450) (2026-08-31): design-space exploration of block quantization — bigger blocks amortize dequant/accumulation; gives principled block-size choice for NVFP4-class formats. Port S (config guidance).
- **SPARQLe** [2606.00365](https://arxiv.org/abs/2606.00365) (2026-05-29): exploits zero-concentration in high-order activation bits for sub-precision activations on quantized datapaths. Port M.
- **W4A16 Ascend kernel study** [2601.16536](https://arxiv.org/abs/2601.16536) (2026-01-23): clean analysis of vector-core dequant fusion + split-K for skinny GEMV on a decoupled architecture — transferable design pattern for sm_121 skinny GEMMs. Port M (reference).

### 2.7 Activation sparsity exploitation
- **Celty** [2608.01536](https://arxiv.org/abs/2608.01536) (2026-08-02): SpMSpV kernel + format co-design for dual-sparse (pruned weights × runtime activation sparsity) single-user decoding — the batch-1 sparsity target. Port L.
- **SharQ** [2606.26587](https://arxiv.org/abs/2606.26587) (2026-06-25): training-free online sparse–dense decomposition bridging N:M activation sparsity and FP4 quant (outliers go sparse-dense path). Composes with NVFP4 weights. Port M.
- **Dynamic sparsity in quantized inference** [2511.04477](https://arxiv.org/abs/2511.04477) (2025-11-06): makes dynamic sparsity compatible with group-wise quantization (they interleave poorly by default). Port M.
- **WiSparse** [2602.14452](https://arxiv.org/abs/2602.14452) (2026-02-16), **SVD contextual sparsity predictors** [2603.14110](https://arxiv.org/abs/2603.14110) (2026-03-14): training-free predictor construction; note applicability is to ReGLU/FFN dense blocks — our model is MoE, so treat as dense-layer-only lever. Port M.
- **Caveat for this model:** Flash-Next's FFN is MoE (already sparse by routing); activation sparsity mainly helps the dense GEMM 73% share only if dense layers (QSA projections, embed/norm, lm_head) show contextual sparsity — measure before investing.

### 2.8 Cross-layer weight sharing
- **CommonKV** [2508.16134](https://arxiv.org/abs/2508.16134) (2025-08-22): cross-layer KV-cache/parameter sharing for compression — KV-side, marginal here (FP8 KV already small; QSA layers are 12/48).
- **Share Your Attention** [2508.04581](https://arxiv.org/abs/2508.04581) (2025-08-06): matrix-dictionary-learning weight sharing across layers — requires retraining-ish distillation → violates zero-quality-loss/low-effort bar. **Not recommended here.**
- **KernelDNA** [2503.23379](https://arxiv.org/abs/2503.23379) (2025-03-30): dynamic kernel sharing via decoupled naive adapters — again a training-time lever. Skip.

### 2.9 Measurement/caution references
- **Memory-Bound but Not Bandwidth-Limited** [2605.30571](https://arxiv.org/abs/2605.30571) (2026-05-28): batch-1 decode on H100/A100/L40S/L4 rarely achieves peak-bandwidth-scaled latency (power/clock, cache effects) — calibrates what "180–200 of 273 GB/s" should optimally be.
- **Speculative Decoding: Performance or Illusion?** [2601.11580](https://arxiv.org/abs/2601.11580) (2025-12-31): benchmarking hygiene for spec-decode claims (overfitting to repetition-heavy evals inflates acceptance).

---

## 3. Hybrid linear-attention (Mamba/GDN) decode optimizations

(Beyond DeltaLog/DAMP above)
- **KVBuffer** [2605.19049](https://arxiv.org/abs/2605.19049) (2026-05-18): IO-aware serving — buffer recent K/V and batch state updates instead of per-token state rewrites; targets exactly the "state much larger than per-token K/V" problem. Port M. Composes with MTP: yes.
- **Tail-Replay** [2608.30310](https://arxiv.org/abs/2608.30310) (2026-08-31): prefix caching for hybrids without state checkpoints — replay only the tail from nearest state; raises cache-hit flexibility for multi-turn (C4/C8 agent reuse). Port M. [Source](https://arxiv.org/html/2608.30310v1)
- **DASC** [2608.30386](https://arxiv.org/abs/2608.30386) (2026-08-31): decay-aware compression of stored GDN/KDA state checkpoints (per-head/channel retention timescales) — less checkpoint memory → fewer evictions. Port M.
- **GDN-on-Blackwell tech report (MSInfer/FlashInfer contest)** [2607.16831](https://arxiv.org/abs/2607.16831) (2026-07-18): **1.58x official speedup on GDN decode+prefill on B200** (decode ~9.3µs/layer); documents that kernel-body codegen alone wasn't the win — graph/launch/layout integration mattered. Directly transferable direction for the GB10 GDN kernels. Port M. [Source](https://arxiv.org/html/2607.16831v1)
- **MatMul-only inversion for quantized GDN** [2606.06034](https://arxiv.org/abs/2606.06034) (2026-06-04): truncated-Neumann parallel inverse for chunked GDN — prefill-side mostly. Port M.
- **Component-aware self-speculative decoding in hybrids** [2605.01106](https://arxiv.org/abs/2605.01106) (2026-05-01): uses the SSM/linear-attention subgraph itself as a free internal drafter (Falcon-H1, and hybrid families) — zero-extra-parameter drafting for hybrid architectures. Lossless. Port M. Composes with MTP: alternative drafter.
- **Qwen3.8-Next architecture report** [2608.30320](https://arxiv.org/abs/2608.30320) (2026-08-31): confirms QSA replaces full attention in backbone+MTP; useful ground truth for kernel planning. [Source](https://arxiv.org/html/2608.30320v1)

---

## 4. lm_head / vocab-parallel GEMV for ~150k vocabs

- **HNSW output embeddings** [2608.27460](https://arxiv.org/abs/2608.27460) (2026-07-01): replace dense vocab GEMV+top-k with MIPS over an HNSW index of token embeddings; scatter retrieved logits into sparse full-vocab tensor. **Batch-1 output-projection speedup >12x on Gemma-256k-vocab** (CPU-measured; gains scale with vocab size — Qwen 152k, Llama 128k). **Lossy-adjacent** (recall/latency trade at ef settings; ef=200 default near-loss). Port M; verify sampling-path equivalence for temperature>0 before trusting zero-loss claim. [Source](https://arxiv.org/html/2608.27460v1)
- **CSV-Decode** [2511.21702](https://arxiv.org/abs/2511.21702) (2025-11-16): offline clustering with centroid+radius geometric upper bounds → per-step sub-vocabulary with **certified exact top-k and ε-certified softmax**; fallback on bound failure. 2–3x on the output layer, peak 4.43x in DSE. **Lossless-by-certificate** — the only lm_head trick in this survey with a hard guarantee. Port M. [Source](https://arxiv.org/html/2511.21702v1)
- **VQ-Logits** [2505.10202](https://arxiv.org/abs/2505.10202) (2025-05-15): VQ-compress the output layer (1% of params); quality-degrading (lossy) → **rejected under zero-quality-loss**.
- Note on INT8 lm_head: **"Spec Sheets Are Not Kernels"** [2608.11693](https://arxiv.org/abs/2608.11693) (2026-08-12) audits INT8 tensor-core withdrawal on Blackwell Ultra through PTX/CUTLASS/vLLM/SGLang — INT8 paths on Blackwell-family chips are software-emulated/absent; weigh their planned "INT8 lm_head + BF16 top-64 rescore" accordingly (sm_121 has no fifth-gen INT8 tensor path either).

---

## 5. MoE token-dispatch & grouped-GEMM on Blackwell sm_12x

- **TMA-Adaptive FP8 Grouped GEMM** [2508.16584](https://arxiv.org/abs/2508.16584) (2025-08-07): eliminates 128-alignment padding via TMA descriptor pool + dual-phase load/store; direct fit for 10-of-512 routing where per-expert token counts are ragged. Hopper-measured; TMA present on sm_120/121. Port M.
- **MonoMoE** (§2.1): the decode-batch-1 answer; persistent weight-major scheduling.
- **SonicMoE** (§2.2): IO-aware dispatch fusion + token-rounding; measured on the same 10/512 topology.
- **TEMPO** [2608.13057](https://arxiv.org/abs/2608.13057) (2026-08-13): expert-parallel load balancing across the memory-bound (<~160 tok/expert: cost attaches to *activated replicas*) vs compute-bound regimes; datacenter EP focus but the max-affine cost model is exactly right for reasoning about verification-batch expert-union costs. Port S (model) → informs EVICT/EcoSpec tuning.
- **CuTile evaluation** [2604.23466](https://arxiv.org/abs/2604.23466) (2026-04-25): first independent eval of CUDA Tile vs cuBLAS/Triton/W MMA on H100/B200/RTX PRO 6000 Blackwell — practical guidance for writing the missing sm_121 skinny-GEMM kernels to replace the Ampere-era cuBLAS wmma path (their 73% bottleneck). Port M (rewrite dense GEMMs).
- **Blackwell GDN native backward / kernel-verifier** [2608.12700](https://arxiv.org/abs/2608.12700) (2026-08-13): contract-grade verification of LLM-generated Blackwell kernels + native GDN backward — evidence that hand/kernel-gen sm_12x GDN kernels are tractable. Port reference.

---

## 6. LPDDR5x / unified-memory inference

- **LP-Spec** [2508.07227](https://arxiv.org/abs/2508.07227) (2025-08-10): LPDDR-PIM + spec-decode co-design for mobile — quantifies the core tension we face: speculation makes decode *more* GEMM/GEMV-heavy per step, which is wrong for GEMV-accelerated low-bandwidth memory unless token management is co-designed. Concept-relevant.
- **THInfer** [2605.25655](https://arxiv.org/abs/2605.25655) (2026-05-25): bandwidth-aware framework for a many-core with limited main-memory bandwidth (MT-3000/Tianhe) — locality-maximizing layout/scheduling under a bandwidth ceiling; the closest software-side analogue to GB10's 273 GB/s constraint. Port M (ideas).
- **AHASD** [2604.25326](https://arxiv.org/abs/2604.25326) (2026-04-28): asynchronous heterogeneous (CPU+NPU) adaptive drafting on mobile — draft-on-CPU/verify-on-accelerator overlap; relevant to GB10's big Grace CPU cores sitting idle during GPU verify. Port M.
- **Bole** (§1.1) — the only paper with **native GB10 measurements** (273 GB/s LPDDR5x): shows tree-verification state materialization is 86% of serial verify time on GB10 vs 38% on A100 — i.e., **GB10 punishes state traffic far more than HBM parts**, and fixing it yields proportionally larger wins there.
- **Catmux-style flash staging / CATS** (§1.16): SSD/flash-staged weights (their PLE mmap pattern) formalized for speculative settings.
- **VitaLLM** [2604.27396](https://arxiv.org/abs/2604.27396) (2026-04-30) & **AccLLM** [2505.03745](https://arxiv.org/abs/2505.03745): edge accelerator co-designs; background.

---

## TOP 5 TO TRY ON GB10 (ranked)

Target: 50 → 80 tok/s single-stream (1.6x) at zero quality loss. Current split: 73% dense BF16 GEMM, 18.7% MoE NVFP4 grouped GEMM, acceptance 2.95 @ k=3, bus 180–200/273 GB/s.

**1. Bole-style factorized GDN tree verification + budget profiler ([2608.01651](https://arxiv.org/abs/2608.01651)) — port M–L.**
The only technique in this survey *measured on GB10*, on the Qwen3-5 GDN-hybrid family, using the **native MTP head as drafter** — i.e., our exact architecture class and spec-decode mode. On GB10, state materialization eats 86% of serial verification time (vs 38% on A100): unified LPDDR5x punishes state traffic hardest, so removing it pays most on our chip. Peak 4.x speedups vs strong speculative baselines; the offline profiler also answers "what verification width can 273 GB/s absorb" (256-tree-tokens on GB10 vs 128 on A100 — a directly reusable calibration). Lossless. Composes with MTP natively.

**2. NVFP4-extend the dense/GDN half, Minima-style ([2609.04098](https://arxiv.org/abs/2609.04098)) + CuTile/sm_121 skinny-GEMM rewrite ([2604.23466](https://arxiv.org/abs/2604.23466)) — port L.**
The 73% dense-BF16-GEMM share is a pure byte problem while the bus runs at 66–73% of peak: W4A4-ing the BF16 GEMMs quarters that traffic ≈ up to ~1.9x model-level ceiling. Minima proves GDN recurrence + gates survive NVFP4 (matches BF16 across PPL/benchs to 64K on a 48-GDN-layer hybrid), removing the "recurrence needs 8-bit" objection. Pair with a real sm_121 skinny-GEMM/persistent kernel (CuTile guidance; MonoMoE's persistent-scheduling lesson) to also kill the Ampere-era cuBLAS wmma inefficiency — GB10 tensor cores are idle anyway at batch-1. Near-lossless PTQ (validate with their benchmark battery); composes with MTP.

**3. EVICT adaptive verification ([2605.00342](https://arxiv.org/abs/2605.00342)) + Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)) — port S–M.**
Two cheap, training-free, lossless scheduler-level fixes that compound with #1. EVICT truncates the draft tree to its cost-effective prefix before verification — on a 512-expert top-10 target the expert-union blowup during verify is second-order expensive (1.35x over AR / 1.21x over EAGLE-3 measured, CUDA-graph compatible). Windowed-MTP caps the MTP head's full-context KV reads (at long ctx the draft phase alone adds +92–138% per step, and it "sharpens under hybrid/linear-attention targets") — this is the C4/C8 stability lever, and it never regresses end-to-end. Both are drafter-side/verifier-scheduling changes: no kernels, no quality risk.

**4. MonoMoE persistent MoE megakernel ([2609.04244](https://arxiv.org/abs/2609.04244)) — port M.**
Attacks the 18.7% MoE share with up to 1.84x on the routed-MoE operator and 18% end-to-end TPOT at batch 1–8 — our regime. Already vLLM-integrated (fork-friendly), and its diagnosis (inter-kernel idle even under CUDA graphs; tile padding when experts see 1–2 tokens) matches a 10-of-512 model exactly. Needs an NVFP4 + sm_121 port of the weight-major persistent schedule. Lossless; composes with MTP (verification batches hit the same grouped-GEMM path).

**5. SlimSpec low-rank draft LM-head ([2605.10453](https://arxiv.org/abs/2605.10453)) or NanoSpec in-context vocab ([2605.26444](https://arxiv.org/abs/2605.26444)) — port M / S–M.**
The MTP head's 47k-vocab projection is paid 3× per draft step (k=3) at batch-1 GEMV rates. SlimSpec: 4–5x draft-head latency cut at near-identical acceptance, +8–9% end-to-end over the EAGLE-3 pipeline — a cleaner replacement for the planned draft-vocab reduction (which only gets ~60% head latency per their comparison). NanoSpec is the training-free alternative (<3k active vocab, −51.6% draft latency, 1.32x e2e over EAGLE-2). Small, isolated, composes with everything above.

**Why this ordering:** #1 and #2 attack the two largest cost centers (verification-state traffic on a chip that punishes it most; dense-GEMM bytes) with measured evidence on our exact architecture class or precision format. #3 is near-free and de-risks C4/C8. #4 is the MoE-share follow-on once dense bytes shrink. #5 is a small constant that multiplies everything. If only one thing is ported this quarter: **Bole**, because GB10-native numbers remove port-risk uncertainty.

**Explicitly not recommended** (quality or fit): VQ-Logits (lossy), HNSW lm_head unless certified-mode added (lossy-adjacent), relax-verify (already rejected), cross-layer sharing via KernelDNA/ShareYourAttention (training-time levers), SPD/JetSpec (competes with rather than composes alongside the shipped MTP head; L-effort).

---

*File generated 2026-09-12 by arXiv survey (25 queries, 334 unique papers screened, 60 abstracts reviewed, 35 full texts extracted). Dense BF16 GEMM kernel guidance (DeepGEMM sm120, flashinfer autotune, KDA Triton QSA) was treated as already-known per task context and not re-surveyed.*
