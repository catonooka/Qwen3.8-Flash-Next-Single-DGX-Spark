# Profiled cost-reduction research — Qwen3.8-Flash-Next 105B on GB10 (research4)

**Deliverable A+B: 2025–2026 techniques + shipped code that attack the *measured* decode-time buckets, mountable as single-file Python patches into the running vLLM container.**

- Stack: DGX Spark GB10 (sm_121a, 128 GB unified LPDDR5x ~273 GB/s, 48 SMs, 24 MiB L2), vLLM `0.1.dev20073+g8e685d198` (flashinfer **0.6.17**+cu130, CUDA 13.0, torch with CuTe-DSL stack), Qwen3.8-Flash-Next 105B A48 MoE NVFP4 (36 GDN linear-attn + 12 QSA full-attn), MTP=3 spec decode greedy, C1 prose ~55 tok/s → goal 80.
- Measured cycle ~60 ms @C1: **~44% verify lm_head** [248320,2560] (INT8-screened), **~19% dense BF16 GEMMs** (HC mixer + MTP mixer), **~18.7% NVFP4 MoE** (flashinfer_cutlass), **~8% GDN state**, rest misc/CPU.
- Mount constraint: single `.py` files only, mountable over `/usr/local/lib/python3.12/dist-packages/vllm/`. No docker, no rebuild, no new wheels; kernels must be callable via torch / triton / flashinfer already in the image.
- Method: GitHub REST API + raw patches + NVIDIA blog/forum verification (web_search backend was returning junk; arXiv API rate-limited from this IP — every external claim below was fetched via curl and is URL-cited). Local grounding: image sources at `/home/nmt/kernel_dive/image_src/` (flashinfer 0.6.17 and vLLM nvidia_full/spec_decode trees), `~/flashnext-spark/research_kernel_findings.md`, `research4/{arxiv-techniques,community-github,cuda-system}.md`, `research4/tactics-121a-*.json` (live autotune cache dumps), `~/flashnext-spark/skinny/patch-skinny-gemm-tp1.py`.
- Every technique is scored **S / M / L** = mount as-is today / mount with adapter code + validation / needs porting work beyond a patch.

---

## Bucket 1 — verify lm_head [248320,2560], INT8-screened BF16, B≤64 (≈44% of cycle)

Current state (banked): per-row-INT8 screen via `torch._int_mm` + exact BF16 top-64 rescore, mounted in `files/logits_processor.py` (`VLLM_INT8_VERIFY_HEAD=1`), 5.12→3.05 ms GEMV + 0.03 rescore, +16.2% C1 stacked with the INT8 draft head. Certificated: BF16-top1 ∈ INT8-top64 at 100% of measured positions. Remaining headroom: the INT8 GEMV runs at ~208 GB/s effective on W bytes — 76% of the 273 GB/s bus — and the head is read 4×/cycle (3 draft + 1 verify, minus the draft-head INT8 already banked).

### 1.1 Triton one-program-per-vocab-row GEMV (b12x `bf16_vocab_projection`) — **S**
- **What**: single-row BF16 vocab projection where grid = vocab rows, each program streams its weight row once (`BLOCK_K=next_pow2(K)`, loop variant for large K), f32 accumulator. b12x's measured winner over cuBLAS for exactly the `(248320,2560)`-class shape: policy gate `bf16, M=1, K≤8192, N≥16384, CC∈{(12,0),(12,1)}`. Kernel is 50 lines of pure Triton — copy `b12x/gemm/bf16_vocab_projection/_kernel.py` into a mounted file verbatim.
- **Measured**: b12x docs (README + corpus quotes): their Triton row-kernel beats cuBLAS WMMA-class picks on this exact shape class on GB10; vladimir-voinea's Triton configs independently reach 157 (M=1)–230 GB/s (M=64) on sm_121a (`results/tune-*.log`, 256-expert W4A16 MoE microbench).
- **Dependency**: `import triton` — already in image (vLLM ships triton for QSA/GDN kernels). No flashinfer API, no cutlass.
- **Mount**: S — replace the `torch._int_mm` screen with two stacked Triton calls (INT8 rows can reuse the same skeleton with int8 loads + f32 dot) or apply to the *draft* head GEMV (M=1 strictly, which the draft loop is). Caveat: M must be 1 for the row kernel; verify is M≤64 → keep INT8 `torch._int_mm` for verify, use row-kernel for the 3 draft steps (M=1 each), or batch the 3 draft steps' GEMVs (same hidden states differ per step — no, hidden differs → run 3× M=1 or pad-merge).
- **Gain estimate**: draft-head GEMV currently ~208 GB/s effective; Triton row GEMV at ~230–250 GB/s → ~10–20% of the draft-head slice (~3–6% of cycle). Modest but near-free.
- **URLs**: https://github.com/local-inference-lab/b12x (`b12x/gemm/bf16_vocab_projection/_kernel.py`, master branch); corpus: `~/mia-lab-research/corpus/en_code.txt` §76382–76384.

### 1.2 NVFP4-screen the lm_head (weight-only FP4, BF16 activations) via `flashinfer.mm_bf16_fp4` — **M**
- **What**: image's flashinfer 0.6.17 ships `mm_bf16_fp4(a_bf16, b_fp4, b_descale, alpha, backend="cute-dsl")` — W4A16 NVFP4 GEMM **explicitly supported on [100,103,110,120,121]** (verified in image: `flashinfer/gemm/gemm_bf16_fp4.py:60,94` `@supported_compute_capability`, kernels `dense_blockscaled_gemm_sm120_b12x.py`). Weight bytes drop 4× vs BF16, 2× vs INT8. Screen in NVFP4, rescore top-K in BF16 (same two-stage protocol as the banked INT8 screen — certified-exact if the certificate holds; must re-measure recall@K on the real tensor).
- **Measured**: FlashInfer release notes: W4A16 vs W4A4 = **1.50× at one token** on B200 at a DeepSeek EP8 shape (memory-bound regime — our regime). b12x measured NVFP4 M=1 on (248320,2560) at **1605 µs vs quantized-wrapper 1741 µs** on GB10 — but their A16 route was *not* promoted for M=1 vs their own quantized baselines; treat GB10-specific numbers as unproven for this exact shape and bench first (30 min with `int8_head_bench.py` adapted).
- **Dependency**: `flashinfer.mm_bf16_fp4` + `flashinfer.prepare_bf16_fp4_weights` — pure Python calls into JIT CuTe-DSL kernels already in the image; one-time weight prep at first load (~1.2 GB FP4 copy + swizzle).
- **Mount**: M — needs a recall-certificate re-measure (BF16-top1 ∈ NVFP4-topK at K=64/128 on real hidden states; NVFP4 quantization of the head is coarser than INT8 → K may need to rise to 128–256, partially eating the byte win), quality gates, and lazy-build-at-first-eager-call discipline (JIT compile must happen pre-graph-capture; same pattern as the INT8 head patch already ships).
- **Gain estimate**: verify-head slice ~26 ms/cycle at INT8 (0.63 GB read) → NVFP4 ~0.32 GB; if the kernel hits ~200 GB/s payload → ~1.6 ms; saves ~2–4 ms/cycle → +3–5% C1. Risk: recall drift changes greedy picks (quality risk, gated by certificate).
- **URLs**: image `flashinfer/gemm/gemm_bf16_fp4.py`; https://flashinfer.ai/releases (W4A16 1.50× at one token, B200); https://github.com/vllm-project/vllm/pull/56535 (b12x W4A16 MoE wrapper landing upstream — same kernel family).

### 1.3 draft-vocab restriction (F4a top-K draft sampler) — **S** (already built locally)
- **What**: restrict draft sampling to top-K of the previous verify-step logits (draft reads ~4–8k head rows instead of 65k/248k). Lossless by construction (verify unchanged; greedy verify makes draft choice non-load-bearing).
- **Measured**: local `f4a/` patch + campaign math: raises bandwidth ceiling 58→96 tok/s class. NanoSpec (arXiv 2605.26444) independently: <3k active vocab, −51.6% draft latency, 1.32× e2e over EAGLE-2.
- **URLs**: local `~/flashnext-spark/f4a/`; https://arxiv.org/abs/2605.26444; https://arxiv.org/abs/2605.10453 (SlimSpec low-rank draft head, 4–5× draft-head cut, +8–9% e2e).

---

## Bucket 2 — dense BF16 GEMMs [B·K,2560]×[2560,2560-ish] small-N (HC mixer, MTP mixer) (≈19%)

The image's own `nvidia_full/low_latency_gemm.py` ships `QWEN38NEXT_GEMM_PLANS` — NVIDIA's CuTe-DSL skinny-GEMM dispatch table for exactly this shape family — but it is gated `_is_sm103()` and keyed on TP=4 shapes, so on our TP=1 sm_121 box everything falls back to `F.linear` → cuBLAS → Ampere-era `cutlass_80_wmma` (55% of decode GPU time in the dolf3131 profile). The skinny path was measured +8.6% C1 here before being reverted, and `skinny/patch-skinny-gemm-tp1.py` is the proven single-file mount.

### 2.1 Re-mount skinny-GEMM gate-relax + TP=1 plan table — **S**
- **What**: `Qwen38NextLowLatencyLinearMethod` routes eligible BF16 linears through `shape_dynamic_skinny_gemm` (CuTe-DSL, in-image at `vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py`). Patch relaxes `_is_sm103()` → include `(12,1)` and adds TP=1 plan entries measured against cuBLAS on the real shapes — notably HC down/inject `(320,10240)` M=1 at **2.20×**, M=3 **1.92×** (97 calls/fwd); GDN QKVZ `(10240,2560)` 1.43×; QSA fused `(6144,2560)` 1.23×.
- **Measured**: +8.6% A/B on this box (CAMPAIGN_NOTES); per-shape wins 1.05–2.20×; (640,2560) M=1 measured 0.87× — deliberately left to cuBLAS.
- **Dependency**: in-image CuTe-DSL kernel + `torch.ops.vllm.qwen3_8_flash_next_low_latency_gemm` custom op; patch is pure Python editing the plans dict.
- **Mount**: S — file exists, was A/B'd, reverted only per campaign discipline. Re-apply, extend plans to the *verify* M=4 batch shape and to HC `(10240,320)` up-proj if the sweep finds wins.
- **URLs**: local `~/flashnext-spark/skinny/patch-skinny-gemm-tp1.py`; upstream gate: `vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py` (image copy at `/home/nmt/kernel_dive/image_src/nvidia_full/low_latency_gemm.py`); context https://github.com/vllm-project/vllm/pull/54048 (router cuBLAS out_dtype on family-120 — the upstream landing zone for the same class of gate-relax).

### 2.2 b12x `bf16_gemv` CuTe-DSL small-N GEMV — **S–M**
- **What**: one-CTA-per-output-column GEMV, 128 threads strided over K, 128-bit loads, shift/mask BF16 unpack with zero convert instructions, warp+block reduce, f32 acc. Target: the tiny-N tail — HC mixer `(320,10240)`-class and QSA indexer `(640,2560)`-class, ×97–108 calls/forward — where cuBLAS burns ~28 µs on a 16×16 WMMA tile for ~1 MB of real traffic.
- **Measured**: b12x on GB10: **2.3–3.9 µs vs cuBLAS 4–19 µs** at m≤4 on N=64..256/K=5120 (graph-replay timed); "~10× per-call on small-N skinny; cap at m=8" (loses ~2× at m=16). On 48-layer GDN stacks this was worth ~1.35 ms/step.
- **Dependency**: single Triton/CuTe kernel file, no flashinfer call — inline into a mounted module.
- **Mount**: S–M — kernel is self-contained; needs a dispatch wrapper on our shapes + numerics check vs F.linear + graph-capture-safe warmup. 
- **URL**: https://github.com/local-inference-lab/b12x (`b12x/gemm/bf16_gemv/_kernel.py`).

### 2.3 vLLM #55180-style FP8 raster swizzle — prefill-only here; **skip for decode**. https://github.com/vllm-project/vllm/pull/55180

---

## Bucket 3 — NVFP4 MoE decode, 10-of-512, B·K≤32 (≈18.7%)

Current kernel: `FLASHINFER_CUTLASS` SM120 TMA grouped GEMM (SM89-style MMA, not tcgen05). The image *also* ships the B12X Cute-DSL SM12x-native family (`flashinfer/fused_moe/cute_dsl/blackwell_sm12x/` + `b12x_moe.py` wrapper, gated `is_device_capability_family(120)` ✓) with a decode-optimized dispatch ladder: **direct_micro ≤32 routed pairs → micro ≤20/40 pairs (≤8 tokens) → static ≤640 pairs → dynamic**, with MAC ladders tuned on GB10 (`moe_dispatch.py:71–91`, comment "Measured on GB10"). Our decode shape is exactly 4 tokens × top-10 = 40 routed pairs → micro/static territory; MTP verify 4-token batch = 40 pairs; draft steps are 1×10 = 10 pairs → direct_micro.

### 3.1 Opt into FLASHINFER_B12X MoE backend — **S**
- **What**: `moe_backend="flashinfer_b12x"` (or mount a one-line oracle patch putting B12X first on family-120). The image's oracle comment says B12X is "intentionally excluded until the upstream CUTLASS SM121 MMA op guard is resolved" — but the known failure mode (TensorRT-LLM issue #15853, `_mma.block_scale` vs `mxf4nvf4` on CUDA-12 CuTe payload) does not apply: **this image reports `DSLCudaVersion(13,3)`**, the fixed configuration.
- **Measured**: (a) flashinfer release notes: SM12x fused-MoE synced to current b12x with **FP4 accuracy fixes** and "shape-stable route packing … recompile-free" decode; NVFP4 W4A4 backend at "kernel parity across decode and prefill". (b) vLLM #54788 validated flashinfer_b12x target + flashinfer_cutlass draft on RTX PRO 6000 (Qwen3.6-35B-A3B-NVFP4 + MTP-3) — our exact quant+spec class. (c) vLLM-Moet: 2-bit experts + FP4 delta on SM120 SASS reaching 105 tok/s GLM-5.2 753B TP4 — SM12x-class MoE decode kernels are production-viable.
- **Dependency**: zero new code for the flag path (`EXTRA_VLLM_ARGS`); kernel JIT-compiles on first use (persist `FLASHINFER_WORKSPACE`/JIT cache volume to avoid recompile per boot).
- **Mount**: S (flag) / M (oracle patch + validation). Gate on 11/11 reasoning suite + gibberish smoke; watch first-launch JIT (~minutes) inside warmup.
- **Gain estimate**: MoE slice ~11 ms/cycle at CUTLASS SM120-TMA; B12X direct_micro/micro are CUDA-core/HFMA2-class kernels designed for exactly ≤32 routed pairs — realistic 1.5–2.5 ms savings → +2–4% C1, plus C4/C8 robustness (shape-stable packing, no per-batch recompile).
- **URLs**: image `flashinfer/fused_moe/cute_dsl/{b12x_moe.py,blackwell_sm12x/moe_dispatch.py}`; https://flashinfer.ai/releases ; https://github.com/vllm-project/vllm/pull/54788 ; https://github.com/NVIDIA/TensorRT-LLM/issues/15853 ; https://github.com/kacper-daftcode/vLLM-Moet .

### 3.2 W4A16 (NVFP4 weights, BF16 activations) expert path — **M**
- **What**: B12x MoE wrapper `quant_mode="w4a16"` decodes FP4 weights to BF16 inside the kernel — no activation-quant launch, no repack per step. Release notes: W4A16 = **1.50× vs W4A4 at one token** on B200 (DeepSeek EP8 shape) because decode is memory-bound and skipping activation quant removes a full kernel + intermediate.
- **Trade**: *changes numerics* vs the checkpoint's W4A4 semantics (activations no longer FP4-quantized) — output should be *more* accurate, but it deviates from the NVFP4 checkpoint contract; must re-run quality gates. ModelOpt checkpoint scales pass through `input_global_scale` (0.6.17 has it).
- **Mount**: M — new quant-mode plumbing in the vLLM b12x experts adapter (upstream template: vLLM #56535 test + adapter code, copy-paste-able).
- **URLs**: https://github.com/vllm-project/vllm/pull/56535 ; https://flashinfer.ai/releases .

### 3.3 Offline tactic table for the *current* CUTLASS path — **S** (zero code)
- **What**: vLLM's `--enable-flashinfer-autotune` is already on and tunes mxfp8_gemm + trtllm MoE buckets, but our standing cache (162 entries) shows `trtllm::fused_moe::gemm1/2` tuned only at the 21-bucket set — and the corpus tick-6 finding stands: MXFP8 buckets run untuned tactic-0 at M≤4 in some configs. Dump/extend the cache: `AutoTuner.get().save_configs/load_configs`, or `python3 -m flashinfer generate-tactics-blocklist` to prune crashing tactics. Zero-kernel-code lever; retune happens at next boot.
- **URLs**: image `flashinfer/autotuner/autotuner.py`; research4/cuda-system.md §3; live dumps `research4/tactics-121a-*.json`.

---

## Bucket 4 — GDN state materialization/read-write per decode step, 36 layers, ~16 KB/layer/request (≈8%)

### 4.1 Fused GDN decode kernel (in-image, already default-on) — **S** (verify it's active; then nothing to do)
- **What**: image env `VLLM_GDN_DECODE_KERNEL="cuda"` default → `torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed` fuses conv-update + recurrent state advance + gated-RMSNorm per layer in one launch. Upstream PR #51674 (NVIDIA, merged): fused post-conv MTP decode kernel supporting BF16 **and FP32 state** and spec-decode intermediate-state scatter (`ssm_state_indices` per draft token — exactly our rollback pattern); #53463 extends it to non-spec decode: **per-layer eager cost 127→12 µs (10.6×, L40S, B≤8); +15.3% output throughput / −13.6% TPOT on Qwen3.8-27B TP2 enforce-eager**; neutral under FULL graphs.
- **Check**: our stack runs FULL_DECODE_ONLY graphs (target) + **eager drafter** → the eager-side 10× applies to the *draft* path if the drafter's GDN goes through the fused op. Verify with one profiler trace that `qwen_gdn_attention_core_fused_norm_packed` appears in draft steps; if the drafter falls back to Triton packed recurrent + separate RMSNorm, mount the #53463 dispatch (pure-Python change to `qwen_gdn_linear_attn.py`).
- **URLs**: https://github.com/vllm-project/vllm/pull/51674 ; https://github.com/vllm-project/vllm/pull/53463 ; image `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:512–560,900–935` (blob `bbbb1c8…`, fetched from tree.json).

### 4.2 FlashInfer GDN decode/MTP CuTe-DSL kernels (in-image, unused by vLLM) — **M**
- **What**: the image's flashinfer 0.6.17 ships a full GDN decode family the vLLM path never calls: `flashinfer.gdn_decode.gated_delta_rule_decode` (K-major state, T=1), `..._pretranspose` (V-major, BF16 fast path incl. T>1 MTP dispatch), and `gated_delta_rule_mtp` with `intermediate_states_buffer`/`ssm_state_indices` for spec-decode state caching — kernel docstring: "v15 dispatch: inline for BS≤2, warp-specialized for BS≥3; BS=8–16 tile_v=32 ilp=4 for ALL T, eliminating the ilp=1 fallback at T≥4 that caused a ~33% per-step slowdown".
- **Measured**: MSInfer GDN-on-Blackwell tech report: **1.58× official speedup on GDN decode+prefill on B200** (decode ~9.3 µs/layer); the win came from graph/launch/layout integration, not kernel-body codegen.
- **Dependency**: `import flashinfer; flashinfer.gated_delta_rule_decode(...)` — pure Python; state pool must be K-major or V-major contiguous per kernel variant.
- **Mount**: M — write an adapter in the mounted `qwen_gdn_linear_attn.py` to call flashinfer's kernel for decode/MTP instead of the vLLM custom op; numerics-equal validation vs the Triton path; state-layout match required (our state pool layout: check `get_gdn_mamba_state_shape_from_config`).
- **URLs**: image `flashinfer/gdn_decode.py`, `flashinfer/gdn_kernels/{gdn_decode_mtp,gdn_decode_bf16_state,gdn_decode_pretranspose}.py`; https://arxiv.org/abs/2607.16831 (MSInfer 1.58×).

### 4.3 DeltaLog-style deferred state materialization — **M–L**
- **What**: eager GDN decode writes full state every token (32→44% of decode latency at B=64 on Qwen3.5-class); DeltaLog keeps a dense base + per-token delta factors, lazily materialized. State-update kernel **1.69× (H200) / 1.86× (4090)**; up to ~1.20× one-token decode.
- **Fit**: our B≤8 decode is far from the B=64 write-amplification regime; at 8% of cycle the ceiling is small (~1 ms/cycle). Ranked below 4.1/4.2.
- **URL**: https://arxiv.org/abs/2608.15533 . Related: DAMP INT8 states https://arxiv.org/abs/2608.27513 (S–M; changes numerics — gate).

### 4.4 Bole factorized GDN tree verification — **L** (the big structural lever, not a patch)
- **What**: GB10-native measurements (Qwen3-5 GDN hybrids + native MTP heads = our class): **state materialization is 86% of serial verification time on GB10 vs 38% on A100**; value-tiled parallel verifier removes it; peak 4.x speedups; profiler picks verification budget (256 tree tokens on GB10).
- **URL**: https://arxiv.org/abs/2608.01651 . SGLang-side; needs factorized kernels ported into vLLM's MTP verify path — L, but the only surveyed technique with native GB10 numbers on our architecture class.

---

## Bucket 5 — CPU-side Python overhead between graph replays (draft loop is EAGER, 3 calls/cycle)

Facts: drafter gets `CUDAGraphMode.NONE` under FULL_DECODE_ONLY; `build_for_drafting` rebuilds attention metadata per draft step; QSA_STATE backend logs "Fused multi-step draft decode is not supported … falling back to rebuilding attention metadata between draft steps" (jschmied's box, same model). Estimate 3–8 ms/cycle pure launch+Python overhead across 3 eager draft steps.

### 5.1 FULL_AND_PIECEWISE cudagraph mode for the drafter — **S–M** (scripted, never run: F3 ladder)
- **What**: set `compilation-config mode≥1 + cudagraph_mode=FULL_AND_PIECEWISE` → drafter gets PIECEWISE graphs; fallback ladder: manual capture of the proposer loop at constant shapes (buffers already static: `self.input_ids`/`hidden_states`, llm_base_proposer.py:723–725).
- **Measured**: upstream #49891 (SM120 NVFP4 KV + **MTP cudagraph fix**): "Allow FULL cudagraph mode (not just PIECEWISE) for draft model … Align draft model capture with target model's cudagraph mode" — i.e. draft-model FULL graphs are now sanctioned upstream on SM12x. #53463's eager-cost numbers (127→12 µs/layer) bound what graph capture additionally saves once the fused kernel is active.
- **Risk**: piecewise compile must partition the QSA/hyperconnection custom ops; fails loudly at startup if not — safe A/B. Watch #53051 (prefill misdispatched into spec FULL graph → silent GDN state loss, open).
- **URLs**: https://github.com/vllm-project/vllm/pull/49891 ; local `research3/g14-vllm-specdec.md` (F3 ladder); start.sh:426 currently hardcodes mode:0.

### 5.2 Batch the 3 draft lm_head+argmax into the graph — **S**
- _greedy_sample does `compute_logits(...).argmax(-1)` per step (llm_base_proposer.py:440–450), each materializing full logits. If graphs capture the drafter (5.1), this is free; without graphs, mount a patch fusing the 3 steps' screen+argmax via the already-mounted INT8 head code path (draft head already INT8 + top-32 rescore). Local `files/mtp_patched_topk.py` is the template.
- **URL**: image `spec_decode/llm_base_proposer.py`; local `files/mtp_patched_topk.py`.

### 5.3 Kill per-step `build_for_drafting` metadata rebuilds — **M**
- **What**: metadata rebuild is pure Python per draft step per attn-group; cache the metadata object across the 3 steps (only positions/KV lens change → patch indices, don't rebuild). The QSA "fused multi-step unsupported" fallback is exactly this cost.
- **URL**: image `spec_decode/qwen3_8_flash_next.py:52–88` (`build_per_group_and_layer_attn_metadata`), `spec_decode/llm_base_proposer.py:1011`.

### 5.4 GB10 fast/slow flip + cpuidle hygiene — **S** (ops, not code)
- The 3.3× GEMV fast/slow flip on our exact kernel/driver (tonyd2wild issue #1) and LPI-1 disable on X925 cores are documented in research4/cuda-system.md §1–2 — they protect C4/C8 stability rather than lift C1; costs nothing to apply alongside.
- **URL**: https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1 .

---

## B) Top-5 action list (expected tok/s gain × feasibility, effort)

Baseline 55.8 tok/s C1 prose; goal 80 (+43%). Gains below are per-action estimates on this stack; they do not stack linearly (shared bandwidth + acceptance coupling).

| # | Action | Bucket | Expected C1 gain | Effort | Why ranked here |
|---|--------|--------|------------------|--------|-----------------|
| **1** | **Re-mount skinny-GEMM gate-relax + TP=1 plans** (`skinny/patch-skinny-gemm-tp1.py`), extend sweep to M=4 verify shapes + HC up-proj `(10240,320)` | 2 | **+8–10%** (measured +8.6% before; HC mixer is the top dense target at 2.20×/1.92×) | **S** (2–4 h: re-apply, sweep 2 shapes, A/B, gates) | Already built, already A/B'd on this box, zero quality risk (numerics = F.linear class), single file. Highest certainty-per-hour on the board. |
| **2** | **FlashInfer B12X MoE backend opt-in** (`--moe-backend flashinfer_b12x` via EXTRA_VLLM_ARGS, then oracle-mount if it holds) | 3 | **+2–4%** C1, +C4/C8 stability (shape-stable packing, GB10-tuned MAC ladder, direct_micro ≤32 pairs matches our 40-pair verify exactly) | **S–M** (0.5 day A/B + 11/11 gates; JIT warmup discipline) | Zero-code first step; kernel family in-image with FP4-accuracy fixes; validated upstream on our quant+MTP class (#54788). The CUTLASS SM120-TMA path we run was never sm121-tuned. |
| **3** | **Drafter graph capture: FULL_AND_PIECEWISE ladder** (mode:1 + cudagraph FULL_AND_PIECE; fallback manual proposer capture; verify `qwen_gdn_attention_core_fused_norm_packed` in draft steps per #53463) | 5 | **+5–9%** (3–8 ms of 60 ms cycle is launch/Python; upstream now sanctions draft FULL graphs on SM12x #49891; #53463 shows 10× eager per-layer cost on the GDN op the drafter runs) | **S–M** (1 day incl. fallback ladder + state-loss check #53051) | Attacks the eager-draft overhead directly; failure modes are loud (startup) not silent; no quality risk. |
| **4** | **Draft-head Triton row-GEMV + F4a top-K draft vocab** (b12x `bf16_vocab_projection` kernel inlined for the 3× M=1 draft GEMVs; then F4a restriction for the byte win) | 1 | **+3–6%** combined (row-GEMV ~10–20% of draft-head slice; F4a raises the ceiling 58→96 class and stacks with INT8 heads) | **S** (0.5–1 day; F4a patch exists locally) | Both pieces exist as code; lossless by construction (verify untouched); acceptance is the only coupling. |
| **5** | **GDN decode via flashinfer CuTe-DSL kernels or verify-fused-op coverage** (adapter calling `flashinfer.gated_delta_rule_decode[_pretranspose]`/`gdn_decode_mtp` with `ssm_state_indices`; or confirm #51674/#53463 fused op covers the drafter) | 4 | **+2–4%** (8% bucket; 1.58× GDN-decode kernel evidence on B200; v15 dispatch fixes the T≥4 33% slowdown our MTP T=4 hits) | **M** (2–3 days: state-layout adapter + numerics validation) | In-image kernels unused by the vLLM path; the MTP variant matches our per-draft-token state-scatter exactly; bounded but real. |

**Order of execution**: 1 → 2 → 3 in the same campaign window (all S-class mounts, independent buckets); re-baseline after each. 4 and 5 next; then the L-class structural moves (Bole factorized verify — the only GB10-native evidence, arXiv 2608.01651; NVFP4-extend dense/GDN Minima-style, arXiv 2609.04098) if the goal still isn't reached.

**Ceiling math** (from F10 decomposition, updated for banked INT8 heads): bandwidth floor ~34–36 ms/cycle → ~59–62 tok/s ceiling at current acceptance; measured 60 ms. Reaching 80 requires killing *both* overhead (~38 ms → ~15) and bytes (HC/dense GEMM efficiency, MoE kernel) — consistent with actions 1+2+3 attacking ~44% of the remaining gap and 4+5 another ~15–20%.

---

## Source index (all verified fetched 2026-09-12)

**PRs (vLLM)**: #51674 fused GDN MTP decode https://github.com/vllm-project/vllm/pull/51674 · #53463 non-spec fused GDN decode (+15.3% TPOT class) https://github.com/vllm-project/vllm/pull/53463 · #49891 SM120 NVFP4 KV + MTP cudagraph https://github.com/vllm-project/vllm/pull/49891 · #56535 b12x W4A16 MoE wrapper https://github.com/vllm-project/vllm/pull/56535 · #54788 draft moe_backend (b12x+MTP validated) https://github.com/vllm-project/vllm/pull/54788 · #54048 router cuBLAS family-120 https://github.com/vllm-project/vllm/pull/54048 · #53896 Qwen3.8-Flash-Next model support https://github.com/vllm-project/vllm/pull/53896 · #55180 FP8 raster swizzle (prefill) https://github.com/vllm-project/vllm/pull/55180
**PRs (SGLang / flashinfer / rtp-llm / TRT-LLM)**: sglang #38170 b12x default on SM120 (+15% decode, 63→89% HBM BW) https://github.com/sgl-project/sglang/pull/38170 · sglang #38685 tiny_gemm decode b/a https://github.com/sgl-project/sglang/pull/38685 · sglang #39126 Qwen3.8 NVFP4 on DGX Spark (file-backed PLE, PDL router fix) https://github.com/sgl-project/sglang/pull/39126 · flashinfer #5099 cuTile MXFP4+W4A16 https://github.com/flashinfer-ai/flashinfer/pull/5099 · #4495 B12x Direct low-token MoE https://github.com/flashinfer-ai/flashinfer/pull/4495 · #4910 W4A16 disk cache https://github.com/flashinfer-ai/flashinfer/pull/4910 · rtp-llm #1399 B12X NVFP4 on CUDA13 https://github.com/alibaba/rtp-llm/pull/1399 · TRT-LLM #15853 SM120 CuTe PTX bug https://github.com/NVIDIA/TensorRT-LLM/issues/15853 · TRT-LLM #18898 W4A16 FC2 tile SM120 https://github.com/NVIDIA/TensorRT-LLM/pull/18898
**Repos**: b12x (SM120/121 CuTe+Triton kernels; bf16_gemv, bf16_vocab_projection, GDN decode, micro MoE) https://github.com/local-inference-lab/b12x · vLLM-Moet (SM120 SASS 2-bit+FP4 MoE) https://github.com/kacper-daftcode/vLLM-Moet · jschmied qwen38-flash-next-gb10 profile https://github.com/jschmied/qwen38-flash-next-gb10 · dolf3131 recipe https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark · tonyd2wild 4×Spark TP4 https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (+ issue #1 fast/slow flip)
**Papers**: Bole 2608.01651 (GB10-native tree verify) · TreeWY 2608.20961 · SpecLA 2607.16673 · EVICT 2605.00342 · NanoSpec 2605.26444 · SlimSpec 2605.10453 · Windowed-MTP 2607.21535 · MonoMoE 2609.04244 · SonicMoE 2512.14080 · Minima 2609.04098 · DeltaLog 2608.15533 · DAMP 2608.27513 · MSInfer GDN-Blackwell 2607.16831 · CuTile 2604.23466 · all at https://arxiv.org/abs/<id> (full annotated list in research4/arxiv-techniques.md)
**Docs/blogs**: FlashInfer releases (W4A16 1.50× @1 tok; SM12x MoE FP4 fixes; GDN CuTe-DSL overhaul) https://flashinfer.ai/releases · NVIDIA DGX Spark optimizations blog (NVFP4 −40% mem, 2.6× Qwen-235B w/ spec decode) https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/ · vLLM CUDA graphs doc https://docs.vllm.ai/en/stable/design/cuda_graphs/
**Local evidence**: `~/flashnext-spark/{research_kernel_findings.md, CAMPAIGN_NOTES.md, CHANGELOG.md, skinny/patch-skinny-gemm-tp1.py, files/, f4a/, int8_head_bench.py, hc_bench.py}` · `~/flashnext-spark/research4/{arxiv-techniques.md, community-github.md, cuda-system.md, tactics-121a-*.json}` · image sources `/home/nmt/kernel_dive/image_src/` (flashinfer 0.6.17 tree; vLLM nvidia_full + spec_decode; GDN skinny-GEMM plans; b12x MoE dispatch ladders; GDN decode/MTP kernels)

---

## LOCAL STATUS CORRECTIONS (Hermes, 2026-09-12 10:45 — applied after on-box verification)

- Action #1 premise stale: the skinny-GEMM patch IS the standing image
  (`vllm-skinny-tp1:v1-spinfix`); it was never un-mounted. The remaining real
  slice of #1 is the PLAN EXTENSION only (HC up/down-proj shapes
  (10240,320)/(320,10240) + M=4 verify shapes), est +1-3% not +8-10%.
- Action #2 CLOSED on this box 2026-09-11: `--moe-backend flashinfer_b12x`
  crashes on sm_121 (b12x_test.sh + CHANGELOG). Do not re-run without an
  image bump carrying the fixed b12x family.
- Action #3 is the live one: drafter runs EAGER under FULL_DECODE_ONLY
  (verified in llm_base_proposer.py:429-436 — eagle_cudagraph_mode=NONE).
  Config-only probe queued.
