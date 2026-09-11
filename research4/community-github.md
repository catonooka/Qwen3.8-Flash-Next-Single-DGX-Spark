# Community + GitHub research: GB10/DGX Spark inference optimization & vLLM spec-decode (Aug–Sep 2026)

Scope: last 3 months, emphasis Aug–Sep 2026. Our stack: Qwen3.8-Flash-Next (~105–125B-class MoE, 6B active, NVFP4, 51B PLE n-gram table) on vLLM `0.1.dev20073+g8e685d198` (2026-08-18 base), single DGX Spark (GB10, sm_121a, 121.63 GiB, 273 GB/s), MTP k=3, ~50 tok/s solo / ~105 tok/s C4. Already merged locally: vLLM #50729, #53388, dolf3131 skinny-GEMM cuBLAS gate-relax, spin-wait fix. All research read-only; nothing local was touched.

Sources: GitHub REST API (issues/PRs/repos/commits), raw READMEs, NVIDIA developer forums (JSON API), vllm.ai blog. Every claim carries a URL. "Port" = S(<1 day)/M(days)/L(week+).

---

## 1. vLLM PRs MERGED after our 2026-08-18 branch point (rebase candidates)

**F1. #55375 — [Bugfix][Qwen4Exp] fix state index strides in fused PLE conv** — merged 2026-09-05.
With speculation configured, prefills after batch row 0 wrote their PLE conv state into request 0's checkpoint blocks (strided index view bug). Root-caused by jschmied/peakcrosser7 on GB10; MTP output corruption. This is a *correctness* fix for exactly our model+MTP configuration.
https://github.com/vllm-project/vllm/pull/55375 — Applies: YES (critical). Port: **S**.

**F2. #54048 — [Bugfix][MoE] Enable cuBLAS out_dtype router GEMM on all CUDA archs (family-120/GB10)** — merged 2026-08-30.
The bf16→fp32 router GEMM was gated Hopper+SM100 only; family-120 fell back to a copy kernel and bf16-rounded router logits. Now the plain cuBLAS out_dtype epilogue engages on sm_120/sm_121. This is the upstream landing zone for the same gate dolf3131 relaxed locally for us — merging it should let us *drop* our local skinny-GEMM gate-relax patch in favor of the official path.
https://github.com/vllm-project/vllm/pull/54048 — Applies: YES. Port: **S** (likely replaces a local patch).

**F3. #55715 — [Perf][GDN] Enable the FlashInfer GDN prefill kernel on SM12x** — merged 2026-09-08.
SM12x silently ran the Triton/FLA fallback for every linear-attention (GDN) layer — on Qwen3.5/3.6/3.8 that is **3 of every 4 layers**. Gate was stale; FlashInfer's SM120 CuTe-DSL delta-rule prefill kernel (flashinfer#3479, merged 2026-06-17) needed `head_k_dim==128` + CUDA ≥ 13. Kernel A/B measured on DGX Spark GB10 (vLLM 0.28.1rc1.dev388, FlashInfer 0.6.18). Prefill-side win, not decode.
https://github.com/vllm-project/vllm/pull/55715 — Applies: YES (prefill/TTFT). Port: **S** (needs FlashInfer ≥0.6.18 + CUDA 13; we're on cu130 already per dolf3131 image lineage).

**F4. #54110 — [Kernel] Fall back from persistent top-k on low-shared-memory GPUs** — merged 2026-09-05.
Oversubscribed `persistent_topk` + <128 KiB opt-in smem (GB10 class) previously *raised* and killed EngineCore during MTP-shaped decode. Now routes to `top_k_per_row_decode`, preserving per-step MTP layout. Directly protects our MTP k=3 decode path from a crash class.
https://github.com/vllm-project/vllm/pull/54110 — Applies: YES. Port: **S**.

**F5. #53945 (merged 2026-08-26) + #54713 (merged 2026-09-10) — EAGLE/MTP × Mamba-align prefix-cache replay boundaries.**
#53945 caches the Mamba state at the block-grid position of the EAGLE resume (fixes hit-flooring pinned by #52371). #54713 retains *both* replay boundaries so an identical block-aligned prompt resend still hits (otherwise reconciled hit collapses to 0). Sibling to our merged #50729/#53388 work.
https://github.com/vllm-project/vllm/pull/53945 , https://github.com/vllm-project/vllm/pull/54713 — Applies: YES (spec-decode + prefix-cache interplay on hybrid Mamba/GDN). Port: **S**.

**F6. #55450 — [Bugfix][Core] Retire Mamba states across null gaps** — merged 2026-09-11.
Align-mode Mamba retirement stopped at null gaps and leaked older states. Replay with the real GLM-5.3-Flash hybrid cache layout (3,584-token blocks, 2 in-flight chunks): peak retained blocks per Mamba group **71 → 9**, logical pool capacity held **2,006 MiB → 254 MiB**. Mechanism is generic to hybrid Mamba/GDN + MTP (GLM-5.3-Flash measured; DGX Spark CPU-only manager tests 3-fail→22-pass). Frees unified memory for KV → more concurrent long-context sessions.
https://github.com/vllm-project/vllm/pull/55450 — Applies: YES (same cache layout family as Flash-Next). Port: **S**.

**F7. #54788 — [Bugfix][Spec Decode] Honour the draft's moe_backend on Model Runner V2** — merged 2026-09-08.
Draft inherits target `VllmConfig`, so `--speculative-config {"moe_backend": ...}` was ignored on V2; quantized-target + unquantized-draft (the normal MTP shape) failed to start. Validated with Qwen3.6-35B-A3B-NVFP4 + MTP-3, `flashinfer_b12x` target / `flashinfer_cutlass` draft, RTX PRO 6000. Matters if/when we move to MRV2 or a b12x MoE backend.
https://github.com/vllm-project/vllm/pull/54788 — Applies: conditional (V2 / moe_backend). Port: **S**.

**F8. #55180 — [Kernel] SM 12.x blockwise FP8: swizzle CTA raster when weight exceeds L2** — merged 2026-09-07.
GB10's 24 MiB L2 cannot hold FP8 weight operands; default raster re-streams weights from DRAM at large M. Swizzled persistent scheduler (`max_swizzle_size=8` when weight > L2 and activations ≥ 14 MiB) recovers prefill GEMM throughput: e.g. 16384×2560 FP8 weight at M=16384: 52 → (near the M=4096 level of 165–170) TFLOPS; bit-identical results. Measured on GB10 with torch.cuda.Event. Prefill-time cost for every FP8-blockwise model on these parts.
https://github.com/vllm-project/vllm/pull/55180 — Applies: prefill only, and only for FP8-blockwise dense projections (we're NVFP4 experts) — but our dense/trunk projections use FP8 paths. Port: **S**.

**F9. #52816 — [Spec Decode] DFlash2: local convolution + candidate selector** — merged 2026-08-21 (3 days after our branch point).
New DFlash2 drafter architecture: grouped dynamic depthwise conv inside each block + a candidate-selector that walks the best path over the target head's top-K per slot (lossless verify at T>0 via inverse-CDF). This is the drafter the community is posting big single-stream wins with (F17, F18).
https://github.com/vllm-project/vllm/pull/52816 — Applies: potential MTP replacement. Port: **M–L** (needs a Flash-Next DFlash2 checkpoint; see F14 blockers).

**F10. #53896 (merged 2026-08-31, Qwen3.8-Flash-Next model support, Triton QSA by construction), #54427 (merged 2026-08-30, weight-only NVFP4 checkpoints routed to W4A16/Marlin — prevents silent all-experts-dead garbage from an uninitialized `input_scale` that manifested on GB10), #53574 (merged 2026-08-31, DSv4 C128A topk contiguity on SM120 — spec-decode verification batches cross the 64-token FlashInfer dispatch cutoff and crashed CUDA-graph capture on sm_121a), #53835 (merged 2026-09-05, fused GDN MTP decode built for SM110-family), #54306 (merged 2026-09-01, gate sm_100-only kernel tests on capability family not >=).**
https://github.com/vllm-project/vllm/pull/53896 , https://github.com/vllm-project/vllm/pull/54427 , https://github.com/vllm-project/vllm/pull/53574 , https://github.com/vllm-project/vllm/pull/53835 , https://github.com/vllm-project/vllm/pull/54306 — Applies: model-correctness set for our exact model; if our image predates 2026-08-31 it does not carry #53896. Port: **S** each.

## 2. vLLM OPEN PRs / issues to watch (high value for us)

**F11. #56273 — [Quantization] Support packed NVFP4 Qwen4Exp PLE embeddings** — open, 2026-09-10.
Packs the 47.7 GiB PLE n-gram table to **26.82 GiB resident NVFP4** (E2M1 + E4M3 block scales, decode-on-read of just the gathered rows) — eliminates CPU/disk PLE offload entirely. Validated on one GB10: model loads at 97.47 GiB, 4,096-token ctx, 2 GiB KV, MTP off. Fixes #56272.
https://github.com/vllm-project/vllm/pull/56273 — Applies: YES — the biggest structural memory lever pending upstream. Port: **M** (test with MTP k=3 + long ctx once CI-green).

**F12. #55557 — [Model] Qwen4Exp: fp8_e4m3 main KV cache on the QSA path** — open, 2026-09-06.
Enables `--kv-cache-dtype fp8` for QSA layers (side caches stay bf16). Halves per-token KV footprint. Corroborating GB10 evidence (jschmied, F16): FP8 KV is a *capacity* optimisation (KV pool ×1.72, 4.11→7.07 concurrent 262k requests), decode-neutral.
https://github.com/vllm-project/vllm/pull/55557 — Applies: YES for capacity/long-context, not for speed. Port: **M**.

**F13. #53504 — [Performance] MTP first repeat misses prefix cache on a hybrid Mamba/GDN model** — open issue, 2026-08-24.
Qwen3.8-hybrid + MTP: the *first* repeat of an identical prompt re-prefills in full; reuse starts only on the second repeat. Retention-policy mismatch between the EAGLE-adjusted reusable boundary and the sparse Mamba retention boundary. **Config workaround exists today: `--prefix-cache-retention-interval <block_size>`.**
https://github.com/vllm-project/vllm/issues/53504 — Applies: YES (we are exactly this model class + MTP). Port: **S** (flag now; fixes F5 mitigate over time).

**F14. #56088 — [Bug] qwen4_exp (Qwen3.8-Flash-Next) cannot serve DeepSpec DFlash/DSpark drafters: five blockers, patches available** — open issue, 2026-09-09.
Field says a DeepSpec-trained DFlash/DSpark checkpoint is a near-exact fit for vLLM's own loader; four small control-flow blockers + one architectural, with patches in the thread. This is the tracked path to a stronger drafter than in-checkpoint MTP for *our* model.
https://github.com/vllm-project/vllm/issues/56088 — Applies: YES (if we chase DFlash2/DSpark). Port: **L**.

**F15. Ops-hygiene pair: #41871 (stale Triton kernel cache on sm_121 silently garbles output — wipe `~/.triton/cache`) + #42859 (merged 2026-05-17; vLLM now warns on PTX fallback and `VLLM_FORCE_TRITON_CACHE_INVALIDATE=1` wipes the cache at import).** Also #49546 (open): `VLLM_MARLIN_INPUT_DTYPE=fp8` silently corrupts output on GB10/sm_121a — do not set it.
https://github.com/vllm-project/vllm/issues/41871 , https://github.com/vllm-project/vllm/pull/42859 , https://github.com/vllm-project/vllm/issues/49546 — Applies: YES, on every image bump. Port: **S**.

Also noted (not detailed): #55390 (open — positional annotation of MTP draft KV-cache groups on hybrid path; Qwen3.5-class `mtp_num_hidden_layers` models otherwise flag *Mamba* groups as draft groups), #55122 (open — deterministic `persistent_topk`, no measurable perf cost, fixes greedy forks), #55737 (open — FlashKDA for KDA chunked prefill, 1.7–3.8× vs Triton chunk path; GLM-5.3-Flash today but the same pattern that produced F3), #54919 (open — Flash-Next long-prefill starves decode 3–7 min on 2-node TP2; single-node less exposed but chunked-prefill scheduling is the lever), #56461 (DeepSeek-V4.1-Flash cannot serve on SM120/121). https://github.com/vllm-project/vllm/pull/55390 , https://github.com/vllm-project/vllm/pull/55122 , https://github.com/vllm-project/vllm/pull/55737 , https://github.com/vllm-project/vllm/issues/54919 , https://github.com/vllm-project/vllm/issues/56461

## 3. Speculative decoding — community measured results on GB10-class hardware

**F16. jschmied/qwen38-flash-next-gb10 — the deepest public single-box vLLM profile of our exact model** (12 ⭐, updated 2026-09-11).
Decode 17.1 → **36.5 tok/s** solo (checkpoint levers + MTP); ~100 / ~110 tok/s aggregate at 16/32 streams; TTFT 2.6 s @7.5k / 10.1 s @29k (≈2,800 tok/s prefill); warm agent turn 2.05 → **1.52 s** from one prefix-cache flag. Key spec-decode data: **MTP k=2: 26.4 → 38.0 tok/s (+44% at c=1) but only +2.6% aggregate at c=16 (speculation stops paying at saturation — experts are 20% of per-token bytes at c=1, ~80% at c=16); MTP still cut c=16 TTFT by 30%** → keep it on for agent work. Limit found: "Fused multi-step draft decode is not supported by attention backend QWEN38_FLASH_NEXT_EXP_QSA_STATE — falling back to rebuilding attention metadata between draft steps", a per-draft-step cost that grows with k and pins the optimum at low k. Also: 7 model-specific kernels JIT during inference (discard first requests before measuring). NVFP4 KV cache closed as a lever on GB10 (two independent measurements + structural MTP-acceptance penalty, fails silently). Hyper-connections (27% of decode GPU time) are latency-bound at ~78% of roofline — quantization/kernel work there measured null three ways (corroborates dolf3131's "GEMM tuning is saturated").
https://github.com/jschmied/qwen38-flash-next-gb10 (see notes/speculation-on-flash-next.md, notes/what-generalises.md, notes/closed-levers.md) — Applies: directly. Their 36.5 solo vs our ~50: we're ahead; their TTFT/prefix-cache findings still transfer. Port: n/a (read), flag-level items **S**.

**F17. 0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark — drafter comparison on GB10 (55 ⭐).**
Stock 7.88 tok/s → MTP k=3 17.99 (acceptance 3.165) → DSpark k=7 19.12 (2.875) → **DFlash2 k=7 31.72 tok/s, acceptance 4.607, 3.99× stock, 1.66× over DSpark k=7** (49.2 on edit-heavy). Constraints: DFlash2 under vLLM requires an **unquantized LM head**; advantage is single-stream only — at c2+ NVFP4+MTP overtakes (+23% at c16). Also: quantization advantage collapses with concurrency (+27% c1 → +0.2% c16, FP8 ties 4-bit).
https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark (DFLASH2.md) — Applies: pattern-level (27B model, not Flash-Next); motivates F9/F14 for us. Port for us: **L**.

**F18. AEON-7/vllm-ultimate-dgx-spark — sm_121a source-built vLLM 0.27.1 container fleet (141 ⭐, updated 2026-09-11).**
Cherry-picked DFlash2 (#52816) one day after their 0.27.1 base: **3.39× single-stream, 1.41× at 64-concurrent on Qwen3.8-27B**, "beating MTP and DSpark at every concurrency level" (their claim; 27B-class model). Also carries NVFP4/FP8 KV (PR #44389 rewrite), DSpark quantized Markov heads, TurboQuant, YaRN 1M ctx (needle 3/3 at 543,761 tokens). Cross-node CUDA-graph stability carried by #48053 thread_local capture-error-mode, not NCCL version.
https://github.com/AEON-7/vllm-ultimate-dgx-spark — Applies: container provenance + DFlash2 evidence; 27B-class numbers. Port: **M** (adopt build flags, not the whole image).

**F19. Draft-vocab / lm-head byte economics — hashd1ve/apodex-1.1-mini-one-dgx-spark (2026-09-01).**
On a 22.5 GiB model with MTP depth 3, **the borrowed BF16 lm_head is read 4× per cycle (draft×3 + verify) = 55% of all decode bytes**; quantizing just lm_head to FP8 (modelopt static, calibrated on traffic incl. 93k-token prompts) took code decode **70.9 → 93.2 tok/s (+31%)**, prose +24–43%, acceptance unchanged (2.675→2.670). Then — only then — re-sweep draft depth: the optimum moved from 3 to 2 steps once the vocab read halved (78.8 → 93+ at depth 1–2). jschmied independently: "FP8 lm_head becomes *more* valuable under MTP" and it follows roofline.
https://github.com/hashd1ve/apodex-1.1-mini-one-dgx-spark , https://github.com/jschmied/qwen38-flash-next-gb10 (notes/quantizing-lm-head.md) — Applies: **YES** — our MTP k=3 amplifies lm_head bytes exactly the same way (Qwen3.8-Flash-Next lm_head ~0.9–1.3 GiB BF16-class, read 4×/cycle). Port: **M** (calibrate + convert one tensor; sweep k after).

**F20. NVFP4 DSpark gathered top-k projection — vLLM #55713, merged 2026-09-09.**
Lets a *quantized* (NVFP4) DSpark Markov head run by gathering+dequantizing selected W2 rows (Triton kernel) — removes the dense-weight assumption that blocked NVFP4 DSpark drafters. Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark, DL=5, concurrency 1: **128–135 tok/s/user** (RTX-class GPU), acceptance ~3.5.
https://github.com/vllm-project/vllm/pull/55713 — Applies: unlocks quantized DSpark heads for MoE-NVFP4 targets. Port: **S** once we adopt DSpark (with F14).

## 4. Community Spark optimizers — repos, records, and how they did it

**F21. dolf3131/qwen3.8-flash-next-dgx-spark (22 ⭐, updated 2026-09-06) — the recipe our own patch came from.**
Current numbers with `nvidia/Qwen3.8-Flash-Next-NVFP4` (123.6 GiB) via `vllm/vllm-openai:qwen38-flash-next-arm64-cu130` (carries #53896 + PLE-offload #53899; vLLM reports exactly our `0.1.dev20073+g8e685d198`): **33.0 tok/s warm decode, 2,719 tok/s prefill @30k, ~102 tok/s aggregate @ 8 concurrent**, MTP **k=3** ("the optimum moved with the checkpoint"), TTFT 280–390 ms warm, 524,288 ctx (YaRN ×2). PLE 47.7 GiB paged to SSD swap via `VLLM_PLE_CPU_OFFLOAD=1` (~73 KiB disk reads/token ≈ 3% — offload is not the bottleneck; the 512-expert MoE + QSA decode latency is). Extra local patches: `scripts/patch-nv-mixed.py` hunk A (mixed-precision PLE, not upstream) + hunks B1/B2 (port of open #55513 — block-FP8 MTP fix in ModelOpt mixed checkpoints; note #55513 actually closed/merged 2026-09-06). Commits 2026-08-28: "Profile the decode path, and unblock vLLM's own fast GEMM at TP=1" (= the skinny-GEMM/cuBLAS gate-relax we carry) and **"GEMM tuning is saturated: record why, and how the microbenchmark misleads"** — stop spending time there. Trap documented: PLE CPU offload silently hangs at TP=1 (spawn only from multiproc executor).
https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark — Our 50 solo beats their 33 with the same base image → our delta (spec-decode tuning + patches) is real. Port: items **S** (drop B1/B2 once #55513 confirmed in).

**F22. hashd1ve/qwen38-flash-next-one-dgx-spark (14 ⭐, 2026-08-30) — SGLang lane, mmap-PLE origin.**
41.5 tok/s on code at full 262k ctx on SGLang with two small patches: (1) PLE table as `torch.from_file` mmap — cold gather 3.58 ms/16 rows vs 0.12 ms warm, ~26× page amplification quantified; now upstream as sglang `--ple-offload-backend file` (sglang#37068, 2026-08-29). (2) **Correction worth internalizing**: their first patch widened the trtllm gate to sm_12x — on GB10 that path is FlashInfer **XQA and silently corrupts long-context decode** (token-id-0 runs at 120k–210k tokens, HTTP 200 throughout); retired in sglang#36806, fixed properly by the SM121 Triton QSA kernel in sglang#36845 (needle 4/4 at 120k/190k/210k).
https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark , https://github.com/sgl-project/sglang/pull/37068 , https://github.com/sgl-project/sglang/pull/36845 — Applies: the mmap-PLE idea is the ancestor of vLLM #54371/#56273; the XQA-on-SM121 corruption warning applies to any FlashInfer decode path we enable. Port: **M**.

**F23. hasso5703/dgx-spark-qwen38 (170 ⭐, updated 2026-09-11) — currently the fastest *published* Flash-Next recipe on one Spark, and it's SGLang.**
Flash lane (Qwen3.8-Flash-Next 176B NVFP4, SGLang + NEXTN, official SGLang GB10 image, PLE via upstream file backend): **47.9 tok/s on code, 47.1 math, 29–31 prose single stream; 27.0 ms/tok agent loop; prefix caching 27k tokens re-served in 2.5 s vs 12.0 s cold; prefill ~2,250 tok/s; 262k ctx on one box; 4 concurrent requests**. v1.8 (Sep 2026): **`--speculative-token-map` recovered 14–25% of decode** — a draft-vocab trick (constrains/cheapens the draft's vocabulary projection); SGLang-only flag, no vLLM equivalent found. 27B lane: 65 tok/s greedy median (DFlash2 16-deep from a calibrated NVFP4 head), 135–148 tok/s @ 8 streams, 258 @ 32.
https://github.com/hasso5703/dgx-spark-qwen38 — Applies: SGLang's 47.9 code ≈ our 50 solo — no engine switch justified on speed; the `--speculative-token-map` mechanism is the transferable idea (pair with F19 lm_head work). Port: **M** (mechanism port), **L** (engine A/B).

**F24. Other GB10 ecosystem repos (new since Aug):**
- **vladimir-voinea/gb10-laguna-s-2.1-w4a16-moe** (2026-07-28): Triton W4A16 (INT4 weights, BF16 activations) MoE kernels for GB10 — predates the NVFP4 wave; superseded by F11-style paths. https://github.com/vladimir-voinea/gb10-laguna-s-2.1-w4a16-moe
- **agjs/gb10-clock-cap** (57 ⭐, 2026-07-31): `nvidia-smi -lgc 0,2200` on 2×GB10 vLLM TP2 (DeepSeek-V4-Flash-DSpark, 1M ctx): **−12 °C peak (90→78), −36% GPU power, thermal throttling 8.2 s→0 s, decode 73.3→72.5 tok/s (−1.0%)**, cold prefill +3.9%. Knee at 2200 MHz (stock is power-limited ~2455, not the 3003 spec). Decode is bandwidth-bound → clocks nearly free to cap; prefill pays. https://github.com/agjs/gb10-clock-cap
- **sf-stav/veloGB10** (50 ⭐, 2026-09-06): Rust+PTX GB10-native engine; Qwen3.5-122B MoE ~39 tok/s one GB10 / ~57 two; Qwen3.8-27B ~40/56/85(4×, DFlash2); **no Flash-Next support** — not a vLLM beater for us. https://github.com/sf-stav/veloGB10
- **joeynyc/spark-doctor** (104 ⭐): read-only diagnostics — detects the 14 W power-cap state, UMA pressure, CUDA-12-wheel-on-CUDA-13, missing sm_121 arch, KV-cache-OOM vs UMA-pressure distinction. https://github.com/joeynyc/spark-doctor
- **0xBakeer/deepseek-v41-flash-spark** (39 ⭐, 2026-09-11): DeepSeek-V4.1-Flash (510 GB) on ONE Spark via measured-routing-trace hot-expert residency + NVMe `O_DIRECT` streaming; v0.3: 18.98 tok/s, DSpark acceptance 3.03, 40% experts resident at 3-bit codebook. Method (hot-set + streaming) is the interesting part, not the speed. https://github.com/0xBakeer/deepseek-v41-flash-spark
- **MiaAI-Lab/sparkDash** (391 ⭐, updated 2026-09-11; ours): multi-Spark dashboard, live tok/s + thermals; use for sustained-mode capture. https://github.com/MiaAI-Lab/sparkDash
- **FujitsuPolycom/sparkring** (58 ⭐): switchless Spark cluster collective transport. **behindther8326/deepseek-v4-deploy**: deploys via the **jasl vLLM fork PR #41834** (broad SM12x model-enablement branch — the alternative to cherry-picking individual fixes). https://github.com/FujitsuPolycom/sparkring , https://github.com/behindther8326/deepseek-v4-deploy
- vLLM official blog (2026-06-01): sm_121-validated images only; CUDA graphs on by default; keep `--max-num-seqs` low; NVFP4 MoE with 10–15B active params is the sweet spot. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark

## 5. NVIDIA forums / Reddit — concrete numbers

**F25. NVIDIA forum, "Qwen3.8-27B-NVFP4 on a single DGX Spark — up to 1M context, vLLM+MTP measurements" (helge, 2026-08-15+).**
Single Spark ~**65 tok/s** decode class for the 27B (confirmed again Sep for the "Aqua" variant: "same 191k window, same ~65 tok/s decode, slightly faster prefill"); 2-node ray: 37 tok/s (another user ~45); **SPEC_TOKENS=5 crashed vLLM, =3 stable: 8 concurrent sessions ≈ 117 tok/s aggregate**; MTP stability wall at k=5 on that build. Unsloth checkpoint shipped a tokenizer bug silently truncating prompts at 2048 tokens (fixed same day) — check checkpoint provenance before quoting others' numbers.
https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-a-single-dgx-spark-up-to-1m-context-vllm-mtp-measurements/380244 — Applies: k-stability ceiling corroboration; 27B-class, not our model. Port: n/a.

**F26. NVIDIA forum, "DeepSeek-V4-Flash on 4× DGX Spark via vLLM (jasl fork, TP=4, RDMA, MTP): 49–54 tok/s single stream" (2026-06-18).**
Bandwidth-bound single-stream 40–54 tok/s regardless of context on 4 Sparks; the jasl fork (#41834) is the SM12x enablement branch people actually run. Related: 2×Spark TP2 DSpark 1M-ctx recipe at 73.3 tok/s decode (agjs reference config, F24). GLM-5.2 NVFP4 on 4× Spark: 36 tok/s @200k (Reddit follow-up "the MTP mystery", 2026-07-03: MTP gains collapsed at 128K ctx — acceptance drops with context; watch ours at long ctx).
https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-4x-dgx-spark-via-vllm-jasl-fork-tp-4-rdma-mtp-49-54-tok-s-single-stream-full-recipe-the-traps/373808 , https://github.com/tonyd2wild/Deepseek-v4-Flash-TP2-DGX-Spark-500k-CTX , https://www.reddit.com/r/LocalLLaMA/comments/1um6pea/followup_glm52_nvfp4_on_four_dgx_sparks_the_mtp/ — Applies: multi-box only; the MTP-acceptance-vs-context decay is the transferable warning. Port: n/a.

**F27. llama.cpp on Spark: gpt-oss-120B ≈ 35 tok/s (llama.cpp discussion #16578)** — well under vLLM on comparable models; no GB10 development beats vLLM for Flash-Next from llama.cpp this quarter. https://github.com/ggml-org/llama.cpp/discussions/16578

## 6. >70 tok/s solo on ~100B-class audit — none verified on ONE Spark

Every >70 solo claim checked resolves to one of: (a) a ≤35B model (hasso5703 65 tok/s = 27B DFlash2 — the closest, still <70; hashd1ve Apodex 93.2 = 22.5 GiB model; veloGB10 111 = 35B MoE), (b) ≥2 Sparks (agjs 73.3 tok/s = 2×Spark TP2 DSv4-Flash DSpark 1M ctx; forum 37–54 tok/s = 2–4 Sparks), or (c) aggregate concurrency (our own 105 C4; hasso 148 @ 8 streams). Single-Spark Flash-Next solo records: **ours ~50 > hasso5703/SGLang 47.9 (code) > jschmied 36.5 > dolf3131 33.0**. The credible path to >70 solo on one Spark for our model is byte-removal (F19 lm_head FP8, F11 packed PLE) + a stronger drafter (F9/F14/F20), not a kernel silver bullet. URLs: F21–F24, F25, F26.

---

## ACTION LIST for our stack (ranked)

1. **Cherry-pick vLLM #55375** (Qwen4Exp PLE conv state strides — MTP output corruption fix; merged 2026-09-05). Correctness, S, zero risk, same model family.
2. **Cherry-pick #54048** (cuBLAS out_dtype router GEMM on family-120; merged 2026-08-30) and **retire our local dolf3131 gate-relax patch** in its favor. S.
3. **Cherry-pick #53945 + #54713** (EAGLE/MTP × Mamba-align prefix-cache replay boundaries) and **set `--prefix-cache-retention-interval <block_size>`** today for #53504 (first-repeat prefix-cache miss on hybrid+MTP). S — biggest TTFT win for agent workloads per jschmied (warm turn 2.05→1.52 s).
4. **Cherry-pick #54110** (persistent top-k low-smem fallback — stops EngineCore death on MTP decode shapes) and **#54788** if we ever move to Model Runner V2. S.
5. **Quantize lm_head to FP8** (hashd1ve method: modelopt static, calibrate on real traffic incl. long prompts; acceptance unchanged) — lm_head is ~half of MTP-k=3 decode bytes, read 4×/cycle. Expected double-digit % solo decode. Then **re-sweep MTP k (try k=2)** — the optimum moves after the vocab read halves (hashd1ve: best depth dropped 3→2; jschmied: per-draft-step metadata rebuild penalizes high k on QSA_STATE). M.
6. **Cherry-pick #55715** (FlashInfer GDN prefill on SM12x — 3 of 4 layers) for prefill/TTFT; requires FlashInfer ≥0.6.18 + CUDA 13 (we're cu130). S.
7. **Cherry-pick #55450** (Mamba state retirement across null gaps — retained-pool 2,006→254 MiB on hybrid layouts) → convert freed unified memory into KV sessions. S.
8. **Track and early-test #56273** (packed NVFP4 PLE embeddings, 26.8 GiB resident — eliminates PLE CPU offload/swap entirely; GB10-validated). M once mergeable.
9. **Evaluate DFlash2/DSpark as drafter replacement for MTP** via #52816 (already merged, in-branch? verify) + #56088 patches + #55713 (NVFP4 DSpark heads): community evidence 3.4–4× solo over no-spec and 1.66× over DSpark on Qwen3.8-27B; acceptance ~4.6 vs our MTP ~3. L — schedule after 3/5 land.
10. **Ops hygiene on every image bump**: wipe `~/.triton/cache` / set `VLLM_FORCE_TRITON_CACHE_INVALIDATE=1` (#41871/#42859 — sm_121 PTX-fallback garble is silent); never set `VLLM_MARLIN_INPUT_DTYPE=fp8` (#49546); run spark-doctor scan to catch the 14 W power-cap state. S.
11. **Consider `nvidia-smi -lgc 0,2200`** clock cap for sustained multi-hour sessions: −12 °C, throttling eliminated, decode −1% (prefill +4% — skip if prefill-heavy). S, reversible (`-rgc`).
12. **Watch, don't port**: #55557 (FP8 QSA KV — capacity only, decode-neutral on GB10), #55122 (deterministic persistent_topk — no perf cost, fixes greedy forks; take it when it lands), SGLang `--speculative-token-map` (+14–25% decode on the SGLang flash lane — steal the mechanism for our lm_head/draft-vocab work in item 5). SGLang engine switch NOT justified: their best Flash-Next solo (47.9 code) does not beat our 50.
