# Bole (arXiv 2608.01651) — Tree Speculation for GDN Hybrids: Port Plan for Qwen3.8-Flash-Next on vLLM/GB10

**Date:** 2026-09-12 · **Scope:** read-only research; no service/container touched.
**Our stack:** vLLM `0.1.dev20073+g8e685d198` (fork), `qwen3_8_flash_next` (36 GDN linear-attn + 12 QSA layers, ~105B A48 MoE NVFP4, vocab 248,320 × hidden 2,560), MTP=3 argmax temp-0, chain speculation, C1 prose 55.8 tok/s (goal 80).

## TL;DR

- Bole is a **kernel–runtime co-design for tree speculation on hybrid-attention (GDN) models**, integrated into **SGLang v0.5.12**, not vLLM. **No public code repo exists** (searched GitHub, SGLang PRs, HF papers page → 404/0 hits). Implementation is ~6.2 kLoC of Python + Triton per the paper, unreleased. Port = **reimplementation from the paper's math**, not a vendor drop.
- Core idea (Theorem 1): replace per-branch serial GDN recurrence with an **exactly equivalent closed form** `O = D_P·Q·S_pre + C·(I+G)⁻¹·D_β·(V − D_P·K·S_pre)` solved via a value-tiled kernel with **on-chip finite Neumann series** for `(I+G)⁻¹`; speculative state kept as token-scale **factors (P, K, U)** instead of per-node state snapshots; one batched matmul commits the accepted path.
- Measured on **GB10 DGX Spark** (same 273 GB/s LPDDR5x as ours): peak **4.72× vs AR**, **2.03× vs SGLang-Tree**, 2.06× vs AdaServe; **86% of serial tree-verify time on GB10 is state materialization** (38% on A100) — the exact bottleneck our chain→tree port must remove.
- It is **training-free** (tree built per-round from the native MTP head's own probabilities, top-k=4, depth 8) and **lossless** (output distribution preserved; kernel algebraically exact). Both claims are paper-verified; residual risk is float reassociation and our fork's known greedy-divergence bug class (vLLM issue #54928 on DFlash2/Qwen3.8).

---

## 1) Paper identity & official code

- **Title:** *Bole: Efficient Tree Speculation for Hybrid-Attention Language Models*, arXiv:2608.01651 (14 pp, 12 fig, 7 tables). Authors: Li Wang, Yi Su, Xiabao Wu, Chiran You, Yongchao Liu, Zhan Qiu, Juelu Zhang, Jiajun Zheng, Fangxin Liu, Jie Zhang, Chen Tian, Chengying Huan. https://arxiv.org/abs/2608.01651
- **Official repository:** **None found.** The full text (https://arxiv.org/html/2608.01651) contains exactly one GitHub URL — citation [36] to the *baseline*: https://github.com/sgl-project/sglang/tree/release/v0.5.12. Verified by:
  - GitHub repo search `bole tree speculation hybrid` / `bole speculative linear-attention` → 0 results (api.github.com, 2026-09-12).
  - SGLang PR search `repo:sgl-project/sglang bole` → 0 results.
  - Hugging Face paper page https://huggingface.co/papers/2608.01651 → 404.
- **License:** therefore **no Bole-specific license**. The host project SGLang is **Apache-2.0**; if Bole lands there it will presumably be Apache-2.0, but today there is nothing to license-gate a direct copy — we must write our own kernel from the paper (arXiv default license permits implementation; no code to fork anyway).

## 2) Components (what Bole actually is)

Paper describes three designs + production integration (§ III–V). No file names are given (closed-source implementation); section map below.

| Component | What it is | Paper anchor |
|---|---|---|
| **Tree drafter** | Uses the model's **native MTP head** as drafter (no extra training). Builds a per-request tree with **top-k = 4, max depth = 8**; scores each node `v` by cumulative draft probability `ρ(v) = ∏_{u∈path(v)} p_draft(u∣π(u))`. | § II-B, § V-A, § VI-A |
| **Batch-wide tree selector** | A **global top-k over ρ across all requests** fills the calibrated budget; monotonicity of ρ keeps each request's selection prefix-connected with no repair pass. Fixed-shape device top-k + scatter → stays inside CUDA Graphs. Variable per-request budget `q_i`, fixed batch total. | § V-A |
| **Factorized GDN tree-verification kernel** | Theorem 1 closed form: `O = D_P Q S_pre + C (I+G)⁻¹ D_β (V − D_P K S_pre)`, exact vs sequential recurrence. Value-tiled kernel: one CTA builds `G, C` in layer-local scratch; `N_v` value-tile CTAs reuse them; `(I+G)⁻¹` via **finite on-chip Neumann recurrence** `Z ← −G·Z; U += Z` (exact because `G` is nilpotent over the tree — ancestor-only propagation, Lemma 3); never materializes the `T×T` diagonals `D_P, D_β`. Fused with GDN's causal-conv front end (§ IV-C). | § IV-A–IV-C, Eq. (2),(8) |
| **Factorized state lifecycle** | One immutable **canonical state** per request; candidate branches stored as token-scale factors **P, K, U**. After sampling, one batched matmul reconstructs & commits only the accepted state; rejected branches need no state slots and no rollback. Three lifetimes: canonical (per-request) / factors (per-round) / G,C + Neumann workspace (per-layer scratch). | § IV-D |
| **Latency-budget profiler** | One-time **offline calibration**: per config `c` (model, GPU, parallelism, batch-size & KV-length bucket, tree-depth bucket) sweep total selected nodes `N` over static CUDA-Graph capacities, record `T_ver(N∣c)` of the *complete* hybrid forward; pick `B_ver(c) = max{N ∈ 𝒢 : T_ver(N∣c) ≤ (1+ε)·T_dec(c)}`. Runtime scheduler reads the bucket's capacity each iteration. | § V-A, Fig. 11 |
| **Engine integration** | In SGLang: device-resident **flat-ragged forest** for selected trees, GPU-resident sampling + accepted-path commit, CUDA-Graph reuse per capacity bucket, continuous batching. ~**6.2 kLoC** Python + Triton total. | § V-B |

**Analogous SGLang v0.5.12 locations** (inferred from repo layout, *not* paper-named — the paper names no files): Triton linear-attn ops under `python/sglang/srt/layers/attention/fla/` (`fused_recurrent.py`, `solve_tril.py`, `wy_fast.py`, `chunk*.py` — where a Bole-style tree kernel would live) and spec-decode workers under `python/sglang/srt/speculative/` (`eagle_worker_v2.py`, `frozen_kv_mtp_worker*.py`, `spec_utils.py`). Source: https://github.com/sgl-project/sglang/tree/release/v0.5.12

## 3) Integration surface with vLLM V1 spec-decode

**Bole provides no vLLM branch or PR** (SGLang-only; no vLLM issues/PRs mention Bole). The vLLM V1 integration surface we must target (verified on upstream `main`, which matches our fork's layout):

- **Proposer:** `SpecDecodeBaseProposer` in `vllm/v1/spec_decode/llm_base_proposer.py` (our chain MTP proposer subclasses it; `_greedy_sample`, `_sample_draft_tokens`, `_get_positions` are the chain-specific bits). Sibling proposers: `EagleProposer` (`eagle.py`), `DFlashProposer` (`dflash.py`), `DSparkSpeculator` (`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, has `_sample_sequential_topk`).
- **Verifier (worker side):** `vllm/v1/worker/gpu/spec_decode/` — `mtp/speculator.py`, `eagle/speculator.py`, `dflash{,2}/speculator.py`, `speculator.py`.
- **Rejection sampler:** `vllm/v1/sample/rejection_sampler.py` (`RejectionSampler`) + `vllm/v1/worker/gpu/spec_decode/rejection_sampler{,_utils}.py`. **Chain-shaped only** — no tree-mask / token-map path found in V1 (EAGLE-3-style tree plumbing is minimal; `eagle/utils.py` only shares a `topk_indices_buffer`).
- **Spec metadata:** `SpecDecodeMetadata` in `vllm/v1/spec_decode/metadata.py` — carries draft-token indices per step; **no ancestor/tree mask concept**.
- **GDN path in vLLM:** V1 backend `GDNAttentionBackend` in `vllm/v1/attention/backends/gdn_attn.py` (chunked prefill via `vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/` — `kernel_h.py`, `kernel_kkt_inv_uw.py`, `kernel_o.py` — i.e. vLLM already ships the UT-transform/WY-style chunk kernels Bole's math is cousin to); layer code `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`; vendored FLA ops in `vllm/third_party/flash_linear_attention/`.

Sources: https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode · https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py · https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/gdn_attn.py · https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl

## 4) Measured numbers on GB10 DGX Spark (paper § VI)

Hardware in paper = ours: **GB10 Grace Blackwell, 256 BF16 TFLOP/s, 128 GB coherent LPDDR5x @ 273 GB/s**.

**Models tested (§ VI-A, Table IV):** Qwen3.5-4B / 9B / 27B on GB10 TP1; Qwen3.5-122B-A10B on 4×A100 only. All four are **Qwen3.5 GDN hybrids with the same 3:1 pattern as ours (3 GDN layers + 1 full-attention layer per repeat)** and all use the **native MTP head as drafter**, unquantized weights, top-k=4, depth=8. **Closest analog to our Qwen3.8-Flash-Next but nothing at 105B-A48/NVFP4 scale; largest GB10 model = 27B dense-ish.**

Key results:

| Metric | A100 | GB10 |
|---|---|---|
| Peak speedup vs SGLang-AR | 3.62× | **4.72×** |
| Peak vs SGLang-Tree | 1.41× | **2.03×** |
| Peak vs AdaServe | 1.39× | **2.06×** |
| Speedup vs best spec baseline, B=1 → B=8 | — | **1.11–1.14× → 1.70–2.03×** |
| Cross-workload (27B, B=4, Fig. 9) vs AR / best spec | 1.98–3.36× / 1.15–1.20× | **2.85–4.12× / 1.38–1.45×** |
| State materialization share of serial verify time | 38% | **86%** ← the GB10-specific win |
| Row-count where extra verify tokens stop being free (Fig. 1c) | ~128 rows | **~256 rows** (a 64-node tree at B=1 is comfortably inside) |
| Geomean over all configs | 2.74× vs AR, 1.26× vs best spec | (same geomean, both platforms pooled) |

- **Kernel-level:** linear-attn tree verification **3.4–7.7× faster**; at B=16 writes **9.8 MB/layer instead of 824 MB** (84×); transient state memory **82–99× lower**; factor commit < 0.5% of target-forward latency.
- **Acceptance (MAT, MBPP, Table III):** Bole 6.38 / **6.88** / 6.62 / 6.40 accepted tokens for 4B / 9B / 27B / 122B-A10B (SGLang-Tree: 6.29 / 6.76 / 6.56 / 6.31). Cross-workload (27B): 6.57–7.08. **Our chain MTP=3 effective length ≈ 3.04 (0.85+0.68+0.51+1) — Bole-class trees ≈ double it.**
- **Memory (Table VI, A100 B=4, 128 nodes):** spec overhead 8.72→1.53 GiB (4B) … 28.04→6.52 GiB (122B) vs SGLang-Tree; lets 27B run B=8 on GB10 where tree baselines **OOM**.
- **Online agents (Open-SWE-Traces):** TTFT/TPOT reduced up to **67.6% / 49.9%** vs strongest baseline; GB10 ran Qwen3.5-27B; prefix-cache hit 92.6% (GB10) vs 58.3% SGLang-Tree.

## 5) Port checklist — Bole pieces → our files (vLLM 0.1.dev20073+g8e685d198)

Our entry points today: `vllm/v1/spec_decode/llm_base_proposer.py` (chain proposer base) and the model's MTP head `vllm/model_executor/models/qwen3_8_flash_next/nvidia/mtp.py` (draft chain of 3, argmax).

| # | Bole piece | Our target file(s) | Work | Difficulty |
|---|---|---|---|---|
| 1 | MTP tree drafter (top-k=4 per step from MTP head, cumulative-ρ scoring, depth ≤ 8; keep argmax child as one branch so temp-0 path is always verified) | `models/qwen3_8_flash_next/nvidia/mtp.py` + new tree logic in a subclass of the proposer in `vllm/v1/spec_decode/llm_base_proposer.py` | Replace `_sample_draft_tokens` chain fill with frontier expansion; emit `draft_token_indices` + parent/child arrays. MTP head itself unchanged (weights reused, training-free) | **M** |
| 2 | Batch budget selector (global ρ top-k, prefix-connected, fixed-shape ops for CUDA Graph) | Same proposer file; scheduler hook where our fork assembles the spec batch | Device top-k + scatter over ≤ ~256 nodes; at C1 trivial (budget = whole tree) | **S–M** |
| 3 | Ancestor-mask verification for the 12 QSA full-attn layers | `vllm/v1/spec_decode/metadata.py` (`SpecDecodeMetadata`) + attention backend mask plumbing in `vllm/v1/worker/gpu/` | V1 has **no tree-mask path** — either custom mask in the FA backend per spec step or flatten tree → padded chain per branch (wasteful). Biggest V1-core surgery | **L** |
| 4 | Factorized GDN tree-verification kernel (Theorem 1 + value tiling + on-chip Neumann; factors P,K,U; batched commit) | New module beside `vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/` (or vendored FLA dir); hooks in `vllm/v1/attention/backends/gdn_attn.py` decode path | Port Eq. (2)/(8) to Triton for sm_121; head dims = our GDN config (d_k×d_v per head, 36 layers). Start from the existing cutedsl chunk kernels' helpers (cumsum, l2norm, conv front end already exist) | **L** (core novelty; ~2–3 kLoC) |
| 5 | Tree-aware rejection sampling / accepted-path commit | `vllm/v1/sample/rejection_sampler.py` + `vllm/v1/worker/gpu/spec_decode/rejection_sampler*.py`; state commit next to GDN cache mgmt | Greedy/temp-0: accept longest argmax-matching root path (no sampling correction needed); factor commit = one batched matmul (paper: <0.5% latency) | **M** |
| 6 | Latency-budget profiler (`B_ver(c)` calibration) | Standalone bench script in `bench/` writing a JSON the engine reads (env/config flag) | Sweep N ∈ {8..256} × KV-length buckets on GB10, record T_ver vs T_dec; ε-gaussian rule from § V-A | **S** |

Suggested order: **6 → 1 → 5 (chain-compatible tree of depth 3, verify as serial branches) → 4 → 3 → 2**. Items 6+1+5 alone give a tree drafter with the old serial verify — measurable acceptance gain, no kernel risk; item 4 is where the paper says **86% of GB10 verify time** lives, and 3 is pure vLLM plumbing.

**Expected gain for us:** paper's GB10 data (row-free zone to ~256 nodes; MAT ≈ 6.9 vs our 3.04; weight-streaming-dominated 105B A48 at 273 GB/s ⇒ tokens/forward ≈ tok/s) supports the 55.8 → 80 tok/s (1.43×) target at C1, but the paper never tested >27B or NVFP4 on GB10 — treat as extrapolation, verify with item 6's profiler first.

## 6) Risks

- **Training-free? Yes.** The tree is constructed per-round from the *existing* native MTP head's probabilities (ρ = path-product of draft probs); no tree structure is learned, no fine-tuning, drafter weights unchanged. (§ V-A, § VI-A: "All tree methods use the same native MTP drafter, model weights, sampling configuration".)
- **Output distribution at temp-0: unchanged in principle.** Bole uses standard acceptance/correction sampling ("without changing the target model's output distribution", § II-B) and the kernel is *algebraically exact* vs sequential recurrence (Theorem 1; finite Neumann is exact since G is nilpotent over a tree, Lemma 3). Residual risks: (a) float reassociation vs the sequential kernel can flip near-tie argmax — same class as chunked-prefill kernels; (b) our fork has a live precedent of greedy divergence from a tree-ish drafter: **vLLM issue #54928 — "DFlash2 changes greedy Qwen3.8 thinking output at token 30"** (https://github.com/vllm-project/vllm/issues/54928). Mitigation: A/B exact-match harness vs chain MTP at temp-0 on prose + thinking traces before any perf claim.
- **No code to port from.** 6.2 kLoC must be written from paper math; kernel (item 4) is original research code, expect the longest tail.
- **Scale gap:** paper's GB10 ceiling is 27B unquantized TP1; ours is 105B A48 MoE NVFP4 with vocab 248,320. MoE routing under multi-token verify and NVFP4 GEMM row-scaling on sm_121 are untested by the paper; the ~256-row "free tokens" threshold may shift.
- **CUDA Graphs:** Bole relies on per-capacity-bucket static graphs; our fork's graph capture must bucket by tree size too, or fall back to eager for spec steps (perf hit).
- **MTP head contract:** Bole assumes the drafter can score arbitrary branches (needs draft probs per branch, not just next-step argmax). Our `mtp.py` currently feeds the last committed token; branching requires feeding each frontier node's token through the MTP head (batched) — confirms feasibility but is the main change in item 1.

## Sources

- arXiv abstract: https://arxiv.org/abs/2608.01651 · full text (all quotes/numbers): https://arxiv.org/html/2608.01651 (accessed 2026-09-12)
- SGLang baseline citation [36]: https://github.com/sgl-project/sglang/tree/release/v0.5.12 (Apache-2.0)
- Qwen3.5 models [34]: https://qwen.ai/blog?id=qwen3.5
- vLLM V1 spec-decode layout: https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode · https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu/spec_decode · https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py · https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/gdn_attn.py · https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl
- Greedy-divergence precedent on our model family: https://github.com/vllm-project/vllm/issues/54928
- Repo/PR non-existence checks (0 results each, 2026-09-12): GitHub search API `bole tree speculation hybrid`, `bole speculative linear-attention`, `repo:sgl-project/sglang bole type:pr`; HF papers page 404.
