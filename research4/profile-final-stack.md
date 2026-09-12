# Live C1 decode profile — final-stack (post INT8 heads + F4b v2)

Captured 2026-09-13 04:11, standing stack @ 9b7b293 (spinfix + ablit + trimix_65k +
MTP3 + #54048 + INT8 both heads + F4b v2 + profiler-config boot). One 800-token
prose generation, C1, temp 0, thinking off. Trace:
`~/.cache/vllm/prof/dp0_pp0_tp0_dcp0_ep0_rank0.1789247454503561230.pt.trace.json.gz`
(161 MB gz, 6.12M events, 23.1s span, 21.5s GPU kernel time).

## Headline: GPU-busy 93% — the engine is memory-floor bound

CPU idle gaps are at most 7% of wall. The "3-8 ms/cycle eager-draft Python
overhead" estimate (bucket 5) is disproven: overhead is hidden behind GPU work,
and the piecewise probe (day 3) showed capturing the drafter loses far more
than that 7% could ever give (acceptance collapse). Bucket 5 is CLOSED.

## Kernel-time decomposition (21.5s total)

| Kernel class | total | launches | ~per cycle* | share |
|---|---|---|---|---|
| cutlass grouped GEMM (MoE NVFP4, FLASHINFER_CUTLASS) | 6.16 s | 37,056 | 49 | 28.6% |
| flashinfer DeviceGemmMxfp8Sm120 (dense projections) | 5.39 s | 78,720 | 105 | 25.1% |
| cutlass_80 WMMA bf16 (cuBLAS small-N fallback) | 3.03 s | 84,829 | 113 | 14.1% |
| cutlass_80 i16832 s8 (INT8 heads: verify+draft) | 1.94 s | 1,544 | 2 | 9.0% |
| skinny CuTe GEMM (SM12x low-latency path) | 1.54 s | 39,274 | 52 | 7.2% |
| Marlin MoE (drafter NVFP4 fallback) | 0.53 s | 2,316 | 3 | 2.4% |
| gdn_decode_post_conv_mtp_kernel (GDN fused state) | 0.26 s | 13,824 | 18 | 1.2% |
| gemvx + elementwise + topk + quantize + misc | ~2.7 s | — | — | ~12.4% |

*cycle = one MTP-3 spec cycle (~270 cycles in the window; per-cycle counts are
launches/750 for comparability with earlier estimates).

## Findings that change the roadmap

1. **GDN state cost is 1.2% of GPU time** (fused `gdn_decode_post_conv_mtp`
   + `_causal_conv1d_update`, ~18 launches/cycle, 227+68 us total per cycle).
   The Bole paper's "86% of verify time is state materialization" is an
   SGLang-stack number and DOES NOT TRANSFER: vLLM's fused GDN decode path
   already solved it. **Bole's kernel work (factorized verification, Neumann
   solve) is unnecessary for us. Bole's remaining value = its TREE ALGORITHM**
   (native-MTP drafter, per-round tree from draft probabilities, batch-wide
   selection, calibrated budget): acceptance-length 2.98 → 3.5-4 class ⇒
   +18-35% throughput at unchanged verify cost. That is the ONLY remaining
   lever of this size, and it is a multi-day port (vLLM V1 rejection sampler
   is chain-shaped; no tree mask support).
2. **Dense MXFP8 at 25% is already quantized** — 105 launches/cycle at
   68.5 us mean. This is the HC-mixer/projection family running the SM120
   MXFP8 path. There is no "quantize the dense layers" headroom left; Minima-
   style W4A4 would have to beat an already-MXFP8 baseline (~2x at best on
   this slice = ~12% cycle, realistically less).
3. **WMMA bf16 fallback at 14%** (113 launches/cycle, 29.8-47.2 us): the
   Ampere-era cuBLAS path for small-N shapes. But the on-box HC microbench
   (day 2) showed these weights are L2-resident at 293-497 GB/s effective —
   the kernels are already near their achievable bandwidth; routing them to
   skinny CuTe plans would trade L2 hits for streamed reads. CLOSED unless a
   tactics-cache entry proves otherwise.
4. **INT8 heads 9%**: verify (1.26 ms) + draft steps; measured elsewhere at
   81% of bus. At floor.
5. **MoE grouped GEMM 28.6%**: NVFP4 FLASHINFER_CUTLASS on sm_121 (SM120
   TMA path). The SM12x-native b12x family crashes on our chip (closed day
   1). Only an image bump could change this kernel.

## Bottom line

Chain-MTP on this image is at its memory floor: every major bucket is either
already quantized (MXFP8/INT8/NVFP4), already fused (GDN), or L2-resident
(WMMA). Remaining paths to 80 tok/s, in order:
- **Bole tree algorithm port** (L, the only +20% class lever left) — plan:
  `research4/bole-port-plan.md`, now motivated by acceptance, not kernels.
- **Image bump** when MiaAI-Lab/upstream ships one (b12x MoE family fixed
  for sm_121 + newer FlashInfer GDN/tactics) — free-kernel class.
- **Minima W4A4 dense** (L) — fights an already-MXFP8 25% slice; low ROI.

Everything S/M-class is exhausted with receipts (see CHANGELOG 2026-09-12/13).
