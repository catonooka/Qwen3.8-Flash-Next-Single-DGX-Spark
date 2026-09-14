<h1 align="center">Qwen3.8-Flash-Next on ONE DGX Spark (TP=1)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow me on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Self-contained recipe for serving the `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`
checkpoint (99 GB) from a single DGX Spark's 121 GiB unified memory, via vLLM
with the PLE table offloaded and memory-mapped. This is a **vision-language**
model: text, images and video all work out of the box (see below). Nothing here depends on the
2-node files it was derived from.

```
cp .env.sample .env        # edit IMAGE / HF_TOKEN if needed
./download.sh              # fetch the ~99 GiB checkpoint (resumable)
./start.sh                 # ~10-12 min to /health; serves on :8888
./stop.sh                  # container + watchdog, graceful
```

`start.sh` never downloads anything — it resolves the checkpoint from the local
Hugging Face cache and fails fast if it is absent. Budget ~130 GiB of free disk:
99 GiB for the checkpoint plus the ~27 GiB packed PLE table built on first launch.

## This fork: measured improvements over upstream `ef1af5f`

Every change was A/B tested on the same host with the same bench scripts and
quality-gated (needles 3/3 at 120k tokens, 11/11 reasoning) before being
kept. The table below is the **current standing stack** (2026-09-12 close,
re-verified unchanged 2026-09-13); the campaign history underneath records
how it got here, newest first. Upstream's own work (65k draft vocab, BF16
recurrent state, FP8 KV hoist, ABLIT lane) is inherited as-is; nothing
below repeats it.

| Lane | Upstream recipe | This fork (standing) | Improvement |
|---|---|---|---|
| Single stream (C1, prose) | 47.6–48.7 tok/s | **54.8–56.7 tok/s** | **+13–17%** |
| Single stream (C1, code) | — | **67.7–68.6 tok/s** | new lane |
| 4 streams (C4, prose) | 107.8 tok/s | **129–133 tok/s** | **+20–23%** |
| 8 streams (C8, prose) | not measured (matched protocol) | **194–198 tok/s** | new lane |
| 8 streams (C8, code / JSON / counting) | — | **252–276 tok/s** (peak 338 counting) | new lane |
| Prefill @32k (262k native) | 2,128 tok/s | 2,332 tok/s | **+9.6%** |
| Prefill @64k (262k native) | 2052 tok/s | 2,279 tok/s | **+11.0%** |
| Prefill @400k (512k YaRN) | 1,602 tok/s | 1,776 tok/s | **+10.9%** (TTFT 250 s → 225 s) |
| Warm agent turn (16k ctx) | ~1.15 s | ~0.51 s | **2.2x faster** |
| Prefix-cache state corruption | present (vLLM race) | fixed (#50729) | correctness |

Acceptance per draft position held at 0.881/0.680/0.485 across the prose decode
wins; near-deterministic text (code, JSON, counting) lifts acceptance to
0.96-1.00 and multiplies the same engine steps into the higher lanes above —
steps/s are identical across content types, the gain is all MTP acceptance.
C8 sustained soak (prose): mean 129.7 / p95 147.2 / min 108.0 tok/s, zero
dips (116,749 tok).

### History (newest first)

**2026-09-14 (day 4) — cluster-recipe levers + full regression.**
Audited github.com/bilikaz/qwen38-flash-next-cluster-recipe (2×Spark kit) for
single-box wins: cpuset pin to the X925 big cores **shipped (+1.1% one-shot,
C4/C8 +3-6% warm)**; `vm.compaction_proactiveness=0` neutral on our CPU-mmap'd
PLE layout (their GPU-resident table is why it pays on theirs); Marlin
atomic-add skipped (already rejected day-2); async-scheduling + hibrid48 NVFP4
output head parked behind the vLLM 0.29 migration. Full post-change regression
(same protocols as day-2): C1 55.2-56.7 / C4 129-133 / C8 194-198 warm,
prefill 2,375-2,433 @32k, quality gates identical to the F4b ship gate
(reasoning 11/11, needles 2/3 known artifact, greedy 9/20 > 2/20 control).
No quality loss.

**2026-09-13 (day 3) — memory floor proven; Bole tree port underway.**
Standing numbers unchanged. Live torch-profiler capture on the final stack:
GPU-busy **93% at C1** — the engine is at its memory floor; eager-draft CPU
overhead ≤7%. MoE NVFP4 grouped GEMM 28.6% / dense MXFP8 25.1% / WMMA bf16
small-N 14.1% / INT8 heads 9.0% / skinny CuTe 7.2% / GDN fused state 1.2%.
Piecewise-drafter lane rejected conclusively (C1 −43%, acceptance collapse
0.57/0.42/0.33); 32k draft-vocab slice closed by tokenizer coverage (VI
prose 30.2% vs 65k's 98–100%); trimix_65k stays. The remaining path to
80 tok/s single-stream is acceptance-side only: Bole tree-verify (plan in
`research4/bole-port-plan.md`; GDN sibling-branch kernel, tree verify-cost
bench, rejection sampler, and QSA ancestor-mask are already
hardware-proven in `f4b/`; est. C1 55 → 65–68 at acceptance 2.98 → 3.6).

**2026-09-12 (day 2) — F4b v2 shipped; standing decode ladder raised.**
Final standing = spinfix + ablit + trimix_fill_65k + MTP3 + vLLM #54048 +
INT8 both heads + F4b v2 (hardened guard a5a8aac) + LPI-1/2/3 off.
Verified warm: **C1 54.8–56.5 / C4 ~125 / C8 186–190** one-shot (from
46–50 / 112–118 / 181). F4b v2 threads the acceptance-aware
`token_indices_to_sample` into the draft stash: step-0 INT8-slice topk
(top-32 of 65,536 per row) replaced by a 64-row gather + exact rescore;
all verify widths B=1..8 engage. Also this day: Marlin atomic-add
REJECTED (clean warm data); HC skinny/INT8 headroom claim REJECTED on-box
(L2-resident 293–497 GB/s). `relaunch.sh` reproduces the whole stack with
no arguments.

**2026-09-07 (day 1) — bandwidth-side campaign.** The four changes:
`MAX_NUM_BATCHED_TOKENS` 2048 → 4096, the skinny-GEMM image (`skinny/`,
build with
`docker build -t vllm-skinny-tp1:v1 -f skinny/Dockerfile.skinny-gemm skinny/`),
vLLM #50729 backport (Mamba copy race), and vLLM #53388 backport
(`disable_eagle_block_drop`, the agent-turn win) — the prefill, agent-turn,
and correctness rows of the table above. Single-stream was flat then
(46–50 tok/s, bandwidth floor); decode was raised on day 2 above.
`.env.sample` ships all of day 1 as default.

Per-change detail, raw A/B rows, and the full list of what was tried and
rejected: [`CHANGELOG.md`](CHANGELOG.md) (newest first) and
[`docs/perf-campaign-2026-09-07.md`](docs/perf-campaign-2026-09-07.md).
