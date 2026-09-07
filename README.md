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

Four changes, each A/B tested on the same host with the same bench scripts,
each quality-gated (needles 3/3 at 120k tokens, 11/11 reasoning) before
being kept. Upstream's own work (65k draft vocab, BF16 recurrent state, FP8
KV hoist, ABLIT lane) is inherited as-is; nothing below repeats it.

| Lane | Upstream recipe | This fork | Improvement |
|---|---|---|---|
| Prefill @32k (262k native) | 2,128 tok/s | 2,332 tok/s | **+9.6%** |
| Prefill @64k (262k native) | 2052 tok/s | 2,279 tok/s | **+11.0%** |
| Prefill @400k (512k YaRN) | 1,602 tok/s | 1,776 tok/s | **+10.9%** (TTFT 250 s → 225 s) |
| Warm agent turn (16k ctx) | ~1.15 s | ~0.51 s | **2.2x faster** |
| 4 streams aggregate | 107.8 tok/s | 112–118 tok/s | **+4–9%** |
| Single stream | 47.6–48.7 tok/s | 46–50 tok/s | flat (bandwidth floor) |
| Prefix-cache state corruption | present (vLLM race) | fixed (#50729) | correctness |

The four changes behind the table: `MAX_NUM_BATCHED_TOKENS` 2048 → 4096,
the skinny-GEMM image (`skinny/`, build with
`docker build -t vllm-skinny-tp1:v1 -f skinny/Dockerfile.skinny-gemm skinny/`),
vLLM #50729 backport (Mamba copy race), and vLLM #53388 backport
(`disable_eagle_block_drop`, the agent-turn win). `.env.sample` ships all
of it as default; a fresh clone reproduces the numbers with no manual
editing.

Trade-offs and the full campaign log (including what was tried and
rejected, with numbers): [`docs/perf-campaign-2026-09-07.md`](docs/perf-campaign-2026-09-07.md).
