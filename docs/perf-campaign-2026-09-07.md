# Perf campaign write-up: 2026-09-07

Goal: +30 percent on single-stream decode, C4 aggregate, and prefill, with
zero quality loss. This document records what was measured on this host,
what shipped, and what was honestly rejected. Every number below came from
a same-boot A/B on this DGX Spark, not from upstream claims.

## TL;DR

Baselines matter and there are two sets in circulation. The base README's
headline numbers (48.7 single, 113.7 at 4 streams, 1,942 prefill at 32k)
were measured at 512k YaRN with MAX_NUM_SEQS=8. The rows below are the
matched A/B: upstream recipe vs this fork, same host, same benches, at the
262k-native settings this repo ships. Both are real measurements; they are
not comparable to each other.

| Lane (262k native, matched A/B) | Upstream recipe, this host | This fork, same host | Change |
|---|---|---|---|
| Single stream, prose | 47.6 to 48.7 tok/s | 46 to 50 tok/s | flat |
| 4 streams aggregate | 107.8 tok/s | 112 to 118 tok/s | +4 to +9 percent |
| Prefill at 32k | 2,128 tok/s | 2,332 tok/s | +9.6 percent |
| Prefill at 64k | 2,195 tok/s | 2,279 tok/s | +3.8 percent |
| Warm agent turn (16k ctx) | about 1.15 s | about 0.51 s | 2.2x faster |

Quality: every shipped change passed the gate (needles 3/3 at 120k tokens
at 5, 50, and 95 percent positions, plus the 11-task reasoning suite).

## What shipped

### 1. Bigger prefill chunks: MAX_NUM_BATCHED_TOKENS 2048 to 4096

When the model reads a long prompt it works in chunks. Every chunk pays
scheduler overhead, and the FP8 GEMM kernels hit better throughput near 4k
rows. Doubling the chunk size gained about 5 percent prefill at 32k.

Nature of change: one line in .env. No code touched.
Note: 8192 was also tested later and lost 3.7 percent. Chunks larger than
about 4k overflow L2 locality for the weights. 4096 is the measured sweet
spot.

### 2. Skinny-GEMM image (from earlier in the campaign)

The stock image ships general-purpose matrix kernels. Decode with 1 to 4
rows is a skinny shape where a generic kernel wastes parallelism. The
skinny image adds a fast path with tile sizes picked for exactly our
shapes. Measured +4.4 percent prefill this week on top of the chunk change,
and +8.6 percent decode in the earlier MTP k=2 A/B.

Nature of change: one Docker layer that rewrites low_latency_gemm.py inside
the image. The patch asserts its own anchors, so the build fails loudly if
upstream moved the code.

### 3. Mamba prefix-cache race fix (vLLM PR 50729, backported)

This model is a hybrid: 36 of 48 layers are linear-attention (Mamba) layers
that keep a running state per conversation, like a rolling summary. When a
new request reuses a cached prefix, the engine copies the old state into a
new slot. The old kernel copied byte-major inside a row loop. If the
destination overlapped the source, one lane could overwrite bytes another
lane had not read yet. Silent corruption, no error.

The upstream fix restructures the copy to be token-major (memmove-safe) and
adds a debug barrier on left-overlap.

Why it mattered to us: prefix caching is ON in our standing config, so the
race was live. This is a correctness fix, not a speed change.

Nature of change: upstream diff sliced to the two Triton kernels, applied
clean at small offsets, mounted read-only over the installed file via
EXTRA_DOCKER_ARGS. Tracked as files/mamba_utils_50729.py.

Lesson learned while shipping it: a parallel session on this host reverted
.env mid-experiment, and one gate run silently executed against an
unpatched boot. Since then, every gate is paired with an in-container
marker check in the same breath.

### 4. Keep the trailing prefix-cache block under MTP (vLLM PR 53388, backported)

The biggest real-world win. Speculative decoding drafts 3 tokens ahead and
verifies in one pass. Because of that, vLLM assumed the last matched
prefix-cache block could be inconsistent after verification, so it dropped
that block from the cache on purpose. The next turn re-read about 1,600
tokens it already had.

Upstream found the drop unnecessary: the trailing block KV is valid. Only
the Mamba state is stale, and state is rebuilt at chunk boundaries anyway.

The flag is disable_eagle_block_drop. It exists on vLLM main but not in our
image, so six vllm/ files were backported and mounted. With the flag unset
the patched files behave identically to stock, which let the A/B flip only
the flag on the same mounts.

Measured with bench/warmturn.py (16k context, five agent turns, two runs):

- First re-prefill after the cold prompt: 2.53 s to 1.17 s
- Steady warm turns: about 1.15 s to about 0.51 s
- Decode lanes: unchanged
- Acceptance: 2.88 vs 2.90 tokens per step, unchanged

We beat the upstream claim of 26 percent because our agent prompts are
longer and more cacheable.

## What was tested and rejected

Each of these was built, measured on a same-boot A/B, and reverted. The
numbers are recorded so nobody re-runs them blindly.

### MXFP8 lm_head quantization: minus 7.7 percent decode

The output layer (lm_head, 248k by 2,560, about 1.2 GiB in BF16) was
rewritten in the checkpoint to 8-bit E4M3 with per-32-group E8M0 scales.
The round-trip error was 2.66 percent, inside the 3 percent gate. The
loader needed one argument added so the head would pick up its quant
config, and the MTP drafter needed a dequant-in-flight path to keep its
BF16 head.

It lost anyway: the 8-bit GEMM kernel at batch 4 or below is slower than
the skinny-tuned BF16 kernel already in the image. Bytes halved, time up
7.7 percent. Reverted. The surgery pipeline is reusable when better MXFP8
kernels land.

### MNBT 8192: minus 3.7 percent prefill at 32k

Tested as a pair against 4096 on the same boot. 2,265 vs 2,181 tok/s at
32k, 2,255 vs 2,203 at 64k. The older one-shot +10.9 percent measurement
did not reproduce; that pair differed in rope and KV config.

### IndexShare MTP: neutral

The image already ships the full machinery (set_skip_topk in mtp.py,
_share_mtp_indices in the proposer). It is off only because the checkpoint
lacks the flag. Enabled via the new HF_TEXT_OVERRIDES lane. Measured flat
on C1, C4, and long-context decode, acceptance unchanged. Our draft
indexer is not the binding constraint; the memory floor is. Left off.

### long_prefill_token_threshold 2048: a trade, not a win

Caps the per-request chunk so decode gets scheduled between prefill
chunks. During a 64k prefill, concurrent decoder ITL improves from 1,489
to 877 ms (minus 41 percent). But solo prefill pays minus 10 to 12 percent.
Since prefill is a target lane, rejected for the standing config. It is
the opt-in for a multi-tenant case: someone coding while a big document
gets ingested.

## Why the 30 percent decode target is not reachable on this image

The model is about 100 GB. At the machine's 273 GB/s memory bandwidth, one
pass over the weights takes about 42 ms. That is the floor for a decode
step. We run at about 60 ms. Every kernel-level lever we tested to close
the gap (head quantization, index sharing, chunk sizes) either matched or
lost to what is already running.

What remains on the table:

- vLLM PR 55180, the FP8 blockwise GEMM raster swizzle, worth 10 to 15
  percent prefill. It touches CUDA source that is not present in the
  runtime image, so it needs a full source build of vLLM. That is an
  hours-class project with the server down.
- PRs 54513 and 54873 (QSA prefill split and sparse GQA) patch the
  qwen4_exp package, not our qwen3_8_flash_next. A manual port.

## How to reproduce the standing config

- IMAGE=vllm-skinny-tp1:v1
- MAX_NUM_BATCHED_TOKENS=4096
- SPEC_DISABLE_EAGLE_BLOCK_DROP=1
- EXTRA_DOCKER_ARGS mounts files/mamba_utils_50729.py and the six s3/out53388
  files over their installed paths
- Patches verified live with in-container marker greps before any gate

Bench scripts: bench/sweep.py (decode lanes), bench/prefill_ladder.py
(salted prefill), bench/warmturn.py (agent turns), bench/mixed.py
(prefill and decode interference).
