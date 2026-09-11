# CUDA + system-level research: maximum decode throughput and stability on GB10

Compiled 2026-09-12 on `gx10-a416` (GB10, sm_121, kernel 6.17.0-1014-nvidia, driver 580.142,
CUDA 13.0, vLLM `vllm-skinny-tp1:v1-spinfix` container, Qwen3.8-Flash-Next NVFP4, MTP k=3,
CUDA graphs FULL_DECODE_ONLY, capture sizes [4,8,12,16,20,24,28,32], MNBT=4096, MAX_NUM_SEQS=8,
FP8 KV, chunked prefill). READ-ONLY research: nothing below was executed against the live
server; every command listed under "IMMEDIATE" is a recommendation, not something this session ran.

Live host facts verified today (all read-only):

- `clocks.max.sm = 3003 MHz`, idle SM 2184 MHz, persistence mode **Enabled**, 23.5 W idle, 55 C,
  `clocks_throttle_reasons.active = 0x0` (`nvidia-smi --query-gpu=...`).
- cpuidle: driver `acpi_idle`, governor `menu`. States per CPU: LPI-0 (WFI, 0 us) enabled,
  LPI-1 (42 us exit) **enabled**, LPI-2 (231 us) **already disabled**, LPI-3 (433 us)
  **already disabled**. NOTE: the task context said "LPI-3 disabled only"; the live system has
  BOTH LPI-2 and LPI-3 disabled — the earlier +13.9% C4 measurement may include LPI-2's effect.
- cpufreq: governor `performance` on all 20 policies already. X925 (3.9 GHz) = CPUs
  5-9 and 15-19; A725 (2.8 GHz) = CPUs 0-4 and 10-14 (from per-CPU `cpuinfo_max_freq`).
- THP: `enabled = madvise`, `defrag = madvise`.
- hwmon: only `acpitz`, `nvme`, `mt7925_phy0` — **no `spbm` node**, so pl1/syspl1 firmware power
  caps are NOT readable on this host without installing antheas/spark_hwmon.
- irqbalance: inactive. NVMe queue IRQs nvme0q1..q5 are individually pinned to CPUs 0-5 (A725 side)
  with 7-9.4M events each (`/proc/interrupts`).
- In-container: flashinfer 0.6.17, vLLM 0.1.dev20073+g8e685d198, NO `deep_gemm` module installed.
- Running engine (from `docker logs`): `cudagraph_mode=FULL_DECODE_ONLY`,
  `cudagraph_capture_sizes=[4,8,12,16,20,24,28,32]`, graph memory 0.18 GiB, capture time 4 s,
  `enable_flashinfer_autotune=True`, KV cache in use 16.5 GiB (headroom to 34.69 GiB at
  gpu_memory_utilization=0.786), 114.11 GiB free at startup.

---

## 1. GB10 clock governance: -lgc, hidden fast/slow states, persistence

**The single most important finding for our "C4 dips during long runs" symptom.**

A same-generation cluster report (DeepSeek-V4.1-Flash on 4x GB10, TP4, DSpark k=5 — nearly our
workload class) documents a **hidden slow state** that nvidia-smi cannot see:

- GEMV 224-233 GB/s (fast) vs 66-80 GB/s (slow) — 3.3x — while reported SM clock stays
  2164-2190 MHz and throttle reasons stay 0x0 in both states. Power under GEMV: 18-23 W fast,
  14-16 W slow. Serving impact: 63 vs 94 ms/decode step = 92 vs 62 tok/s. States flip abruptly,
  last 7-32 s each, can flip mid-request, and the first request after a long idle runs slow end
  to end. Ruled out: thermal, SW power cap (counter never moves), CPU power sharing, PCIe ASPM,
  network, core placement, KV/prefix cache. Leads: firmware power limits (pl1/syspl1), hidden
  memory/L2/fabric clock scaling (device copy unchanged while GEMV loses 3.3x — points below the
  SM clock).
  Source: https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1 (2026-09-10,
  kernels 6.17.0-1014/-1021, drivers 580.142/580.159.03 — **our exact kernel and driver build**).

- An 8x DGX Spark FE fleet running **BIOS 5.36_0ACUM027, kernel 6.17.0-1032-nvidia, driver
  580.173.02** saw **0 slow seconds in ~540 measured seconds** across all nodes, with vLLM TP8
  resident — the slow state never appeared there, locked or unlocked.
  Source: same issue, comment by im0xMagnus (2026-09-11).

**`nvidia-smi -lgc` on GB10 — does it work?** Yes, on driver 580.173.02: a systemd unit running
`nvidia-smi -lgc 0,2200` holds SM at 2184-2190 MHz even though `applications_clocks_setting`
still reads "Not Active". But the same comment measured that the lock is a **power cap, not a
fast-state holder**: locked = 77-83 TFLOPS matmul at ~18 W; unlocked = 85-86 TFLOPS at ~28 W
(boost to 2489 MHz median, 2554 max). GEMV bandwidth was unchanged (223 GB/s both ways), and the
unlocked node was just as flip-free. On our older 580.142 the original reporters locked clocks
and were "not sure the lock is doing anything".

- Separate, known failure: the **513 MHz cap** (3x slowdown, no throttle reason) is a USB-PD
  negotiation fault. Register-level signature via the reverse-engineered `spbm` hwmon node:
  pl1 = 20 W / syspl1 = 30 W vs healthy ~140 W / ~231 W after reboot. Fix: full AC power cycle
  of the brick (a warm `reboot` cleared it for one reporter but the PD MCU stays powered, so
  unplug-from-wall is the reliable procedure). Diagnostic tools:
  https://github.com/hoesing/spark-gpu-throttle-check and fork
  https://github.com/parallelArchitect/spark-gpu-throttle-check.
  Sources: https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296
  (posts #5, #8, #10, #11); driver https://github.com/antheas/spark_hwmon.

- **Persistence mode**: already Enabled on this host; `nvidia-smi -pm 1` is a no-op/idempotent
  safety check. GB10 does downclock between steps at idle-ish decode loads (2184 MHz observed
  idle; DVFS boosts to ~2.5 GHz only under heavier load), which is normal and not by itself the
  slow-state bug.

**Does GB10 downclock under sustained decode?** No thermal/power downclock evidence in any
source; the risk is the *hidden* fast/slow flip and the PD-cap failure mode, both invisible to
`clocks_throttle_reasons`.

**Recommendation / expected gain / risk:**

| Action | Expected gain | Risk |
|---|---|---|
| Plan platform update to BIOS 5.36_0ACUM027 + kernel 6.17.0-1032 + driver 580.173.02 | Removes the suspected cause of our C4 dips (fast/slow flip). If our dips are the same 3.3x GEMV state, C4 sustained gets its ~30-50% dip windows back | Firmware update requires maintenance window; re-validate benchmark after |
| AC power-cycle the brick (unplug wall, 60 s) before long benchmark campaigns | Clears any latched PD state (pl1/syspl1 at 20/30 W) that degrades clocks 3x with zero NVML signature | None; takes minutes |
| Install spark_hwmon (DKMS) to log pl1/syspl1 during a dip | Converts "mystery dip" into a readable firmware counter | Out-of-tree kernel module; build risk on 1014 kernel |
| `nvidia-smi -lgc 0,2200` | None for decode (GEMV unchanged); REDUCES matmul boost ~9% (85->78 TFLOPS). Only useful as a power/heat cap | Confirmed perf loss on compute-bound phases; on 580.142 effectiveness unverified |
| `nvidia-smi -lgc 1800,3003` (raise floor, keep boost) | Untested anywhere; plausible jitter smoothing at step boundaries | Unmeasured thermals; run as explicit A/B with bench_sustained.py |

## 2. cpuidle/pstate tuning beyond LPI-3

Current state: LPI-1 (42 us) is the deepest state still enabled; LPI-2/LPI-3 already off;
governor `menu`; cpufreq governor `performance` everywhere (verified live today).

- **Disable LPI-1 on the X925 cores only** (5-9, 15-19): the vLLM EngineCore/API threads live
  there; WFI exit latency of 42 us lands directly on the PLE handshake and step scheduling tail.
  Keep LPI-1 enabled on A725 cores (0-4, 10-14) where background work idles — this preserves
  most of the idle power saving. Command pattern (per-CPU):
  `for c in 5 6 7 8 9 15 16 17 18 19; do echo 1 | sudo tee /sys/devices/system/cpu/cpu$c/cpuidle/state1/disable; done`
  Expected gain: small (the big win, LPI-3's 433 us, is already banked; LPI-1 is 10x shallower).
  Evidence class: mechanism (exit latency on the critical path) + our own LPI-3 result; no
  published GB10 LPI-1 measurement exists. Risk: idle power up a few W on 10 cores; trivially
  reversible.
- **Menu governor tunables**: `/sys/devices/system/cpu/cpuidle/current_governor` = menu. The
  menu governor's only knob is `menu_latency_factor` (module param `cpuidle.menu_latency_factor`,
  default 6? — int, higher = more conservative about deep states). Lowering it makes the governor
  pick shallower states at short predicted idle; with only LPI-0/LPI-1 left the governor has
  almost no choice space, so skip this. Source: Linux kernel docs
  https://docs.kernel.org/admin-guide/pm/cpuidle.html and
  https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html (cpuidle_menu gov).
- **Alternative `teo` governor**: better for timer-driven serving loads in principle
  (https://docs.kernel.org/admin-guide/pm/cpuidle.html#menu-and-teo-governors). With 2 states
  left the difference is noise — not worth a reboot-time param.
- **Boot-param isolation** for the API server lane (stronger than cpuidle knobs):
  `isolcpus=5-9,15-19 nohz_full=5-9,15-19 rcu_nocbs=5-9,15-19 irqaffinity=0-4,10-14`
  removes scheduler tick + RCU callbacks + default IRQ routing from the serving cores.
  Sources: https://docs.kernel.org/admin-guide/kernel-parameters.html;
  https://ohyaan.github.io/tips/cpu_isolation_and_task_affinity_for_multicore_optimization/
  Expected gain: jitter reduction on step-time p99; no GB10-specific measurement. Risk: boot
  param; must ensure system services are affined off the isolated set or the box feels sluggish.
- **big.LITTLE placement** (already cheap to do without boot params): pin the container to X925
  cores with `--cpuset-cpus=5-9,15-19` (see topic 8). The flip-issue authors verified their busy
  threads spent 0% on A725 cores via per-thread last-CPU sampling — placement was not their
  problem and is probably not ours, but pinning removes migration jitter for free.
  Source: flip issue section 3 ("CPU core placement" row).
- Reference for cpupower idle-state manipulation:
  https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/index
  (`cpupower idle-set --disable 1` per core is the packaged equivalent of the sysfs write).

## 3. flashinfer offline autotune: exact API to dump/load tactics JSON (0.6.17)

Verified by reading the installed source in the running container
(`/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner/autotuner.py`, v0.6.17):

- **Python API** (preferred):
  ```python
  from flashinfer import autotune
  # tune + auto-save on exit:
  with autotune(True, cache="/path/tactics.json"):
      model(inputs)                      # replay real/bench shapes
  # later: load-only (no profiling):
  with autotune(False, cache="/path/tactics.json", round_up=True):
      model(inputs)
  # advanced/manual:
  from flashinfer.autotuner import AutoTuner
  AutoTuner.get().save_configs("/path/tactics.json")
  AutoTuner.get().load_configs("/path/tactics.json")   # -> bool (False = env mismatch, skipped)
  ```
- **File format**: JSON object; first key `_metadata`
  `{flashinfer_version, cuda_version, cublas_version, cudnn_version,
  cudnn_frontend_version, gpu}`; then one key per tuned op:
  `"('mxfp8_gemm', 'CutlassMxfp8GemmRunner', ((1,2560),(2560,1280),(-1,),(102400,),(0,),(-1,1280),(33554432,)), ())"` →
  `["CutlassMxfp8GemmRunner", 1]` (runner class name + tactic id; tuples serialized as lists and
  restored via `_json_to_tactic`).
- **Hard env gate on load**: all six metadata fields must match (or be `*`) or the entire cache
  is skipped. Ours today: `0.6.17 / 13.0 / 13.1.1 / 92000 / 1.27.0 / "NVIDIA GB10"`.
  Consequence: a flashinfer 0.6.17 -> 0.6.18 bump silently invalidates the file — retune after
  any wheel change and diff entry counts.
- **Legacy env**: `FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1` enables the older bundled `.py`
  configs path; `FLASHINFER_AUTOTUNE_TIMER` overrides the timing-loop choice (source lines
  ~1158-1162, 1342).
- **vLLM integration (this build, already ON)**: `--enable-flashinfer-autotune` /
  `kernel_config.enable_flashinfer_autotune=True` (seen in live engine config). vLLM resolves the
  cache path itself: `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/<fi-ver>/<arch>/<sha256(aot_compile_hash_factors)>/autotune_configs.json`
  — concretely
  `/root/.cache/vllm/flashinfer_autotune_cache/0.6.17/121a/65158578.../autotune_configs.json`
  (logged as "Using FlashInfer autotune cache file: ..."). It tunes with
  `runner._dummy_run(num_tokens=max_num_batched_tokens=4096, randomize_inputs=True)` inside
  `autotune(tune_mode=True)`, then `tuner.save_configs(path)` — flashinfer profiles all token
  counts up to that max in one pass (comment in `vllm/model_executor/warmup/kernel_warmup.py`).
  Env overrides: `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` (relocate cache),
  `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` (skip op list; vLLM auto-skips `fp4_gemm` when the
  CuTe-DSL NVFP4 kernel is selected because tuning JIT-compiles every tactic).
- **Current cache contents (read live)**: 162 entries, ops tuned this boot: `mxfp8_gemm` (45
  profile buckets) and `trtllm` (44) — i.e. the MXFP8 dense/blockscaled GEMMs and TRT-LLM-gen
  MoE path. 4 tactics were skipped as unsupported on 121a per shape bucket
  ("Skipped 4 unsupported tactic(s) for mxfp8_gemm" in logs). The cache persists across restarts
  (15 dated files since 2026-09-04; the current hash dir was written at yesterday's launch).
- **Offline dump/load workflow for grouped MoE GEMM on our box** (no server restart needed to
  *inspect*; retune happens at next normal boot):
  1. `docker exec vllm-fn-ab1 cat /root/.cache/vllm/flashinfer_autotune_cache/0.6.17/121a/<hash>/autotune_configs.json > tactics-121a.json`
     (backup/diff artifact).
  2. To force a fresh tune at next launch: delete/move the `<hash>` dir; boot the server; the
     warmup dummy-run at 4096 tokens regenerates it (~10-20 s, measured in logs 18:51:45->18:52:37
     including graph capture).
  3. To hand-port tuning across config changes (e.g. new MNBT): the hash comes from
     `aot_compile_hash_factors(vllm_config)`, so a changed config writes a NEW dir and re-tunes
     from scratch — copying the old JSON into the new dir works only if metadata matches; the
     config-derived key set must cover the same shapes.
- **Known sm_12x tunings**: no public tactics table exists for 121a; our own cache (162 entries
  above) IS the sm121a tuning record. Additionally `python3 -m flashinfer
  generate-tactics-blocklist` probes the local GPU and writes an offline blocklist of tactics
  that crash/miscompute (subcommand confirmed in `python3 -m flashinfer --help`), and
  `tactics_blocklist.py` ships in the wheel.

## 4. CUDA graphs + MTP: interplay, capture-size tuning, graph memory pools

- **Mode semantics** (vLLM v1 doc, https://docs.vllm.ai/en/stable/design/cuda_graphs/):
  uniform decode = `max_query_len == 1` OR `1 + num_spec_tokens` — MTP verify batches ARE
  uniform decode, so FULL_DECODE_ONLY captures them. FULL_AND_PIECEWISE is the upstream default
  when piecewise compilation is available; FULL_DECODE_ONLY "saves the memory needed for
  PIECEWISE CUDA Graphs" — right choice for us since COMPILATION_MODE=0 and the PLE custom op
  must stay eager.
- **Capture sizes — measured on this exact stack** (upstream repo commit e33a9f8, merged into
  our fork 2026-09-08): vLLM's stock list `[1,2,4]+multiples of 8` rounded to multiples of
  `(1+MTP)` and capped at `(1+MTP)*MAX_NUM_SEQS` leaves only keys {4,8,16} at MTP 3 — a
  12-token (3-seq) verify batch pads to 16 and reads a 4th request's experts for nothing, and a
  20-token (5-seq) batch matches no key and decodes EAGER. `CUDAGRAPH_CAPTURE_SIZES=auto`
  (shipped in our `.env`) captures every `(1+K)*S` width: at MAX_NUM_SEQS=8 that is
  [4,8,12,16,20,24,28,32] — exactly the live engine's list. Measured value ~4-5 ms on the
  5-sequence step; nothing at 1/2/4 streams where graphs already existed. Our sizes are already
  optimal; the only check left is that MAX_NUM_SEQS stays 8 so the width-32 graph exists.
- **MTP_K_SCHEDULE trap (do not set)**: upstream measured that setting it makes vLLM fall back
  FULL_DECODE_ONLY -> PIECEWISE: +24% single-stream step time and driver memory 99.4 GiB against
  a 94.87 GiB budget until the watchdog killed the server (commit e33a9f8 message). Our `.env`
  correctly ships it empty.
- **Graph memory pool**: measured 0.18 GiB for 8 decode graphs + 2 prefill/decode speculator
  graphs, capture time 4 s (live log). Memory is NOT a constraint at these widths; capturing
  more sizes would be cheap, but there are no more widths the scheduler can build at MAX_NUM_SEQS=8.
- **Graph capture + MTP crash class**: vllm-ascend #8587 documents FULL_DECODE_ONLY + MTP
  crashing under ACL graph replay (Ascend NPU-specific; not our CUDA path) — useful only as a
  reminder that spec-decode graphs assume the uniform-decode contract above.
  Source: https://github.com/vllm-project/vllm-ascend/issues/8587
- **KV headroom note from the same log line**: engine suggests `--kv-cache-memory=37249486336`
  (34.69 GiB) to fully use the 0.786 budget vs 16.5 GiB in use. Raising KV pool helps C8+
  concurrency headroom, NOT step time; and the GUI-coexistence rule (0.94 util NVRM-OOM-crashes
  desktop apps) caps how far to push it.

## 5. PDL and DeepGEMM sm120 kernels

- **No DeepGEMM here**: `import deep_gemm` fails in our container. Upstream DeepGEMM supports
  SM90/SM100 (README news 2025.07.20); sm_120/121 is not in its supported-arch list, and its PDL
  work (2026-04-16, PR #304 "Mega MoE, FP8xFP4 GEMM, ... PDL") targets those archs. Porting
  DeepGEMM to sm_121 is a source-build project, not a config flip.
  Source: https://github.com/deepseek-ai/DeepGEMM (README + News).
- **PDL in flashinfer on SM120 — already present and default-ON for the path we use**:
  `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` has `enable_pdl: bool = True`
  and calls `griddepcontrol_launch_dependents` / `griddepcontrol_wait` (CUTLASS CuTe DSL PDL
  primitives). This is the B12X blockscaled (MXFP8/NVFP4-class) GEMM path for sm_120. So MXFP8
  GEMMs in our stack already overlap dependent launches; no env to flip.
- **bf16 GEMM M=4**: the `pdl=True` argument exists on flashinfer's `mm/bmm` wrappers, but
  CUTLASS and cuBLASLt backends explicitly raise "does not support PDL — use the TGV backend"
  (`gemm_base.py` lines ~298-334). TGV (TinyGEMM) is the sm100-family small-M GEMV-shaped backend
  and the SM120/SM121 kernel "supports batch operations natively" — it is the PDL-capable route
  for tiny-M bf16 GEMMs if a layer ever routes there. Expected gain class: launch-latency
  hiding at M<=8, where each kernel is microseconds; % e2e depends on how many small GEMMs sit
  on the step path. Evidence: flashinfer source above; CUTLASS PDL example
  https://docs.nvidia.com/cutlass/4.5.2/CHANGELOG.html ("Added PDL support along with example
  Kernel launch with Programmatic Dependent Launch").
- **PDL elsewhere in vLLM**: release notes mention "programmatic dependent launch for the DSA
  decode" and FlashInfer XQA decode support on SM12x (#49718)
  (https://github.com/vllm-project/vllm/releases). Known bug class: PDL + LoRA on SM100 →
  `VLLM_LORA_DISABLE_PDL` env exists in our build's `vllm/envs.py` (not relevant to us, no LoRA).
- **MXFP8/NVFP4 grouped GEMM on sm_121**: our tuned `mxfp8_gemm` + `trtllm` MoE ops (topic 3
  cache) are the shipping sm12x grouped-GEMM path; DeepGEMM-style sm120 kernels are not
  available to this stack today. Expected-gain verdict: nothing actionable beyond keeping
  autotune on; the honest estimate for adding PDL anywhere it is missing is single-digit % at
  small batch, unmeasured on GB10.

## 6. Memory: THP/hugepages, torch allocator envs, malloc arenas and UVM

- **THP**: currently `madvise` + `defrag=madvise` (live read) — this is the recommended setting
  for latency-sensitive services; `always` risks compaction stalls exactly during long runs
  (the classic latency-spike source; see kernel doc
  https://docs.kernel.org/admin-guide/mm/transhuge.html and the debate summarized at
  https://news.ycombinator.com/item?id=34439387). The PLE table is a file-backed mmap — page
  cache pages, not anonymous THP — so THP mode does not apply to it. Do NOT flip to `always`.
  Possible micro-lever: `madvise(MADV_HUGEPAGE)` on the model-weight arena (torch already maps
  large blocks; on 64KiB-kernel-page ARM the win is smaller than on 4KiB x86). Unmeasured here;
  low priority.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: documented real-world GB10 fix —
  vision-encoder fragmentation froze a whole Spark until earlyoom killed it; the setting fixed
  it (https://forums.developer.nvidia.com/t/longcat-next/372494 post #2; also recommended in
  https://www.spheron.network/blog/nvidia-dgx-spark-gpu-cloud-pipeline/). Our serving lane is
  steady-shape decode, so the fragmentation risk is lower, but the setting also returns freed
  segments more gracefully on unified memory. Expected gain: stability under long runs (our C4
  "dips during long runs" are more likely topic-1's flip, but allocator fragmentation is the
  other classic slow-onset degradation). Risk: none observed; PyTorch-supported since 2.1.
- **glibc arenas**: `MALLOC_ARENA_MAX` unset; with 20 cores glibc defaults to 8x20=160 arenas,
  which multiplies RSS and worsens fragmentation for the many-threaded API server. Standard
  latency-server practice is `MALLOC_ARENA_MAX=2` (+ optionally `MALLOC_TRIM_THRESHOLD_` /
  `MALLOC_MMAP_THRESHOLD_`). No GB10-specific measurement exists; evidence class is general
  server-tuning practice (glibc malloc docs; https://ohyaan.github.io/tips/... for affinity-
  adjacent jitter). Risk: none (worst case slightly more mmap churn).
- **UVM interplay**: GB10's single LPDDR5x pool means host malloc pressure competes with GPU
  residency; large host arenas + page-cache growth (PLE table) shrink effective GPU headroom.
  The start.sh preflight (~103 GiB MemAvailable) already guards this. `expandable_segments`
  + arena cap + keeping the PLE table page-cached-but-not-resident (MADV_RANDOM, already
  shipped) is the right triad.

## 7. PCIe/NVMe PLE table reads: fadvise vs mmap MADV_RANDOM vs preadv/io_uring

- **Already shipped and measured on this stack** (upstream commit e33a9f8, in our build):
  the PLE row gather is a genuine single-threaded minor-fault loop — a 280-row cold gather took
  20.12 ms (71.4 us/fault) and did NOT improve with more torch threads (1->17.39 ms, 4->16.35,
  8->16.34 ms), proving serial fault processing. Fix shipped: `_ple_prefetch_rows()` maps row
  ids -> page offsets, dedups with `torch.unique` (so reads issue in ascending file order), and
  calls `os.posix_fadvise(fd, page<<12, 4096, POSIX_FADV_WILLNEED)` through an fd kept beside
  the packed mmap, before `torch.index_select`. Result: 280-row gather 20.12 -> **1.51 ms
  (13x)**; end-to-end **-3.2% mean step time across 1/2/4/5/6/8 streams, 6 of 6 improving**,
  with the page-cache confound ruled out (patched run faulted MORE pages and was still faster).
  Only ~20% of lookups miss the page cache at steady state (5.51 pages/token at 1 stream).
- **mmap + MADV_RANDOM** (our current base) remains correct: it disables readahead for
  non-touched pages and avoids polluting the cache with 2-of-3 unused rows per 4 KiB page.
  Kernel-page-cache reference: https://kernel-internals.org/mm/page-cache/ (mmap vs read
  trade-offs), https://biriukov.dev/docs/page-cache/3-page-cache-and-basic-file-operations/
  (readahead behavior).
- **io_uring/preadv instead of mmap**: no measurement exists for this workload, and the fadvise
  prefetch already converted the fault loop into kernel-side batched reads. Switching to
  explicit async reads would remove minor-fault cost entirely but adds a copy and a completion
  path on the critical handshake; upstream measured the mmap+fadvise approach and stopped there.
  Verdict: not worth it now; revisit only if /proc/vmstat minor-fault rates during decode are
  still high (check `pgmajfault`/`pgminorfault` deltas per generated token).
- **Ordering matters**: the dedup->ascending-order trick is why the prefetch is fast (sequential
  NVMe queues, no seeky pattern). Keep it if the patch is ever edited.

## 8. Docker/GC tweaks: ipc host, cgroup pinning, IRQ affinity

- **`--ipc host`**: already used (required for CUDA shared memory in this image class); no
  further IPC lever.
- **cpuset pinning of the API server + EngineCore to X925**: `--cpuset-cpus=5-9,15-19` on the
  container (or `docker update --cpuset-cpus` on the running container without restart — though
  task rules here forbid touching it, so this is for the next planned relaunch). Design
  reference: vLLM CPU-binding doc
  https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html
  and upstream `numa_bind_cpus` config
  https://docs.vllm.ai/en/latest/api/vllm/config/parallel/. Expected gain: removes cross-cluster
  migration jitter (X925 is 1.39x the clock of A725 and the PLE fault loop is single-threaded,
  so a migration to A725 mid-gather costs directly). Risk: none; single NUMA node so no NUMA
  penalty. Note: the flip-issue authors verified 0% busy time on A725 already via scheduler
  sampling, so treat this as jitter insurance, not a fix.
- **IRQ affinity**: irqbalance is inactive; NVMe queues are already affined 1:1 to CPUs 0-5
  (A725 side, verified in /proc/interrupts). This is actually a good layout: NVMe completions
  for PLE prefetch land on A725 cores while X925 runs the serving threads — leave as-is.
  If ever revised, the ARM references are:
  https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/spdk-nvme-over-tcp-optimization-on-arm
  (irqbalance vs manual affinity on Arm servers),
  https://blog.cloudflare.com/how-to-achieve-low-latency/ (manual IRQ spreading for latency),
  https://docs.kernel.org/admin-guide/kernel-parameters.html (`irqaffinity=` boot default).
- **GC / allocator in-container**: python GC pauses in the API server are a known vLLM jitter
  source; the standard mitigations (already common in vLLM images) are `VLLM_API_SERVER` worker
  count 1 and relying on vLLM's internal buffer reuse. `MALLOC_ARENA_MAX=2` (topic 6) belongs
  here too. No further docker-level lever beyond cpuset; `--shm-size` is covered by `--ipc host`.
- **Zombie-process hygiene** (from campaign ops notes, affects long-run stability measurements):
  after stopping experiment containers, `docker rm -f` then `pkill -f 'VLLM::'` and require
  ~103 GiB MemAvailable before relaunch — stale workers holding ~7 G are the classic cause of
  "degraded" long-run numbers that are actually memory-pressure artifacts.

---

## IMMEDIATE sysfs/env commands safe to try (ranked)

Ranked by (expected gain x confidence) / risk. All are reversible without a reboot except the
platform update (last). None were executed during this research session.

1. **Backup + verify the flashinfer tactics cache is being hit** (zero risk, protects topic 3):
   `docker exec vllm-fn-ab1 cat /root/.cache/vllm/flashinfer_autotune_cache/0.6.17/121a/65158578d47e366d67a4456a0cc6cb3084501340b9d3c356eac50e9867492e99/autotune_configs.json > ~/flashnext-spark/research4/tactics-121a-backup.json`
   then `docker logs vllm-fn-ab1 2>&1 | grep -c 'Config cache hit'` on next boot.
2. **AC power-cycle the brick before the next long C4/C8 soak** (unplug from wall 60 s; clears
   latched USB-PD pl1/syspl1 caps that NVML cannot see). Expected: removes the 3x-class dip
   mode if present. Risk: none. Evidence: topic 1 forum thread #5/#8/#10/#11.
3. **Disable LPI-1 on X925 cores only** (jitter; the last remaining idle state on the serving
   cores; fully reversible):
   `for c in 5 6 7 8 9 15 16 17 18 19; do echo 1 | sudo tee /sys/devices/system/cpu/cpu$c/cpuidle/state1/disable >/dev/null; done`
   (undo with `echo 0`). Expected: small p99 step-time improvement; mechanism-based, our LPI-3
   result is the closest measurement. Risk: a few W idle power.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in the container env at next planned
   relaunch (`-e` in EXTRA_DOCKER_ARGS). Expected: long-run stability against fragmentation on
   unified memory; documented GB10 fix (Longcat thread). Risk: none observed.
5. **`MALLOC_ARENA_MAX=2`** (and optionally `MALLOC_TRIM_THRESHOLD_=131072`) as container env.
   Expected: less RSS growth/fragmentation over long runs; general practice, unmeasured here.
6. **Pin the container to X925 cores** at next relaunch: add `--cpuset-cpus=5-9,15-19`.
   Expected: removes big.LITTLE migration jitter from the PLE fault loop. Risk: none.
7. **Install spark_hwmon (or hoesing/spark-gpu-throttle-check) and log pl1/syspl1 + power during
   a C4 soak** to catch the hidden slow state red-handed (reads only; DKMS build risk on this
   kernel is the only concern). Evidence: forum thread #8/#10.
8. **A/B `nvidia-smi -lgc 1800,3003`** (raise DVFS floor, keep boost) for one soak, with
   `nvidia-smi -rgc` to undo. Expected: unknown — the only measured lock (`0,2200`) is a power
   cap that LOWERS matmul boost ~9%; do not adopt that one. Risk: unmeasured thermals.
9. **Keep as-is (verified correct today):** persistence mode on, cpufreq `performance`,
   THP `madvise`, `CUDAGRAPH_CAPTURE_SIZES=auto`, `MTP_K_SCHEDULE` empty, irqbalance off with
   NVMe IRQs on A725 cores, `--ipc host`.
10. **Plan the platform update** to BIOS 5.36_0ACUM027 + kernel 6.17.0-1032-nvidia + driver
    580.173.02 (the 8-node fleet combination with zero slow-state seconds). Highest expected
    gain for the "C4 dips during long runs" goal, but needs a maintenance window and full
    re-benchmark; also invalidates the flashinfer autotune metadata (driver/cublas versions) —
    expect a fresh autotune pass on first boot.

## Source index

- GB10 fast/slow flip (63 vs 94 ms/step; reproduction script; ruled-out table; 8-node fleet):
  https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1
- 513 MHz USB-PD cap, spbm pl1/syspl1 registers, power-cycle fix, throttle-check tools:
  https://forums.developer.nvidia.com/t/investigating-513mhz-cap-for-gpu/361296 ;
  https://github.com/hoesing/spark-gpu-throttle-check ;
  https://github.com/parallelArchitect/spark-gpu-throttle-check ;
  https://github.com/antheas/spark_hwmon
- flashinfer 0.6.17 autotuner API/format/metadata gate: installed source, read live in
  `vllm-fn-ab1:/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner/autotuner.py`;
  vLLM wrapper: `.../vllm/model_executor/warmup/kernel_warmup.py` and
  `.../vllm/model_executor/warmup/flashinfer_autotune_cache.py`
- vLLM CUDA graphs design (modes, uniform decode = 1+num_spec_tokens):
  https://docs.vllm.ai/en/stable/design/cuda_graphs/
- Capture-size + PLE fadvise measurements on this stack: git commit e33a9f8 in
  ~/flashnext-spark (upstream MiaAI-Lab, merged 2026-09-08)
- FULL_DECODE_ONLY+MTP graph replay crash (Ascend): https://github.com/vllm-project/vllm-ascend/issues/8587
- DeepGEMM archs + PDL PR: https://github.com/deepseek-ai/DeepGEMM ; PDL in CUTLASS:
  https://docs.nvidia.com/cutlass/4.5.2/CHANGELOG.html ; vLLM PDL/XQA SM12x notes:
  https://github.com/vllm-project/vllm/releases
- expandable_segments GB10 fix: https://forums.developer.nvidia.com/t/longcat-next/372494 ;
  https://www.spheron.network/blog/nvidia-dgx-spark-gpu-cloud-pipeline/
- THP: https://docs.kernel.org/admin-guide/mm/transhuge.html ; page cache / readahead / mmap vs
  read: https://kernel-internals.org/mm/page-cache/ ;
  https://biriukov.dev/docs/page-cache/3-page-cache-and-basic-file-operations/
- cpuidle governors / menu / teo: https://docs.kernel.org/admin-guide/pm/cpuidle.html ;
  boot-param isolation: https://docs.kernel.org/admin-guide/kernel-parameters.html ;
  cpupower idle-set:
  https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/index
- CPU isolation/affinity practice:
  https://ohyaan.github.io/tips/cpu_isolation_and_task_affinity_for_multicore_optimization/ ;
  vLLM CPU binding: https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html
  ; numa_bind_cpus: https://docs.vllm.ai/en/latest/api/vllm/config/parallel/
- IRQ affinity on ARM: https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/spdk-nvme-over-tcp-optimization-on-arm ;
  https://blog.cloudflare.com/how-to-achieve-low-latency/
- Local measurements: all "live" facts above were read directly from gx10-a416 sysfs,
  /proc, nvidia-smi, docker logs, and the container filesystem on 2026-09-12 (read-only).
