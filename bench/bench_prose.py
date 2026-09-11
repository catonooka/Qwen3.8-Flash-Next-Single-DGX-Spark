#!/usr/bin/env python3
"""Prose-focused single-stream decode benchmark (Trí's C1-prose lane).

8 prose prompts, temperature 0, thinking OFF, N repeats. Reports per-prompt
and mean tok/s plus the spec-decode acceptance window from the server log.
Appends one JSON line per label to prose_results.jsonl.
"""
import argparse, atexit, json, os, re, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lock

bench_lock.acquire()
atexit.register(bench_lock.release)

PROMPTS = [
    ("prose-nightmarket", "Write a vivid 320-word travel essay about a night market in Hanoi.", 350),
    ("prose-rivers", "Write a reflective 320-word essay on why rivers shape civilizations.", 350),
    ("prose-letter", "Write a heartfelt 320-word letter from a grandmother to her granddaughter leaving for college.", 350),
    ("prose-review", "Write a 320-word thoughtful review of a small family-run noodle shop.", 350),
    ("prose-story", "Write a 320-word short story about a lighthouse keeper who finds a message in a bottle.", 350),
    ("prose-describe", "Describe in 320 words the experience of watching a thunderstorm roll across rice paddies.", 350),
    ("prose-argument", "Argue in 320 words that public libraries are among civilization's greatest inventions.", 350),
    ("prose-memoir", "Write a 320-word memoir fragment about learning to ride a motorbike at forty.", 350),
]

def chat(url, model, content, max_tokens):
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    u = out.get("usage", {})
    return dt, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)

def acceptance_snapshot(container):
    """Mean acceptance length + per-position rates from the last SpecDecoding line."""
    try:
        log = subprocess.run(["docker", "logs", "--tail", "3", container],
                             capture_output=True, text=True, timeout=10).stderr
        lines = [l for l in log.splitlines() if "SpecDecoding metrics" in l]
        if not lines:
            return None
        m = re.search(r"Mean acceptance length: ([\d.]+)", lines[-1])
        p = re.search(r"Per-position acceptance rate: ([\d., ]+)", lines[-1])
        return {"mean_accept": float(m.group(1)) if m else None,
                "per_pos": p.group(1).strip() if p else None}
    except Exception as e:
        return {"error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="/home/nmt/flashnext-spark/prose_results.jsonl")
    ap.add_argument("--container", default="vllm-fn-ab1")
    args = ap.parse_args()

    # warmup (JIT shapes)
    chat(args.url, args.model, "Hello.", 16)

    rates = {}
    for rep in range(args.repeats):
        for name, prompt, mt in PROMPTS:
            dt, ct, pt = chat(args.url, args.model, prompt, mt)
            tps = ct / dt if dt > 0 else 0.0
            rates.setdefault(name, []).append((tps, ct, dt))
            print(f"[{args.label}] rep{rep} {name}: {tps:.1f} tok/s ({ct} tok in {dt:.1f}s)", flush=True)

    per_prompt = {k: sum(r for r, _, _ in v) / len(v) for k, v in rates.items()}
    overall = sum(sum(r for r, _, _ in v) / len(v) for v in rates.values()) / len(rates)
    acc = acceptance_snapshot(args.container)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label,
           "mean_tok_s": round(overall, 2),
           "per_prompt": {k: round(v, 2) for k, v in per_prompt.items()},
           "acceptance": acc, "repeats": args.repeats}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
