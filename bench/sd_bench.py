#!/usr/bin/env python3
"""sparkDash bench client: engine-level C1/C4/C8 tok/s via the local sparkDash API.

Usage: python3 sd_bench.py --label X [--concurrency 1,4,8] [--prompt prose|code] [--reps 2]
Appends results to sd_results.jsonl. Measures engine steady-state decode tps
(sparkDash protocol: 32-tok warmup, completion-tokens clock-stop).
"""
import argparse, json, time, urllib.request

API = "http://localhost:5555/api/sparks/spark-1/llm/bench"

def bench_one(conc, prompt_type, max_tokens=600):
    body = json.dumps({"port": 8888, "concurrencies": [conc], "maxTokens": max_tokens,
                       "promptType": prompt_type}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        bid = json.loads(r.read())["benchId"]
    for _ in range(200):
        time.sleep(3)
        with urllib.request.urlopen(f"{API}/{bid}", timeout=30) as r:
            st = json.loads(r.read())
        if st["status"] == "completed":
            return st["results"][0], bid
        if st["status"] == "failed":
            raise RuntimeError(f"bench {bid} failed: {json.dumps(st)[:500]}")
    raise TimeoutError(f"bench {bid} never completed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--prompt", default="prose")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", default="/home/nmt/flashnext-spark/sd_results.jsonl")
    args = ap.parse_args()

    for rep in range(args.reps):
        for pt in args.prompt.split(","):
            for c in [int(x) for x in args.concurrency.split(",")]:
                r, bid = bench_one(c, pt, args.max_tokens)
                rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label,
                       "rep": rep, "prompt": pt, "conc": c,
                       "agg_tps": r.get("aggregateDecodeTps"),
                       "mean_tps": r.get("meanDecodeTps"),
                       "p95_tps": r.get("p95DecodeTps"),
                       "ttft_ms": r.get("meanTtftMs"), "benchId": bid}
                print(json.dumps(rec), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
