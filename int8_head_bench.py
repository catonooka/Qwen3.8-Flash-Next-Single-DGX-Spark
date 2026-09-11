import torch, time, glob, json

dev = "cuda"
SNAP = glob.glob("/root/.cache/huggingface/hub/models--drowzeys--keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only/snapshots/*")[0]

# find the lm_head shard via index
idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))
wname = [k for k in idx["weight_map"] if k.endswith("lm_head.weight")][0]
shard = idx["weight_map"][wname]
print("tensor:", wname, "shard:", shard)

from safetensors import safe_open
with safe_open(f"{SNAP}/{shard}", framework="pt", device="cpu") as f:
    Wbf_cpu = f.get_tensor(wname)  # bf16 [248320, 2560] on CPU
V, H = Wbf_cpu.shape
print("loaded", Wbf_cpu.shape, Wbf_cpu.dtype)

# quantize on CPU (avoid GPU fp32 peak), then move both to GPU
scale_cpu = (Wbf_cpu.abs().amax(dim=1).float() / 127.0)
Wi8_cpu = (Wbf_cpu.float() / scale_cpu[:, None]).round().clamp(-127, 127).to(torch.int8)
del scale_cpu
Wbf = Wbf_cpu.to(dev)          # 1.27 GB
Wi8 = Wi8_cpu.to(dev)          # 0.63 GB
scale = (Wbf_cpu.abs().amax(dim=1).float() / 127.0).contiguous().to(dev)
del Wi8_cpu, Wbf_cpu
torch.cuda.empty_cache()
print("on-GPU: BF16 + INT8 heads resident")

torch.manual_seed(0)
B = 4
h = (torch.randn(B, H, device=dev, dtype=torch.float32) * 8.0).to(torch.bfloat16)  # |h| ~ sqrt(2560)*rms class

def bench(fn, n=30, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

# 1) stock BF16 GEMV (the verify lm_head read)
t_bf = bench(lambda: torch.nn.functional.linear(h, Wbf))
print(f"BF16 F.linear: {t_bf:.3f} ms = {V*H*2/(t_bf/1000)/1e9:.0f} GB/s effective")

# 2) INT8 per-row (already quantized above), torch._int_mm path
hf = h.float()
ascale = hf.abs().amax(dim=1, keepdim=True) / 127.0
hi8 = torch.round(hf / ascale).clamp(-127, 127).to(torch.int8)
# torch._int_mm needs M>16 (strictly): pad to 32 rows
PAD = 32
hi8p = torch.zeros(PAD, H, device=dev, dtype=torch.int8); hi8p[:B] = hi8
ascp = torch.ones(PAD, 1, device=dev); ascp[:B] = ascale
def int8_path():
    acc = torch._int_mm(hi8p, Wi8.t())  # [16, V] int32
    return acc.float()[:B] * ascale[:B] * scale[None, :]
try:
    t_i8 = bench(int8_path)
    print(f"INT8 _int_mm:  {t_i8:.3f} ms = {V*H/(t_i8/1000)/1e9:.0f} GB/s (W bytes)")
except Exception as e:
    print("int_mm failed:", repr(e)); t_i8 = None

# 3) census: BF16-top1 inside INT8-top64? margins?
with torch.no_grad():
    lg_bf = torch.nn.functional.linear(h, Wbf).float()
    am_bf = lg_bf.argmax(-1)
    if t_i8 is not None:
        lg_i8 = int8_path()
        top64 = lg_i8.topk(64, dim=-1).indices
        in64 = (top64 == am_bf.unsqueeze(1)).any(dim=1).float().mean().item()
        # exact rescore of top-64
        w64 = Wbf[top64.reshape(-1)].view(B, 64, H)
        hh = h.unsqueeze(1).expand(-1, 64, -1)
        ex = torch.einsum('bkh,bkh->bk', hh.float(), w64.float())
        pick = top64.gather(1, ex.argmax(-1, keepdim=True)).squeeze(1)
        match = (pick == am_bf).float().mean().item()
        t2 = lg_bf.topk(2, dim=-1).values
        margins = (t2[:, 0] - t2[:, 1])
        print(f"BF16-top1 in INT8-top64: {in64*100:.2f}%")
        print(f"argmax match after rescore: {match*100:.2f}%")
        print(f"margins min/median: {margins.min():.4f} / {margins.median():.4f}")

# 4) rescore cost
def rescore(top64):
    w64 = Wbf[top64.reshape(-1)].view(B, 64, H)
    hh = h.unsqueeze(1).expand(-1, 64, -1)
    return torch.einsum('bkh,bkh->bk', hh.float(), w64.float())
top64r = torch.randint(0, V, (B, 64), device=dev)
t_rs = bench(lambda: rescore(top64r))
print(f"top-64 BF16 rescore: {t_rs:.3f} ms")
print(f"projected verify-head time: {t_i8 if t_i8 else 'NA'} + {t_rs:.3f} vs {t_bf:.3f} BF16")
