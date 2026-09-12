import os
# Shrink the CUDA context before torch initializes it (0.94-util server coexists).
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import torch
import sys
sys.path.insert(0, '/m')

torch.manual_seed(42)
dev = 'cuda'
HV, K, V = 4, 128, 128

from fused_sigmoid_gating_tree import fused_sigmoid_gating_delta_rule_update as f_tree

def make_inputs(T):
    q = torch.randn(1, T, HV, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, HV, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(1, T, HV, device=dev, dtype=torch.float32)
    b = torch.randn(1, T, HV, device=dev, dtype=torch.float32)
    return q, k, v, a, b

A_log = torch.randn(HV, device=dev, dtype=torch.float32)
dt_bias = torch.randn(HV, device=dev, dtype=torch.float32)
h_all = torch.zeros(32, HV, V, K, device=dev, dtype=torch.float32)
h_all[7] = torch.randn(HV, V, K, device=dev) * 0.05

qA, kA, vA, aA, bA = make_inputs(3)
qB, kB, vB, aB, bB = make_inputs(3)
# flatten to [1, 6, ...] — varlen API requires batch dim 1
q6 = torch.cat([qA, qB], dim=1); k6 = torch.cat([kA, kB], dim=1); v6 = torch.cat([vA, vB], dim=1)
a6 = torch.cat([aA, aB], dim=1); b6 = torch.cat([bA, bB], dim=1)
cu6 = torch.tensor([0, 3, 6], dtype=torch.int32, device=dev)
idx6 = torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.int32, device=dev)
init_idx = torch.tensor([7, 7], dtype=torch.int32, device=dev)

o_tree, _ = f_tree(A_log, a6, b6, dt_bias, q6, k6, v6,
                   initial_state=h_all, inplace_final_state=True,
                   cu_seqlens=cu6, ssm_state_indices=idx6,
                   num_accepted_tokens=torch.tensor([3, 3], dtype=torch.int32, device=dev),
                   initial_state_indices=init_idx,
                   use_qk_l2norm_in_kernel=True)

def run_chain(qT, kT, vT, aT, bT, start_slot, save_slots):
    Tn = qT.shape[1]
    idx = torch.zeros(1, Tn, dtype=torch.int32, device=dev)
    idx[0, 0] = start_slot
    for t in range(1, Tn):
        idx[0, t] = save_slots[t]
    out, _ = f_tree(A_log, aT, bT, dt_bias, qT, kT, vT,
                    initial_state=h_all, inplace_final_state=True,
                    cu_seqlens=torch.tensor([0, Tn], dtype=torch.int32, device=dev),
                    ssm_state_indices=idx,
                    use_qk_l2norm_in_kernel=True)
    return out

o_refA = run_chain(qA, kA, vA, aA, bA, 7, [0, 20, 21])
o_refB = run_chain(qB, kB, vB, aB, bB, 7, [0, 22, 23])

diffA = (o_tree[0, :3].float() - o_refA[0].float()).abs().max().item()
diffB = (o_tree[0, 3:6].float() - o_refB[0].float()).abs().max().item()
print(f'TREE vs CHAIN sibling A max-diff: {diffA:.2e}')
print(f'TREE vs CHAIN sibling B max-diff: {diffB:.2e}')
normA0 = h_all[10].norm().item(); normB0 = h_all[13].norm().item()
print(f'snapshot norms: slot10 {normA0:.3f}, slot13 {normB0:.3f}')
ok = diffA < 2e-2 and diffB < 2e-2 and normA0 > 0 and normB0 > 0
print('PASS' if ok else 'FAIL')
