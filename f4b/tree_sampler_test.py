"""CPU reference test for tree_greedy_sample — verifies greedy semantics."""
import torch
import sys
sys.path.insert(0, '/m')

def tree_greedy_reference(draft, targ, bonus, parent, depth):
    """Pure-python reference of the tree greedy accept."""
    n = len(draft)
    matched = [draft[i] == int(targ[i]) for i in range(n)]
    accepted = [False] * n
    for i in range(n):
        ok = matched[i]
        p = parent[i]
        while p >= 0 and ok:
            ok = ok and matched[p]
            p = parent[p]
        accepted[i] = ok
    best, bd = -1, -1
    for i in range(n):
        if accepted[i] and depth[i] > bd:
            bd, best = depth[i], i
    out = []
    if best < 0:
        return [int(targ[0])]
    path = []
    cur = best
    while cur >= 0:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    for node in path:
        out.append(int(targ[node]))
    out.append(int(bonus[0]))
    return out

# tree: root(0) -> [1, 2]; 1 -> [3, 4]; 2 -> 5
parent = [-1, 0, 0, 1, 1, 2]
depth  = [ 0, 1, 1, 2, 2, 2]
# draft tokens; target argmax designed so path root->1->3 fully matches
draft = [10, 20, 99, 30, 40, 50]
targ  = [10, 20, 11, 30, 41, 12]   # nodes 0,1,3 match; 2,4,5 don't
bonus = [77]
ref = tree_greedy_reference(draft, targ, bonus, parent, depth)
print('reference output (expect [10, 20, 30, 77]):', ref)

# kernel run (cuda) with same data
from tree_sampler import tree_greedy_sample
d = torch.tensor(draft, dtype=torch.int32, device='cuda')
t = torch.tensor(targ, dtype=torch.int64, device='cuda')
b = torch.tensor(bonus, dtype=torch.int32, device='cuda')
p = torch.tensor(parent, dtype=torch.int32, device='cuda')
dep = torch.tensor(depth, dtype=torch.int32, device='cuda')
ndt = torch.tensor([6], dtype=torch.int32, device='cuda')
out = tree_greedy_sample(d, t, b, p, dep, ndt, max_spec_len=3)
got = [x for x in out[0].tolist() if x != -1]
print('kernel output:', got)
print('PASS' if got == ref else 'FAIL')

# case 2: nothing matches at root
targ2 = [55, 20, 11, 30, 41, 12]
ref2 = tree_greedy_reference(draft, targ2, bonus, parent, depth)
out2 = tree_greedy_sample(d, torch.tensor(targ2, dtype=torch.int64, device='cuda'),
                          b, p, dep, ndt, max_spec_len=3)
got2 = [x for x in out2[0].tolist() if x != -1]
print('case2 reference (expect [55]):', ref2, '| kernel:', got2)
print('PASS2' if got2 == ref2 else 'FAIL2')
