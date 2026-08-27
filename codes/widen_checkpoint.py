"""
widen_checkpoint.py — function-preserving widen of a residual-actor checkpoint for the
history + shared-reference observation (obs 44 -> 98):
    [0:44]   original gt2 obs (load est, own, f_g, f_lam, clock)
    [44:80]  2 delta-history frames  (36)  -> mean 0 / std 1 (small, zero-mean deltas)
    [80:98]  shared reference anchor (18)  -> REAL mean/std (large, varying; p_d to ~11m,
             constant on the line -> mean0/std1 would reproduce the old desired-state norm bug)

The ONLY tensor that changes is the actor's first Linear (`body.0.weight`): 54 ZERO columns are
appended, so the widened net computes the IDENTICAL map as gt2 at init (fully preserved); the 54 new
inputs get recruited by gradient during training. The reference block's mean/std is computed
ANALYTICALLY from get_reference_trajectory over the full training-trajectory distribution (it's a
deterministic fn of (t, traj) — no plant rollout needed). Critic untouched (true global state input).

Run:  python widen_checkpoint.py
"""
import numpy as np
import torch

from networks import Actor
from controller import get_reference_trajectory
from expert_reference import training_pairs
from collect_il_data import DT, T_END

SRC = "residual_mappo_gt2.pt"
DST = "residual_mappo_gt2_wide.pt"
BASE, HIST, REF = 44, 36, 18
NEW = HIST + REF                      # 54 appended input dims

# --- 1. reference-block mean/std over the WHOLE training-traj distribution (analytic, instant) ---
pairs, n_anchor = training_pairs()    # [(traj_callable, expert_dpos)] x 56 (traj=None -> default line)
ts = np.arange(0.0, T_END, DT)
buf = []
for traj, _ in pairs:
    for t in ts:
        pd, vd, Rd, wd = get_reference_trajectory(float(t), traj)     # (p, v, R, omega)
        buf.append(np.concatenate([np.asarray(pd, float), np.asarray(Rd, float).flatten(order="C"),
                                   np.asarray(vd, float), np.asarray(wd, float)]))
arr = np.asarray(buf, dtype=np.float32)                               # (56*3500, 18)
ref_mean = arr.mean(0, keepdims=True)
ref_std = (arr.std(0, keepdims=True) + 1e-6).astype(np.float32)
print(f"reference block from {len(pairs)} trajs x {len(ts)} steps: "
      f"std range [{ref_std.min():.4f}, {ref_std.max():.4f}]  (const-on-line dims get real quintic spread)")

# --- 2. widen the checkpoint (zero-pad first layer, extend norm) ---
ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
W = sd["body.0.weight"]               # (hidden0, 44)
h0, old = W.shape
assert old == BASE, f"expected src obs {BASE}, got {old}"
Wn = torch.zeros(h0, old + NEW, dtype=W.dtype)
Wn[:, :old] = W                       # first 44 cols = untouched gt2; last 54 = 0 -> identical map at init
sd["body.0.weight"] = Wn

hist_mean = np.zeros((1, HIST), np.float32)
hist_std = np.ones((1, HIST), np.float32)
ck["obs_dim"] = old + NEW
ck["obs_mean"] = np.concatenate([ck["obs_mean"], hist_mean, ref_mean], axis=1)   # (1,98)
ck["obs_std"] = np.concatenate([ck["obs_std"], hist_std, ref_std], axis=1)
torch.save(ck, DST)
print(f"widened {SRC} (obs {old}) -> {DST} (obs {old + NEW}); body.0.weight {tuple(Wn.shape)}")

# --- 3. verify identical map: same first-`old` obs -> identical output, any history/reference values ---
hidden = tuple(ck.get("hidden", (128, 128)))
src = torch.load(SRC, map_location="cpu", weights_only=False)
a_old = Actor(obs_dim=old, act_dim=src["act_dim"], hidden=hidden)
a_old.load_state_dict(src["state_dict"]); a_old.eval()
a_new = Actor(obs_dim=old + NEW, act_dim=ck["act_dim"], hidden=hidden)
a_new.load_state_dict(sd); a_new.eval()
rng = np.random.default_rng(0)
x_old = rng.standard_normal((64, old)).astype(np.float32)
x_new = np.concatenate([x_old, rng.standard_normal((64, NEW)).astype(np.float32)], axis=1)
with torch.no_grad():
    y_old = a_old(torch.tensor(x_old)).numpy()
    y_new = a_new(torch.tensor(x_new)).numpy()
max_diff = float(np.abs(y_old - y_new).max())
print(f"max |delta output| over 64 obs (arbitrary new dims): {max_diff:.2e}")
assert max_diff < 1e-6, "widened map DIFFERS from source — not function-preserving!"
print("OK — widened actor is function-identical to gt2 on the original 44 inputs.")
