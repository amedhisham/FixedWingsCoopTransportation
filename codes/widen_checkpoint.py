"""
widen_checkpoint.py — function-preserving widen of a residual-actor checkpoint for the
history-augmented observation (obs 44 -> 80; +2 delta frames x 18 = 36 new inputs).

The ONLY tensor that changes is the actor's first Linear (`body.0.weight`): we append 36
ZERO columns. New inputs therefore contribute nothing at init, so the widened net computes
the IDENTICAL map as the source (gt2 fully preserved), while the 36 delta-history inputs
start at zero weight and get recruited by gradient during training. obs_mean/obs_std are
extended by 36 (mean 0, std 1): moot at init (weight is 0), and well-scaled once learned
since the deltas are small, zero-mean quantities. The critic is untouched (its input is the
true global state, which history does not change).

Run:  python widen_checkpoint.py
"""
import numpy as np
import torch

from networks import Actor

SRC = "residual_mappo_gt2.pt"
DST = "residual_mappo_gt2_wide.pt"
NEW = 36                                   # appended history dims (2 delta frames x 18)

ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
W = sd["body.0.weight"]                     # (hidden0, old_obs)
h0, old = W.shape
Wn = torch.zeros(h0, old + NEW, dtype=W.dtype)
Wn[:, :old] = W                             # first `old` columns = untouched gt2 weights
sd["body.0.weight"] = Wn                     # last NEW columns = 0  -> identical map at init
# bias + every other layer left exactly as-is.

ck["obs_dim"] = old + NEW
ck["obs_mean"] = np.concatenate([ck["obs_mean"], np.zeros((1, NEW), dtype=np.float32)], axis=1)
ck["obs_std"] = np.concatenate([ck["obs_std"], np.ones((1, NEW), dtype=np.float32)], axis=1)
torch.save(ck, DST)
print(f"widened {SRC} (obs {old}) -> {DST} (obs {old + NEW}); body.0.weight {tuple(Wn.shape)}")

# --- verify identical map: same first-`old` obs -> identical output, for ANY new-dim values ---
hidden = tuple(ck.get("hidden", (128, 128)))
src = torch.load(SRC, map_location="cpu", weights_only=False)   # reload pristine source
a_old = Actor(obs_dim=old, act_dim=src["act_dim"], hidden=hidden)
a_old.load_state_dict(src["state_dict"]); a_old.eval()
a_new = Actor(obs_dim=old + NEW, act_dim=ck["act_dim"], hidden=hidden)
a_new.load_state_dict(sd); a_new.eval()

rng = np.random.default_rng(0)
x_old = rng.standard_normal((64, old)).astype(np.float32)
x_new = np.concatenate([x_old, rng.standard_normal((64, NEW)).astype(np.float32)], axis=1)   # arbitrary history
with torch.no_grad():
    y_old = a_old(torch.tensor(x_old)).numpy()
    y_new = a_new(torch.tensor(x_new)).numpy()
max_diff = float(np.abs(y_old - y_new).max())
print(f"max |delta output| over 64 obs (arbitrary history dims): {max_diff:.2e}")
assert max_diff < 1e-6, "widened map DIFFERS from source — not function-preserving!"
print("OK — widened actor is function-identical to the source on the original 44 inputs.")
