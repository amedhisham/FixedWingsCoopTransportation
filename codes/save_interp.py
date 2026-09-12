"""save_interp.py — save the alpha=ALPHA interpolation between CKPT_A and CKPT_B as a valid warm-start
checkpoint (matches mappo.save_ckpt format). Used 2026-09-12 to bank the FREE -0.380 net (better than the
150+60-iter grind's -0.385) found by interp_search.py at alpha=0.75. See [[f2-rotation-limited]].
Uses CKPT_A's obs-norm (what the line-search eval scored with). Run:  python -u save_interp.py"""
import torch

A = "residual_mappo_overfit_xyz_2trj_ch.pt"   # start (~ -0.392)
B = "residual_mappo_overfit.pt"               # end   (~ -0.383)
OUT = "residual_mappo_overfit_interp075.pt"
ALPHA = 0.75

ckA = torch.load(A, map_location="cpu", weights_only=False)
ckB = torch.load(B, map_location="cpu", weights_only=False)


def lerp(sdA, sdB, a):
    out = {}
    for k in sdA:
        out[k] = (sdA[k].float() + a * (sdB[k].float() - sdA[k].float())).to(sdA[k].dtype)
    return out


actor_i = lerp(ckA["state_dict"], ckB["state_dict"], ALPHA)
crit_i = lerp(ckA["critic_state"], ckB["critic_state"], ALPHA) if ckA.get("critic_state") is not None else None

out = {"state_dict": actor_i,
       "critic_state": crit_i,
       "obs_mean": ckA["obs_mean"], "obs_std": ckA["obs_std"],
       "obs_dim": ckA.get("obs_dim"), "act_dim": ckA.get("act_dim"),
       "hidden": ckA.get("hidden"), "best_reward": -0.3797}
torch.save(out, OUT)
print(f"saved {OUT}  (alpha={ALPHA} interp of A={A}, B={B}; norm from A; best_reward=-0.3797)")
print("keys:", list(out.keys()))
