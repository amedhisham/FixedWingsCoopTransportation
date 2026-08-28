"""
library_sweep.py — run the DET policy on the 50 TRAINING quintics (IN-SAMPLE) and report loop/satL/
blowup per quintic + a png, same style as direction_sweep. This answers: does the policy even FIT its
own training set, or does it blow up / saturate on it too (the truncation problem, in-sample)?

Uses the library's PRECOMPUTED expert paths (training_pairs) -> fast, no CasADi. Rolls each at T_END
(the training horizon) under the same EVAL_SEED + 0-1-2 delay walk as direction_sweep, so the numbers
are comparable. Covariate panel = move MAGNITUDE (does blowup track move size / saturation?).
"""
import numpy as np
import matplotlib.pyplot as plt
from residual_marl_env import ResidualMARLEnv
from expert_reference import training_pairs
from trajectories import train_set
from collect_prdot_data import N_TRAJ
from collect_il_data import T_END
from mappo import DESYNC
from direction_sweep import load_actor, roll, CKPT


def main():
    env = ResidualMARLEnv(**DESYNC, end_time=T_END)
    actor, om, os_ = load_actor(env)
    pairs, n_anchor = training_pairs()
    quintics = pairs[n_anchor:]                 # 50 (traj, expert_dpos), index-aligned with train_set
    params = train_set(N_TRAJ)
    assert len(quintics) == len(params) == N_TRAJ, (len(quintics), len(params), N_TRAJ)

    print(f"library quintic sweep  ckpt={CKPT}  end {T_END}s  ({N_TRAJ} TRAINING quintics, in-sample)")
    print(f"{'#':>3} {'pos_delta (x,y,z)':>24}{'|d|':>7}{'loop':>8}{'satL':>7}  result")
    rows = []
    for i, ((traj, dpos), (_, p)) in enumerate(zip(quintics, params)):
        pos = np.asarray(p["pos_delta"], float); mag = float(np.linalg.norm(pos))
        loop, sat, blew = roll(env, actor, om, os_, traj, dpos)
        rows.append(dict(i=i, pos=pos, mag=mag, loop=loop, sat=sat, blew=blew))
        tag = "BLEW UP" if blew else ("SATURATED" if sat >= 1.0 else "ok")
        print(f"{i+1:>3} ({pos[0]:+5.2f},{pos[1]:+5.2f},{pos[2]:+5.2f}){mag:>7.2f}{loop:>8.3f}{sat:>7.2f}  {tag}")
    env.close()

    n_blew = sum(r["blew"] for r in rows)
    n_sat = sum((not r["blew"]) and r["sat"] >= 1.0 for r in rows)
    n_ok = sum((not r["blew"]) and r["sat"] < 1.0 for r in rows)
    surv = [r["loop"] for r in rows if not r["blew"]]
    print(f"\nsummary: {n_ok} ok  |  {n_sat} saturated-survive  |  {n_blew} BLEW UP   "
          f"(of {N_TRAJ})   mean loop over survivors {np.mean(surv):.3f}")

    # --- plot: sorted worst->best, loop bars (regime color) + magnitude bars ---
    rows.sort(key=lambda r: r["loop"], reverse=True)
    labels = [f"#{r['i']+1} ({r['pos'][0]:+.1f},{r['pos'][1]:+.1f},{r['pos'][2]:+.1f})" for r in rows]
    loops = np.array([r["loop"] for r in rows])
    mags = np.array([r["mag"] for r in rows])

    def color(r):
        if r["blew"]:
            return "black"
        return "crimson" if r["sat"] >= 1.0 else "steelblue"
    colors = [color(r) for r in rows]
    y = np.arange(len(rows))

    fig, (axL, axM) = plt.subplots(1, 2, figsize=(14, 12), sharey=True)
    axL.barh(y, loops, color=colors)
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=6)
    axL.invert_yaxis()
    axL.set_xlabel("mean loop_dist  (5 = blowup)")
    axL.set_title(f"Training quintics (in-sample) — {CKPT}  @ {T_END}s")
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(color="crimson", label="saturated (satL>=1)"),
                        Patch(color="steelblue", label="unsaturated (satL<1)"),
                        Patch(color="black", label="blowup")], loc="lower right", fontsize=8)
    axM.barh(y, mags, color="gray")
    axM.set_xlabel("move magnitude |pos_delta| (m)")
    axM.set_title("Move size per quintic")
    plt.tight_layout()
    plt.savefig("library_sweep.png", dpi=120)
    print("saved figure -> library_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
