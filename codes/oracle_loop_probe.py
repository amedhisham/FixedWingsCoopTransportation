"""
oracle_loop_probe.py — NO training. Answers: what does the ORACLE clean-view correction actually
achieve in CLOSED LOOP? For each eval trajectory we roll the desynced plant under 4 modes and read
the mean loop_dist (+ load / coord / leak):

  zero  : residual OFF (base under desync)                      -> the desync baseline
  dlam  : oracle dlam* = clip(lam_clean - lam_base) per drone   -> perfect-perception nullspace correction
  dw    : oracle dw*   = clip(w_clean   - w_base)   per drone   -> perfect-perception range correction
  both  : oracle dlam* + dw*                                    -> full clean-view correction

The oracle is injected INSIDE step at the exact point the actor's action would be (env._oracle_mode),
so it goes through the SAME cap-clip -> zero lag, no confound. If oracle-dlam gives a GOOD quintic loop
(~0.43 or better) then dlam* genuinely IS the target (improve estimability -> the pin works). If it gives
a BAD loop (~0.65, worse than RL's reward-driven dlam) then the frozen base is OOD on the desync-scattered
true state -> the "oracle" is only "frozen-base-with-perfect-eyes", not optimal control -> RL must own dlam.
"""
import numpy as np
from residual_marl_env import ResidualMARLEnv
from expert_reference import eval_scenarios
from collect_il_data import T_END
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS

MODES = ["zero", "dlam", "dw", "both"]
GRACE = 20                       # skip startup (lambda history warming from LAM0)


def roll(env, scen, mode):
    label, traj, epos = scen
    env.traj, env.expert_pos = traj, epos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    env._oracle_mode = None if mode == "zero" else mode
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    zero = {a: np.zeros(env._act_space.shape[0], np.float32) for a in agents}
    loop, load, coord, leak, satl, satw = [], [], [], [], [], []
    k = 0
    while env.agents:
        obs, r, term, trunc, infos = env.step(zero)
        k += 1
        if k <= GRACE:
            continue
        loop.append(np.mean([infos[a]["loop_dist"] for a in agents]))
        load.append(np.mean([infos[a]["load_err"] for a in agents]))
        coord.append(np.mean([infos[a]["coord"] for a in agents]))
        leak.append(np.mean([infos[a]["leak"] for a in agents]))
        satl.append(np.mean([infos[a]["sat_lam"] for a in agents]))
        satw.append(np.mean([infos[a]["sat_w"] for a in agents]))
    env._oracle_mode = None
    return dict(loop=np.mean(loop), load=np.mean(load), coord=np.mean(coord),
                leak=np.mean(leak), satl=np.mean(satl), satw=np.mean(satw), n=len(loop))


def main():
    env = ResidualMARLEnv(**DESYNC, end_time=T_END, track_clean_lambda=True)
    print("oracle closed-loop probe: what does the clean-view correction achieve? (mean over episode, GRACE-skipped)\n")
    print(f"{'traj':<16}{'mode':<7}{'loop':>8}{'load':>8}{'coord':>8}{'leak':>8}{'sat_lam':>9}{'sat_w':>8}")
    for scen in eval_scenarios():
        label = scen[0]
        for mode in MODES:
            d = roll(env, scen, mode)
            print(f"{label[:15]:<16}{mode:<7}{d['loop']:>8.3f}{d['load']:>8.3f}{d['coord']:>8.3f}"
                  f"{d['leak']:>8.3f}{d['satl']:>9.2f}{d['satw']:>8.2f}")
        print()
    env.close()


if __name__ == "__main__":
    main()
