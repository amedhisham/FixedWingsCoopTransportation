"""
diagnose_buzz.py — characterize the closed-loop lambda buzz to pick the fix.

The buzz is time-localized: the INITIAL hover (t=0-5) is clean, but the FINAL hover
(t>=15, load parked) buzzes — and it got there through the self-fed move phase. That
asymmetry (same loiter regime, only the later one buzzes) is the signature of a GROWING
self-excited limit cycle in the lambda -> pR_dot -> lambda loop, NOT a coverage gap
(DAgger plateaued) and NOT lag (reconstruct_lp made it worse).

This runs the current policy on the DEFAULT straight-line but with an EXTENDED final
hover (park at t=15, sit until T_LONG) and answers two questions from the logged lambda:

  1. GROWTH: does the per-window buzz (mean |d lambda/step|) RISE across the final hover
     and then SATURATE? -> limit cycle. Or is it flat from t=15? -> fixed coverage error.
  2. FREQUENCY (FFT on the final-hover segment, detrended):
       - dominant f in the LOITER band [0.24, 0.48] Hz (omega/2pi, omega in [1.5,3])
         -> slow instability, needs real phase MEMORY (GRU).
       - dominant f near the STEP rate (tens of Hz) -> discrete-gain limit cycle,
         a lightweight output-side damping (load-safe: any lambda stays in nullspace)
         may kill it far cheaper than a GRU.

Own episode loop (longer end_time) so the shared T_END is untouched.
"""

import numpy as np
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from collect_il_data import read_params, N, DT, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import Reconstructor, build_input, LAM0
from deploy_prdot import load_policy, POLICY
import torch

T_LONG = 45.0          # extend the final hover: park at t=15, sit until here
MOVE_END = 15.0        # default trajectory parks its pose at t=15
WIN = 2.0              # window (s) for the sliding buzz curve
LOITER_BAND = (0.24, 0.48)   # omega/2pi for omega in linspace(1.5, 3, 7)

# DISCRIMINATOR knob: clip the per-drone pR_dot NORM fed to the net (None = off).
# Expert pR_dot is ~1.3 (p999=2.4); the self-fed loop drives it to ~105 (a divide-by-dt
# artifact, NOT real carrier velocity). Clamping toward ~5 pushes the input back into the
# densely-covered smooth-label regime AND severs the spike-feedback loop at the sensor.
# If the buzz dies -> feature amplifier, not coverage. If unchanged -> coverage/capacity.
CLIP_PRDOT = None     # e.g. 5.0 to test; None reproduces the current (buzzy) behaviour

# OUTPUT-side lambda low-pass (None = off). Smooths the ACTUATOR/feedback state, not the
# sensor: applied lambda = a*lam_raw + (1-a)*applied_prev, a = DT/(tau+DT), and the SMOOTHED
# lambda is what drives the force AND feeds back as lambda_{t-1}. Kills the >~1 Hz ripple at
# its source (so lambda_dot can't amplify it) while the 0.33 Hz loiter passes (~12 deg lag at
# tau=0.1). Load-safe (lambda stays in the nullspace). Distinct from reconstruct_lp, which
# lagged the SENSOR (pR_dot) and made it worse. tau ~ 0.1-0.2 to test.
LAM_LP_TAU = 0.1


def run_long(net, om, os_):
    """Closed-loop policy episode on the DEFAULT straight-line, extended to T_LONG.
    Mirrors deploy_prdot.run_episode step-for-step; returns t + per-drone lambda."""
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_LONG)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    obs42, _ = env.reset()
    agent.reset()
    prev_f = np.array([0.0, 0.0, FZ] * N)
    prev_lam = LAM0.copy()
    lam_lp = LAM0.copy()                    # output-side low-pass state
    lam_a = None if LAM_LP_TAU is None else DT / (LAM_LP_TAU + DT)
    recon = Reconstructor(Bb, L0, DT)

    t_hist = []
    lam_hist = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    tension = [[] for _ in range(N)]      # T_i = ||f_i||  (low T -> 1/T recon blowup)
    prnorm = [[] for _ in range(N)]       # ||pR_dot_i||  (the fed feature)

    t = 0.0
    while t < T_LONG - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t, None)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)
        vR = recon(R, vel, angvel, w_d, prev_lam)   # built from lambda_{t-1}: the fed feature
        vR_in = vR                                  # what the net actually sees
        if CLIP_PRDOT is not None:                  # clamp per-drone norm toward the physical range
            nrm = np.linalg.norm(vR, axis=1, keepdims=True)
            vR_in = vR * np.minimum(1.0, CLIP_PRDOT / (nrm + 1e-9))
        X = ((build_input(t, vR_in, prev_lam)[None, :] - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam = net(torch.tensor(X)).numpy().flatten()

        if lam_a is not None:                       # output-side low-pass: smooth the applied lambda
            lam_lp = lam_a * lam + (1.0 - lam_a) * lam_lp
            lam = lam_lp.copy()                     # drives force AND feeds back (recon.roll / prev_lam)

        f_full, _ = cable_force_calculation(R, Bb, w_d, lam, N)
        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()

        t_hist.append(t)
        for i in range(N):
            lam_hist[i].append(lam[i])
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
            tension[i].append(float(np.linalg.norm(f_full[3 * i: 3 * i + 3])))
            prnorm[i].append(float(np.linalg.norm(vR[i])))
        obs42, *_ = env.step(np.concatenate([ff, deriv]))

        recon.roll(lam)
        prev_lam = lam.copy()
        t += DT
    env.close()
    return (np.array(t_hist),
            [np.array(l) for l in lam_hist],
            [np.array(v) for v in dvel],
            [np.array(v) for v in tension],
            [np.array(v) for v in prnorm])


def window_buzz(t, lam, win=WIN):
    """Sliding-window mean |d lambda/step| averaged over drones, one value per window."""
    step = int(win / DT)
    centers, vals = [], []
    dl = [np.abs(np.diff(l)) for l in lam]        # (N, T-1)
    for s in range(0, len(t) - step, step):
        e = s + step
        centers.append(t[s:e].mean())
        vals.append(np.mean([d[s:e - 1].mean() for d in dl]))
    return np.array(centers), np.array(vals)


def dominant_freq(t, lam, t0, t1):
    """FFT the detrended per-drone lambda over [t0,t1], return (freqs, mean_power, f_peak)."""
    m = (t >= t0) & (t < t1)
    seg_t = t[m]
    n = seg_t.size
    freqs = np.fft.rfftfreq(n, d=DT)
    power = np.zeros_like(freqs)
    for l in lam:
        seg = l[m]
        seg = seg - np.polyval(np.polyfit(seg_t, seg, 1), seg_t)   # remove linear trend
        P = np.abs(np.fft.rfft(seg)) ** 2
        power += P
    power /= len(lam)
    f_peak = freqs[1:][np.argmax(power[1:])]      # skip DC
    return freqs, power, f_peak


def buzz_envelope(lam_i, win=0.3):
    """Per-drone high-frequency buzz envelope: |lambda - moving_average(lambda)|, smoothed.
    Isolates the fast hash riding on the ~0.33 Hz loiter fundamental."""
    k = max(int(win / DT), 3)
    box = np.ones(k) / k
    smooth = np.convolve(lam_i, box, mode="same")     # removes the fast buzz, keeps loiter
    hf = np.abs(lam_i - smooth)                        # the buzz magnitude
    return np.convolve(hf, box, mode="same")           # envelope


def main():
    net, om, os_ = load_policy(POLICY)
    print(f"policy: {POLICY}   extended horizon T_LONG={T_LONG}s (park at t={MOVE_END})")
    print(f"CLIP_PRDOT = {CLIP_PRDOT}\n")
    t, lam, dvel, tension, prnorm = run_long(net, om, os_)

    # artifact proof: reconstructed pR_dot (fed feature) vs the plant's TRUE carrier velocity
    pr_max = max(p.max() for p in prnorm)
    dv_max = max(v.max() for v in dvel)
    print(f"reconstructed pR_dot max (fed feature): {pr_max:8.2f} m/s")
    print(f"TRUE plant carrier velocity max:        {dv_max:8.2f} m/s   "
          f"-> ratio {pr_max / max(dv_max, 1e-9):.0f}x  ({'ARTIFACT' if pr_max > 5 * dv_max else 'plausible'})")
    vmin = min(v.min() for v in dvel)
    # argmin time (is the min a t=0 startup transient or a real mid-loiter stall?)
    stk = np.array([v.min() for v in dvel]); di = int(stk.argmin())
    ti = float(t[dvel[di].argmin()])
    skip = int(1.0 / DT)                                        # min excluding the first 1 s (startup)
    vmin_post = min(v[skip:].min() for v in dvel)
    print(f"TRUE plant carrier velocity MIN:         {vmin:8.3f} m/s  @ t={ti:.2f}s (drone {di+1})   "
          f"(eps={EPS}; {'STALL RISK' if vmin < EPS else 'ok'})")
    print(f"  MIN excluding first 1s (startup):      {vmin_post:8.3f} m/s   "
          f"({'real mid-flight stall' if vmin_post < EPS else 'ok -> the 0.000 was startup'})\n")

    # --- 1. growth: windowed buzz across the whole episode -----------------------------
    c, b = window_buzz(t, lam)
    print("windowed buzz  (mean |d lambda/step|):")
    for ci, bi in zip(c, b):
        bar = "#" * int(bi / max(b) * 40)
        print(f"  t~{ci:5.1f}s   {bi:.5f}  {bar}")

    hov0 = b[(c >= 0) & (c < 5)].mean()
    move = b[(c >= 5) & (c < 15)].mean()
    early_final = b[(c >= 15) & (c < 22)].mean()
    late_final = b[(c >= 30)].mean()
    print(f"\n  init hover (0-5):     {hov0:.5f}")
    print(f"  move (5-15):          {move:.5f}")
    print(f"  early final (15-22):  {early_final:.5f}")
    print(f"  late  final (>=30):   {late_final:.5f}")
    print(f"  -> growth ratio late/early final hover: {late_final / max(early_final, 1e-9):.2f}x  "
          f"(>>1 = still growing, ~1 = saturated/flat)")

    # --- 2. frequency: FFT the final hover ---------------------------------------------
    freqs, power, f_peak = dominant_freq(t, lam, MOVE_END, T_LONG)
    step_rate = 1.0 / (2 * DT)
    print(f"\nfinal-hover dominant lambda frequency: {f_peak:.2f} Hz")
    print(f"  loiter band (omega/2pi): {LOITER_BAND[0]:.2f}-{LOITER_BAND[1]:.2f} Hz")
    print(f"  step-to-step (Nyquist):  {step_rate:.1f} Hz")
    if f_peak <= LOITER_BAND[1] * 1.5:
        print("  => SLOW / loiter-rate -> genuine phase instability -> MEMORY (GRU) is the lever.")
    elif f_peak >= step_rate * 0.3:
        print("  => FAST / near step-rate -> discrete-gain limit cycle -> lightweight output damping first.")
    else:
        print("  => MID-band -> inspect the plot; likely a harmonic of the loiter mode.")

    # --- 3. TRIGGER: does the buzz burst when tension dips / pR_dot spikes? -------------
    env_all = [buzz_envelope(l) for l in lam]
    print("\nper-drone buzz vs conditioning (correlation over t):")
    print(f"  {'drone':>5}  {'buzz~-T (corr)':>15}  {'buzz~pRdot (corr)':>18}  {'min T':>7}  {'max pRdot':>9}")
    for i in range(N):
        e = env_all[i]
        c_negT = np.corrcoef(e, -tension[i])[0, 1]      # buzz high when tension low?
        c_pr = np.corrcoef(e, prnorm[i])[0, 1]          # buzz high when pR_dot spikes?
        print(f"  {i+1:>5}  {c_negT:>15.3f}  {c_pr:>18.3f}  {tension[i].min():>7.3f}  {prnorm[i].max():>9.2f}")
    # aggregate: pool the drone that buzzes most against its own tension
    buzz_tot = [e.mean() for e in env_all]
    j = int(np.argmax(buzz_tot))
    cj = np.corrcoef(env_all[j], -tension[j])[0, 1]
    print(f"  -> buzziest drone {j+1}: corr(buzz, -tension) = {cj:.3f}   "
          f"({'CONFIRMS low-tension trigger' if cj > 0.3 else 'weak -> not the tension gain'})")

    # --- plots -------------------------------------------------------------------------
    # overlay: buzziest drone's buzz envelope vs its tension + pR_dot norm
    figo, axo = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axo[0].plot(t, lam[j], "C0", lw=0.8, label=f"$\\lambda_{j+1}$")
    axo[0].plot(t, env_all[j], "r", lw=1.2, label="buzz envelope")
    axo[0].axvline(MOVE_END, ls="--", c="gray"); axo[0].legend(loc="upper right"); axo[0].grid(True)
    axo[0].set_title(f"buzziest drone {j+1}: lambda + buzz envelope")
    axo[1].plot(t, tension[j], "C2", label="tension $T_i=\\|f_i\\|$")
    axo[1].plot(t, prnorm[j], "C3", alpha=0.7, label="$\\|pR\\_dot_i\\|$ (fed feature)")
    axo[1].axvline(MOVE_END, ls="--", c="gray"); axo[1].legend(loc="upper right"); axo[1].grid(True)
    axo[1].set_xlabel("t (s)"); axo[1].set_title("tension (low -> 1/T recon blowup) + pR_dot norm")
    figo.tight_layout()

    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    for i in range(N):
        ax[0].plot(t, lam[i], label=f"$\\lambda_{i+1}$")
    ax[0].axvline(MOVE_END, ls="--", c="gray"); ax[0].set_ylabel("lambda")
    ax[0].set_title(f"lambda (closed loop, extended hover) — {POLICY}"); ax[0].legend(loc="upper right"); ax[0].grid(True)

    ax[1].plot(c, b, "o-"); ax[1].axvline(MOVE_END, ls="--", c="gray")
    ax[1].set_ylabel("windowed buzz"); ax[1].set_xlabel("t (s)")
    ax[1].set_title("sliding-window buzz  (growth = limit cycle)"); ax[1].grid(True)

    ax[2].semilogy(freqs[1:], power[1:])
    ax[2].axvspan(LOITER_BAND[0], LOITER_BAND[1], color="C2", alpha=0.3, label="loiter band")
    ax[2].axvline(f_peak, ls="--", c="r", label=f"peak {f_peak:.2f} Hz")
    ax[2].set_xlabel("frequency (Hz)"); ax[2].set_ylabel("lambda power")
    ax[2].set_xlim(0, min(step_rate, 25)); ax[2].set_title("final-hover lambda spectrum")
    ax[2].legend(); ax[2].grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
