"""
trajectories.py — the trajectory DISTRIBUTION for generalization (single source of truth).

Samples rest-to-rest QUINTIC 6D pose trajectories (paper Fig.10 family, arXiv p2025ut):
3D position + small rotation, gentle enough that the classical base tracks them to ~cm
(validated in examine_base.py for position; the paper validates <=10 deg rotation with the
same PID gains -> 0.015 m). The move-end is now RANDOMIZED (ramp ~ U[15,25]s) to stop the net
memorizing a fixed timing; alongside these, custom_set() adds harsh constant-velocity moves that
actually ENGAGE the deadbanded optimizer. (NB: F2's time-indexed reward assumed a fixed ramp/T_END
alignment — revisit that if reusing this distribution there.)

Import this everywhere a reference is needed so collect / dagger / F2 share ONE distribution
and ONE held-out split:

    from trajectories import train_set, heldout_set, sample_traj
    for traj, params in train_set(K):   ...        # K training trajectories (fixed seed)
    for traj, params in heldout_set(M): ...        # M held-out (disjoint seed) trajectories
"""

import numpy as np
from controller import make_quintic_pose, make_linear_move

# ---- distribution knobs (tweak here) --------------------------------------------------
POS_RANGE = 3.0           # each of dx,dy,dz ~ U[-POS_RANGE, POS_RANGE] m (bumped 1.5->3.0 with the
                          #   longer ramp so peak velocity stays gentle ~0.3 m/s; base tracks to cm)
ROT_RANGE_DEG = 10.0      # each of roll,pitch,yaw ~ U[-ROT, ROT] deg (paper 5-10 deg; 0 -> pos-only)
RAMP_MIN, RAMP_MAX = 15.0, 25.0   # move duration ~ U[15,25] -> move-END = hold + ramp ~ U[20,30]s.
                          #   RANDOMIZED (was fixed 10) so there's no constant "move ends at t=15"
                          #   marker for the clock/positional-encoding to memorize.
HOLD = 5.0                # initial hold (s)
BASE_POS = (0.0, 0.0, 1.39)   # start position (level start, R=I)
MIN_POS_NORM = 0.3        # reject near-zero moves (trivial / degenerate)

# ---- custom SOLVER-ENGAGING trajectories (non-quintic) --------------------------------
# The gentle quintics rarely stress the eps-floor, so the deadbanded optimizer almost never
# SOLVES on them -> the net could learn a pure time-indexed A/xi schedule. These 5 constant-
# velocity moves (like the harsh default straight-line) give the optimizer real adaptive solves,
# forcing the net to learn the pR_dot-conditioned law. Few (5 of ~55) so they don't drown the
# distribution. Movement spans [5, 30]s -> no pure-loiter tail to memorize.
# ALL HORIZONTAL: verified headless that ANY vertical component keeps the formation symmetric and
# lifts every ||v_Ri|| off the eps-floor (+z / +x+z / +x+y+z solve ~0%), so the internal-force
# coordination is a HORIZONTAL-plane phenomenon. z-variety is already covered by the 3-D quintics;
# the customs' job is engagement, which needs in-plane motion. Solve rates below (v=1.1-scale).
CUSTOM_HOLD = 5.0
CUSTOM_MOVE_DUR = 25.0            # move [5, 30] (T_END=35 -> 5 s tail hold)
CUSTOM_VELS = [                  # (name, cruise velocity m/s)   [headless solve%]
    ("+x",   (1.1, 0.0, 0.0)),   # ~5%
    ("+y",   (0.0, 1.1, 0.0)),   # ~5%
    ("+x+y", (0.8, 0.8, 0.0)),   # ~10%   |v| ~ 1.13
    ("-x+y", (-0.8, 0.8, 0.0)),  # ~10%   sign variety
    ("+x-y", (0.8, -0.8, 0.0)),  # ~10%   sign variety
]

TRAIN_SEED = 20250803     # training trajectories are drawn from this stream
HELDOUT_SEED = 999999     # DISJOINT stream -> trajectories the policy never trains on
# ---------------------------------------------------------------------------------------


def sample_params(rng):
    """Draw one quintic-pose parameter set. Rejects near-zero position moves.
    Ramp is randomized per-trajectory (move-end ~ U[20,30]s) to break fixed-timing memorization."""
    while True:
        pos_delta = rng.uniform(-POS_RANGE, POS_RANGE, size=3)
        if np.linalg.norm(pos_delta) >= MIN_POS_NORM:
            break
    rot_delta = np.deg2rad(rng.uniform(-ROT_RANGE_DEG, ROT_RANGE_DEG, size=3))
    ramp = float(rng.uniform(RAMP_MIN, RAMP_MAX))
    return dict(pos_delta=pos_delta, rot_delta=rot_delta,
                ramp=ramp, hold=HOLD, base_pos=np.asarray(BASE_POS, float))


def traj_from_params(p):
    """Build the callable traj(t) -> (p,v,R,omega) from a parameter set."""
    return make_quintic_pose(p["pos_delta"], p["rot_delta"], p["ramp"], p["hold"], p["base_pos"])


def sample_traj(rng):
    """Return (traj_callable, params) for one freshly sampled trajectory."""
    p = sample_params(rng)
    return traj_from_params(p), p


def custom_set():
    """The fixed set of 5 solver-engaging constant-velocity trajectories (deterministic, no seed).
    Returns (traj_callable, params) pairs; params carry pos_delta/rot_delta (for the shared print
    path) plus 'name' and kind='custom'."""
    out = []
    for name, vel in CUSTOM_VELS:
        traj = make_linear_move(vel, hold=CUSTOM_HOLD, move_dur=CUSTOM_MOVE_DUR, base_pos=BASE_POS)
        v = np.asarray(vel, float)
        p = dict(name=name, kind="custom", pos_delta=v * CUSTOM_MOVE_DUR,
                 rot_delta=np.zeros(3), hold=CUSTOM_HOLD, move_dur=CUSTOM_MOVE_DUR)
        out.append((traj, p))
    return out


def _make_set(K, seed):
    rng = np.random.default_rng(seed)
    return [sample_traj(rng) for _ in range(K)]


# ---- showcase / demo trajectories (test-only, NEVER collected) -------------------------
# For thesis plots. Decoupled from the training set so we can pick ANY length/timing and show
# the (memorization-fixed) net generalizes: SHORT presets (25 s, move-end t=15) use timing
# OUTSIDE the training window (quintics train on move-end U[20,30]); LONG presets (35 s) mirror
# training. run at each traj's own t_end (the deploy scripts loop these).
SHOWCASE_SEED = 424242    # DISJOINT from TRAIN_SEED / HELDOUT_SEED


def showcase_line(t_end=25.0):
    """The original straight-line demo (+x, move [5,15]). t_end=25 -> the original compact plot;
    t_end=35 -> the default training anchor (same shape, collected at T_END). Returns (traj, params)."""
    traj = make_linear_move((1.1, 0.0, 0.0), hold=5.0, move_dur=10.0, base_pos=BASE_POS)
    p = dict(name="line", kind="showcase", pos_delta=np.array([11.0, 0.0, 0.0]),
             rot_delta=np.zeros(3), t_end=t_end)
    return traj, p


def showcase_quintics(M, t_end, ramp, seed=SHOWCASE_SEED):
    """M quintic demos at a CHOSEN timing (fixed `ramp`), seeded + DISJOINT from train/heldout.
    Like heldout_set but with a controllable ramp/horizon. Returns [(traj, params)]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(M):
        while True:
            pos_delta = rng.uniform(-POS_RANGE, POS_RANGE, size=3)
            if np.linalg.norm(pos_delta) >= MIN_POS_NORM:
                break
        rot_delta = np.deg2rad(rng.uniform(-ROT_RANGE_DEG, ROT_RANGE_DEG, size=3))
        traj = make_quintic_pose(pos_delta, rot_delta, ramp=ramp, hold=HOLD, base_pos=BASE_POS)
        out.append((traj, dict(pos_delta=pos_delta, rot_delta=rot_delta, ramp=ramp,
                               kind="showcase", t_end=t_end)))
    return out


def showcase_set(kind="short", M=3):
    """Named demo preset -> [(label, traj, t_end)] for the deploy scripts to loop.
      'short' : compact 25 s — original line + M short quintics (ramp 10, move-end 15 = OUTSIDE
                training's U[20,30] -> length/timing-generalization demo).
      'long'  : 35 s training-like — line-at-35 (default anchor) + M quintics (ramp 23, move-end 28).
    """
    out = []
    if kind == "short":
        tl, _ = showcase_line(25.0)
        out.append(("short_line", tl, 25.0))
        for i, (tr, _p) in enumerate(showcase_quintics(M, t_end=25.0, ramp=10.0)):
            out.append((f"short_quintic{i+1}", tr, 25.0))
    elif kind == "long":
        tl, _ = showcase_line(35.0)
        out.append(("long_line", tl, 35.0))
        for i, (tr, _p) in enumerate(showcase_quintics(M, t_end=35.0, ramp=23.0)):
            out.append((f"long_quintic{i+1}", tr, 35.0))
    else:
        raise ValueError(f"unknown showcase kind {kind!r} (use 'short' or 'long')")
    return out


def train_set(K, seed=TRAIN_SEED):
    """K training trajectories (reproducible from TRAIN_SEED)."""
    return _make_set(K, seed)


def heldout_set(M, seed=HELDOUT_SEED):
    """M held-out trajectories, drawn from a DISJOINT stream (never trained on)."""
    return _make_set(M, seed)
