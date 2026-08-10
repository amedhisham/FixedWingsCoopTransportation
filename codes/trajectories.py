"""
trajectories.py — the trajectory DISTRIBUTION for generalization (single source of truth).

Samples rest-to-rest QUINTIC 6D pose trajectories (paper Fig.10 family, arXiv p2025ut):
3D position + small rotation, gentle enough that the classical base tracks them to ~cm
(validated in examine_base.py for position; the paper validates <=10 deg rotation with the
same PID gains -> 0.015 m). ramp/hold are fixed so every trajectory spans the same T_END,
keeping any time-indexed pieces (F2 reward) aligned.

Import this everywhere a reference is needed so collect / dagger / F2 share ONE distribution
and ONE held-out split:

    from trajectories import train_set, heldout_set, sample_traj
    for traj, params in train_set(K):   ...        # K training trajectories (fixed seed)
    for traj, params in heldout_set(M): ...        # M held-out (disjoint seed) trajectories
"""

import numpy as np
from controller import make_quintic_pose

# ---- distribution knobs (tweak here) --------------------------------------------------
POS_RANGE = 1.5           # each of dx,dy,dz ~ U[-POS_RANGE, POS_RANGE] m (paper move ~1 m)
ROT_RANGE_DEG = 10.0      # each of roll,pitch,yaw ~ U[-ROT, ROT] deg (paper 5-10 deg; 0 -> pos-only)
RAMP = 10.0               # move duration (s). FIXED -> T_END alignment (hold + ramp + hold = 25)
HOLD = 5.0                # initial hold (s)
BASE_POS = (0.0, 0.0, 1.39)   # start position (level start, R=I)
MIN_POS_NORM = 0.3        # reject near-zero moves (trivial / degenerate)

TRAIN_SEED = 20250803     # training trajectories are drawn from this stream
HELDOUT_SEED = 999999     # DISJOINT stream -> trajectories the policy never trains on
# ---------------------------------------------------------------------------------------


def sample_params(rng):
    """Draw one quintic-pose parameter set. Rejects near-zero position moves."""
    while True:
        pos_delta = rng.uniform(-POS_RANGE, POS_RANGE, size=3)
        if np.linalg.norm(pos_delta) >= MIN_POS_NORM:
            break
    rot_delta = np.deg2rad(rng.uniform(-ROT_RANGE_DEG, ROT_RANGE_DEG, size=3))
    return dict(pos_delta=pos_delta, rot_delta=rot_delta,
                ramp=RAMP, hold=HOLD, base_pos=np.asarray(BASE_POS, float))


def traj_from_params(p):
    """Build the callable traj(t) -> (p,v,R,omega) from a parameter set."""
    return make_quintic_pose(p["pos_delta"], p["rot_delta"], p["ramp"], p["hold"], p["base_pos"])


def sample_traj(rng):
    """Return (traj_callable, params) for one freshly sampled trajectory."""
    p = sample_params(rng)
    return traj_from_params(p), p


def _make_set(K, seed):
    rng = np.random.default_rng(seed)
    return [sample_traj(rng) for _ in range(K)]


def train_set(K, seed=TRAIN_SEED):
    """K training trajectories (reproducible from TRAIN_SEED)."""
    return _make_set(K, seed)


def heldout_set(M, seed=HELDOUT_SEED):
    """M held-out trajectories, drawn from a DISJOINT stream (never trained on)."""
    return _make_set(M, seed)
