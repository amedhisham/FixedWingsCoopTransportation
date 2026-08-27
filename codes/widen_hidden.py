"""
widen_hidden.py — function-preserving widen of the residual checkpoint's HIDDEN layers
(128,128) -> (256,256), for BOTH the actor and the critic. Distinct from widen_checkpoint.py,
which widened the OBS INPUT (44->98); this widens the internal width to give the net capacity to
represent a direction-dependent residual law instead of the jagged x-specialized fit
(see f2-axis-generalization: +y blows up, off-x is OOD).

FUNCTION-PRESERVING recipe (net2net "zero the outgoing weights of the new units"):
  For a 2-hidden-layer MLP  in -> h1 -> h2 -> out, widening each hidden 128 -> 256:
    * new h1 units (rows 128:256 of layer0): incoming = FRESH RANDOM (so they're alive & break
      symmetry -> gradient can recruit them), OUTGOING into old h2 = ZERO -> they don't perturb h2.
    * new h2 units (rows 128:256 of layer1): incoming from OLD h1 = fresh random, OUTGOING into
      the output head = ZERO -> they don't perturb the output.
  So at init the widened net computes the IDENTICAL map; the 256-2*128 new units start with zero
  influence and get recruited by gradient during training. log_std, obs-norm, act_dim unchanged.

Implementation: build a FRESH (256,256) net (correct default init everywhere), copy the old weights
into the preserved blocks, then zero the two "new-unit -> old-unit" outgoing blocks. Verified against
the source on random inputs (max |delta| < 1e-6) for both actor and critic.

Run:  python widen_hidden.py
"""
import numpy as np
import torch

from networks import Actor, Critic

SRC = "residual_mappo_gt2_wide.pt"          # obs 98, hidden (128,128)
DST = "residual_mappo_gt2_wide256.pt"       # obs 98, hidden (256,256)
OLD_H, NEW_H = 128, 256


def widen_actor(sd, obs_dim, act_dim):
    """Return a (256,256) actor state_dict computing the same map as the (128,128) `sd`."""
    new = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=(NEW_H, NEW_H))
    ns = new.state_dict()
    # log_std + output-head bias carry over verbatim (act_dim unchanged)
    ns["log_std"].copy_(sd["log_std"])
    ns["mean.bias"].copy_(sd["mean.bias"])
    # layer 0 (in -> h1): first 128 rows = old; rows 128:256 stay FRESH random (new h1 units, alive)
    ns["body.0.weight"][:OLD_H, :].copy_(sd["body.0.weight"])
    ns["body.0.bias"][:OLD_H].copy_(sd["body.0.bias"])
    # layer 1 (h1 -> h2): old block preserved; new-h1 -> old-h2 ZEROED; new-h2 rows stay fresh
    ns["body.2.weight"][:OLD_H, :OLD_H].copy_(sd["body.2.weight"])
    ns["body.2.weight"][:OLD_H, OLD_H:] = 0.0        # new h1 units do NOT reach old h2 -> h2 preserved
    ns["body.2.bias"][:OLD_H].copy_(sd["body.2.bias"])
    # output head (h2 -> act): old cols preserved; new-h2 -> out ZEROED -> output preserved
    ns["mean.weight"][:, :OLD_H].copy_(sd["mean.weight"])
    ns["mean.weight"][:, OLD_H:] = 0.0
    return ns


def widen_critic(sd, state_dim):
    """Return a (256,256) critic state_dict computing the same map as the (128,128) `sd`."""
    new = Critic(state_dim=state_dim, hidden=(NEW_H, NEW_H))
    ns = new.state_dict()
    ns["net.4.bias"].copy_(sd["net.4.bias"])
    ns["net.0.weight"][:OLD_H, :].copy_(sd["net.0.weight"])
    ns["net.0.bias"][:OLD_H].copy_(sd["net.0.bias"])
    ns["net.2.weight"][:OLD_H, :OLD_H].copy_(sd["net.2.weight"])
    ns["net.2.weight"][:OLD_H, OLD_H:] = 0.0
    ns["net.2.bias"][:OLD_H].copy_(sd["net.2.bias"])
    ns["net.4.weight"][:, :OLD_H].copy_(sd["net.4.weight"])
    ns["net.4.weight"][:, OLD_H:] = 0.0
    return ns


def _check(make, sd_old, sd_new, in_dim, hidden_old, **kw):
    """max |output delta| between the old and widened nets on random inputs."""
    a_old = make(hidden=(OLD_H, OLD_H), **kw); a_old.load_state_dict(sd_old); a_old.eval()
    a_new = make(hidden=(NEW_H, NEW_H), **kw); a_new.load_state_dict(sd_new); a_new.eval()
    x = torch.tensor(np.random.default_rng(0).standard_normal((128, in_dim)), dtype=torch.float32)
    with torch.no_grad():
        y_old = a_old(x) if not isinstance(a_old, Actor) else a_old(x)
        y_new = a_new(x)
    return float((y_old - y_new).abs().max())


def main():
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    obs_dim, act_dim = int(ck["obs_dim"]), int(ck["act_dim"])
    state_dim = ck["critic_state"]["net.0.weight"].shape[1]

    ck["state_dict"] = widen_actor(ck["state_dict"], obs_dim, act_dim)
    ck["critic_state"] = widen_critic(ck["critic_state"], state_dim)
    ck["hidden"] = [NEW_H, NEW_H]
    torch.save(ck, DST)
    print(f"widened {SRC} hidden ({OLD_H},{OLD_H}) -> ({NEW_H},{NEW_H})  ->  {DST}")

    # verify function-identical map for BOTH nets
    src = torch.load(SRC, map_location="cpu", weights_only=False)
    d_actor = _check(lambda hidden: Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden),
                     src["state_dict"], ck["state_dict"], obs_dim, (OLD_H, OLD_H))
    d_critic = _check(lambda hidden: Critic(state_dim=state_dim, hidden=hidden),
                      src["critic_state"], ck["critic_state"], state_dim, (OLD_H, OLD_H))
    print(f"max |delta output|  actor {d_actor:.2e}   critic {d_critic:.2e}")
    assert d_actor < 1e-6 and d_critic < 1e-6, "widened map DIFFERS — not function-preserving!"
    print("OK — widened actor+critic are function-identical to the source at init.")


if __name__ == "__main__":
    main()
