# claude_memory — portable project memory

These are the persistent memory notes Claude accumulated while working on this project (F1 pR_dot base +
F2 MAPPO residual). They normally live under Claude's per-user home (`~/.claude/...`), which does **not**
cross the Windows↔Ubuntu dual-boot, so this is the portable copy checked into the repo.

## For a fresh Claude session (e.g. on Ubuntu)

These files are **not auto-loaded** here (the auto-memory system reads `~/.claude`, not the repo). To
bootstrap context without starting from scratch:

1. **Read `MEMORY.md` first** — it's the one-line-per-note index of everything below.
2. Then read whichever linked notes are relevant to the task (they link each other with `[[name]]`).
3. Treat them as **point-in-time observations**, not live state — verify any file:line citation or code
   claim against the current code before relying on it (some may be stale).

Optionally, to get them auto-loading on the Ubuntu side, copy these `.md` files into that machine's
`~/.claude/projects/<this-project's-key>/memory/` (the key is derived from the repo path, so it differs
from Windows — let Claude create the folder once, then drop these in).

## Where to start for the current work

- **HPC migration** (in progress): see `../HPC_MIGRATION.md` (repo root of `codes/`) + `f2-hpc-migration.md`.
- **F2 state of play**: `f2-residual-rl-plan.md`, `f2-distillation-capacity-verdict.md`,
  `f2-axis-generalization.md`, `f2-wd-nullleak-breakthrough.md`.
- **Who the user is / how to work with them**: `user_profile.md`, `project_overview.md`.
