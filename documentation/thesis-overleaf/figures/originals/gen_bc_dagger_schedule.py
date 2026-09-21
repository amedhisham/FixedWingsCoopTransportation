import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

expert_weight = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15,
                  0.1, 0.08, 0.06, 0.04, 0.02, 0.0]
iters = list(range(len(expert_weight)))
policy_weight = [1.0 - w for w in expert_weight]

BLUE = "#1f6fb2"
ORANGE = "#d9711a"
DARKBLUE = "#003366"

fig = plt.figure(figsize=(11.5, 11.6))

ax = fig.add_axes([0.09, 0.60, 0.87, 0.34])
ax.set_facecolor("#fbfbfb")

ax.axvspan(-0.3, 0.5, color=BLUE, alpha=0.10, zorder=0)
ax.axvspan(0.5, 15.3, color=ORANGE, alpha=0.10, zorder=0)
ax.axvline(0.5, color="#666666", linestyle=(0, (5, 3)), linewidth=1.1, zorder=1)

ax.plot(iters, expert_weight, color=BLUE, marker="o", markersize=6,
        linewidth=2.4, label="Expert Weight (1 − α)", zorder=3)
ax.plot(iters, policy_weight, color=ORANGE, marker="s", markersize=6,
        linewidth=2.4, label="Policy Weight (α)", zorder=3)

ax.set_xlim(-0.4, 15.4)
ax.set_ylim(-0.03, 1.08)
ax.set_xticks(iters)
ax.set_xlabel("Training Iteration", fontsize=12, labelpad=6)
ax.set_ylabel("Action Weight Ratio", fontsize=12, labelpad=8)
ax.set_title("Behavioral Cloning Schedule: Phase 1 (Supervised Learning) → Phase 2 (DAgger)",
             fontsize=14, fontweight="bold", pad=12)
ax.grid(True, color="white", linewidth=1.4, zorder=0)
for spine in ax.spines.values():
    spine.set_color("#bbbbbb")

phase1_patch = mpatches.Patch(color=BLUE, alpha=0.10, label="Phase 1 (1 Iteration)")
phase2_patch = mpatches.Patch(color=ORANGE, alpha=0.10, label="Phase 2 (15-Step Injection)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [phase1_patch, phase2_patch], loc="center right",
          fontsize=10, framealpha=0.95, edgecolor="#bbbbbb")

fig.text(0.5, 0.535, "Mechanics: Phase 1 Supervised Learning vs. Phase 2 DAgger",
          ha="center", va="center", fontsize=13.5, fontweight="bold", color=DARKBLUE)

# --- Bottom boxes -----------------------------------------------------
box_ax = fig.add_axes([0, 0, 1, 0.48])
box_ax.set_xlim(0, 1)
box_ax.set_ylim(0, 1)
box_ax.axis("off")
box_ax.patch.set_alpha(0)

box_style = dict(boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.6)

left_text = (
    r"$\bf{Phase\ 1:\ Initial\ Pass\ (Iter\ 0)}$" + "\n\n"
    r"$\bullet$ Action: $a_0 = a_0^{*}$   ($\alpha = 0$)" + "\n"
    r"$\bullet$ Goal: Pure expert trajectory collection" + "\n"
    r"    & initial fitting of $\pi_{\theta_0}$." + "\n"
    r"$\bullet$ Result: Baseline offline policy."
)
right_text = (
    r"$\bf{Phase\ 2:\ Policy\ Action\ Injection\ (Iter\ 1–15)}$" + "\n\n"
    r"$\bullet$ Action: $a_t = (1-\alpha)\,a_t^{*} + \alpha\,\pi(s_t)$" + "\n"
    r"$\bullet$ Visited States: $\alpha$ increases $0.1 \to 1.0$" + "\n"
    r"    over the 15-step schedule." + "\n"
    r"$\bullet$ Goal: Exposes policy to its own mistakes" + "\n"
    r"    while expert re-labels $s_{t+1}$ with $a_{t+1}^{*}$." + "\n"
    r"$\bullet$ Data is aggregated and the network is" + "\n"
    r"    retrained on the full dataset each iteration."
)

box_ax.add_patch(FancyBboxPatch((0.045, 0.25), 0.38, 0.62, facecolor="#eaf2fb",
                                 edgecolor=BLUE, **box_style))
box_ax.text(0.235, 0.83, left_text, ha="center", va="top", fontsize=12.5,
            linespacing=1.7, transform=box_ax.transAxes)

box_ax.add_patch(FancyBboxPatch((0.575, 0.25), 0.38, 0.62, facecolor="#fbecdf",
                                 edgecolor=ORANGE, **box_style))
box_ax.text(0.765, 0.83, right_text, ha="center", va="top", fontsize=12.5,
            linespacing=1.7, transform=box_ax.transAxes)

arrow = FancyArrowPatch((0.445, 0.56), (0.555, 0.56), transform=box_ax.transAxes,
                         arrowstyle="-|>", mutation_scale=22, linewidth=2.4,
                         shrinkA=0, shrinkB=0, color="#444444")
box_ax.add_patch(arrow)

fig.savefig("/tmp/claude-1000/-media-hisham-New-Volume-Masters-Internship-and-Thesis-prep-FixedWingsCoopTransportation/6e7632d7-2270-413b-bb70-8394a88bf0b9/scratchpad/bc_dagger_new.png",
            dpi=200, facecolor="white")
print("done")
