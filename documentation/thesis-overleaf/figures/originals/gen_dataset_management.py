import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

plt.rcParams["font.family"] = "DejaVu Sans"

BLUE = "#1f6fb2"
ORANGE = "#d9711a"
GREEN = "#2e8b3d"
RED = "#b23b3b"
GREY = "#666666"
DARKBLUE = "#003366"

fig = plt.figure(figsize=(12.5, 13.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 140)
ax.axis("off")

fig.text(0.5, 0.975, "Final Dataset Management Strategy (per DAgger iteration)",
          ha="center", va="center", fontsize=15, fontweight="bold", color=DARKBLUE)

def box(cx, cy, w, h, text, facecolor, edgecolor, fontsize=11.5):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, facecolor=facecolor,
                        edgecolor=edgecolor, linewidth=1.8,
                        boxstyle="round,pad=0,rounding_size=1.2")
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            linespacing=1.5)

def arrow(p1, p2, color=GREY, lw=2.0, ls="solid", head=True):
    style = "-|>" if head else "-"
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=20,
                         linewidth=lw, linestyle=ls, color=color,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def label(x, y, text, color=GREY, fontsize=9.5, weight="normal", ha="center", boxed=False):
    kwargs = dict(ha=ha, va="center", fontsize=fontsize, color=color,
                  fontweight=weight, style="italic" if weight == "normal" else "normal")
    if boxed:
        kwargs["bbox"] = dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor="none", alpha=0.9)
    ax.text(x, y, text, **kwargs)

# ---- Row 1: inputs (cy=124, h=20 -> 114 to 134) ------------------------
R1_CY, R1_H = 124, 20
box(20, R1_CY, 36, R1_H,
    "New DAgger Rollout\n"
    r"$\approx$120K points" + "\n"
    "labeled: hardness, age, bin",
    "#eaf2fb", BLUE)

box(80, R1_CY, 36, R1_H,
    "Historical Data Pool\n"
    r"cap $\approx$250K points",
    "#fbecdf", ORANGE)

R1_BOTTOM = R1_CY - R1_H / 2  # 114

# ---- Row 2: training set (cy=94, h=20 -> 84 to 104) ---------------------
R2_CY, R2_H = 94, 20
box(50, R2_CY, 54, R2_H,
    "Training Set  " + r"($\approx$200K points)" + "\n"
    r"$\approx$60% new / 40% old",
    "#eef7ee", GREEN, fontsize=12)
R2_TOP = R2_CY + R2_H / 2       # 104
R2_BOTTOM = R2_CY - R2_H / 2    # 84

arrow((20, R1_BOTTOM), (44, R2_TOP))
label(11, 109, "all new data\nkept", fontsize=9.2, boxed=True)
arrow((80, R1_BOTTOM), (56, R2_TOP))
label(66, 111.5,
      r"sample $\approx$65% of new size" + "\n(min. 20% per bin enforced)",
      fontsize=9.2, boxed=True)

# ---- Row 3: train policy (cy=71, h=14 -> 64 to 78) ----------------------
R3_CY, R3_H = 71, 14
box(50, R3_CY, 34, R3_H, "Train Policy\n(this DAgger iteration)", "#ffffff", GREY, fontsize=11.5)
R3_TOP = R3_CY + R3_H / 2   # 78
R3_BOTTOM = R3_CY - R3_H / 2  # 64
arrow((50, R2_BOTTOM), (50, R3_TOP))

# ---- Decision diamond (dy=49, half=9 -> 40 to 58) ------------------------
DY, DHALF = 49, 9
diamond = Polygon([(50, DY + DHALF), (70, DY), (50, DY - DHALF), (30, DY)],
                   closed=True, facecolor="#fdf6e3", edgecolor="#b8860b", linewidth=1.8)
ax.add_patch(diamond)
ax.text(50, DY, "Policy imitates\noptimizer well?", ha="center", va="center", fontsize=10.8)
arrow((50, R3_BOTTOM), (50, DY + DHALF))

# ---- No / Yes branches (row5 cy=23, h=18 -> 14 to 32) --------------------
R5_CY, R5_H = 23, 18
R5_TOP = R5_CY + R5_H / 2     # 32
R5_BOTTOM = R5_CY - R5_H / 2  # 14

arrow((30, DY), (15, DY), color=RED, head=False)
arrow((15, DY), (15, R5_TOP), color=RED)
label(22, DY + 3.5, "No", color=RED, fontsize=11, weight="bold")

arrow((70, DY), (85, DY), color=GREEN, head=False)
arrow((85, DY), (85, R5_TOP), color=GREEN)
label(78, DY + 3.5, "Yes", color=GREEN, fontsize=11, weight="bold")

box(15, R5_CY, 26, R5_H,
    r"$\bf{Repeat}$" + " iteration at same " + r"$\beta$" + "\n"
    "(dataset & pool grow\nuncapped, no eviction)",
    "#fbe9e9", RED, fontsize=9.6)

box(85, R5_CY, 26, R5_H,
    "Add new 120K to pool, then\nevict uniformly to restore\n"
    "250K cap (min. 20% per bin kept)",
    "#eef7ee", GREEN, fontsize=9.6)

# ---- Loop-backs along the outer margins (kept perfectly vertical) ------
arrow((9, R5_TOP), (9, R1_BOTTOM), color=RED, ls=(0, (5, 3)))
arrow((91, R5_TOP), (91, R1_BOTTOM), color=GREEN)

# ---- Bin legend ---------------------------------------------------------
fig.text(0.5, 0.02,
          r"Bins: $\bf{0}$ = Loiter (ref. speed $<0.05$ m/s)     "
          r"$\bf{1}$ = Cruise (steady motion)     "
          r"$\bf{2}$ = Transition ($|$accel$| > 0.02$ m/s$^2$)",
          ha="center", va="center", fontsize=10.5, color="#333333")

fig.savefig("/tmp/claude-1000/-media-hisham-New-Volume-Masters-Internship-and-Thesis-prep-FixedWingsCoopTransportation/6e7632d7-2270-413b-bb70-8394a88bf0b9/scratchpad/dataset_management_new.png",
            dpi=200, facecolor="white")
print("done")
