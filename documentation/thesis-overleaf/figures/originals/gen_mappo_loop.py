import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

BLUE = "#1f6fb2"
ORANGE = "#d9711a"
GREEN = "#2e8b3d"
GREY = "#666666"
DARKBLUE = "#003366"
GOLD = "#b8860b"

W, H = 100, 136
FIG_W = 12.5
fig = plt.figure(figsize=(FIG_W, FIG_W * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
PT = FIG_W * 72 / W   # points per data unit

fig.text(0.5, 0.981, "MAPPO Training Loop and Environment Step",
          ha="center", va="center", fontsize=16, fontweight="bold", color=DARKBLUE)


def panel(x0, y0, x1, y1, face, edge):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, facecolor=face,
                                 edgecolor=edge, linewidth=1.4, linestyle=(0, (6, 3)),
                                 boxstyle="round,pad=0,rounding_size=1.6", zorder=0))


def header(x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=13.5, fontweight="bold",
            style="italic", color=color, zorder=3)


def box(cx, cy, w, h, title, sub, face, edge, fs_t=13, fs_s=10.5, sub_ls=1.35):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, facecolor=face,
                                 edgecolor=edge, linewidth=1.8,
                                 boxstyle="round,pad=0,rounding_size=1.2", zorder=2))
    n_t = title.count("\n") + 1
    n_s = (sub.count("\n") + 1) if sub else 0
    th = n_t * fs_t * 1.45 / PT
    sh = n_s * fs_s * sub_ls * 1.2 / PT
    gap = 0.8 if sub else 0
    top = cy + (th + gap + sh) / 2
    ax.text(cx, top, title, ha="center", va="top", fontsize=fs_t,
            fontweight="bold", linespacing=1.15, zorder=3)
    if sub:
        ax.text(cx, top - th - gap, sub, ha="center", va="top", fontsize=fs_s,
                linespacing=sub_ls, zorder=3)


def arrow(p1, p2, color=GREY, lw=2.0, ls="solid", head=True):
    style = "-|>" if head else "-"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=20,
                                  linewidth=lw, linestyle=ls, color=color,
                                  shrinkA=0, shrinkB=0, zorder=1))


def label(x, y, text, color=GREY, fontsize=10.5, rotation=0):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color,
            style="italic", rotation=rotation, zorder=4)


# ---------------- Panel 1: collect experience ------------------------------
panel(1.5, 102.5, 98.5, 128.5, "#f3f8fd", "#9cc0e3")
LX, RX, BW = 26, 77, 31       # left / right column centres and box width
LEFT_EDGE = LX - BW / 2       # 10.5
header(LEFT_EDGE, 125.4, "\u2460  Collect experience (multiple episodes)", BLUE)

R1_CY, R1_H = 113.5, 15
box(LX, R1_CY, BW, R1_H, r"Actor $\pi_\theta$",
    "decentralized\n" r"observation $\rightarrow$ $[\delta\lambda\ (4),\ \delta w\ (6)]$" "\nsample action",
    "#eaf2fb", BLUE)
box(RX, R1_CY, BW, R1_H, "Environment Step",
    "per-drone internals\n(expanded below)", "#f2f2f2", GREY)

MID = (LX + RX) / 2
arrow((RX - BW / 2, 117.5), (LX + BW / 2, 117.5))
label(MID, 119.7, "per-carrier observation")
arrow((LX + BW / 2, 109.5), (RX - BW / 2, 109.5))
label(MID, 107.5, "action")

# ---------------- Panel 2: update policy -----------------------------------
panel(1.5, 50.5, 98.5, 100, "#fdf8f2", "#e8b98a")
header(LEFT_EDGE, 97.4, "\u2461  Update policy", ORANGE)

R2_CY, R2_H = 87, 16
box(LX, R2_CY, BW, R2_H, r"Critic $V_\phi$",
    "input: privileged state (76)\n" r"$\rightarrow$ value $V$", "#fdf6e3", GOLD)
box(RX, R2_CY, BW, R2_H, "Rollout buffer",
    "observations, actions,\nlog-probs of actions,\nprivileged states, rewards", "#fbecdf", ORANGE)

# collected transitions go into the buffer
arrow((RX, R1_CY - R1_H / 2), (RX, R2_CY + R2_H / 2))

# privileged state goes to the critic, value V comes back
arrow((RX - BW / 2, 90.5), (LX + BW / 2, 90.5), color=GOLD)
label(MID, 93.0, "privileged state", color=GOLD)
arrow((LX + BW / 2, 83.5), (RX - BW / 2, 83.5), color=GOLD)
label(MID, 81.3, r"value $V$", color=GOLD)

R3_CY, R3_H = 65, 14
box(RX, R3_CY, BW, R3_H, "GAE",
    "per-drone advantages\n+ returns (returns averaged\nacross drones for critic target)",
    "#eef7ee", GREEN)
box(LX, R3_CY, BW, R3_H, "PPO update",
    "8 epochs\n" r"clip $\epsilon = 0.2$", "#eef7ee", GREEN)
arrow((RX, R2_CY - R2_H / 2), (RX, R3_CY + R3_H / 2))
arrow((RX - BW / 2, R3_CY), (LX + BW / 2, R3_CY))

# PPO update -> critic
arrow((LX, R3_CY + R3_H / 2), (LX, R2_CY - R2_H / 2), color=GOLD)
ax.text(LX + 1.6, (R3_CY + R3_H / 2 + R2_CY - R2_H / 2) / 2, "update critic", ha="left",
        va="center", fontsize=10.5, color=GOLD, style="italic", zorder=4)

# PPO update -> actor, along the left margin (inner lane)
ACT_LANE = 6.4
STUB_Y = 68
arrow((LEFT_EDGE, STUB_Y), (ACT_LANE, STUB_Y), color=BLUE, head=False)
arrow((ACT_LANE, STUB_Y), (ACT_LANE, R1_CY), color=BLUE, head=False)
arrow((ACT_LANE, R1_CY), (LEFT_EDGE, R1_CY), color=BLUE)
label(8.3, 90, "update actor", color=BLUE, rotation=90)

# go to next iteration: back to the collect-experience panel (outer lane)
NEXT_LANE = 3.8
LOOP_Y = 55.5
DARK = "#444444"
arrow((LX, R3_CY - R3_H / 2), (LX, LOOP_Y), color=DARK, ls=(0, (5, 3)), head=False)
arrow((LX, LOOP_Y), (NEXT_LANE, LOOP_Y), color=DARK, ls=(0, (5, 3)), head=False)
arrow((NEXT_LANE, LOOP_Y), (NEXT_LANE, 102.5), color=DARK, ls=(0, (5, 3)))
label(15, 53.3, "Go to next iteration", color=DARK)

# ---------------- Panel 3: environment step (expanded) ---------------------
panel(1.5, 0.8, 98.5, 48.5, "#f8f8f8", "#bbbbbb")
header(6, 45.7, "Environment step (expanded)", "#444444")

BOX_W, BOX_H = 26, 14
CX = [19, 50, 81]
ROW_A, ROW_B = 35.0, 12.0
box(CX[0], ROW_A, BOX_W, BOX_H, "1  Base controller",
    r"$\rightarrow\ \lambda_{base},\ w_d$", "#ffffff", GREY)
box(CX[1], ROW_A, BOX_W, BOX_H, "2  Add residual /\npolicy action + clip",
    r"$\|\delta\lambda\| \leq \kappa_\lambda\,\|\lambda_{base}\|$" "\n"
    r"$\|\delta w\| \leq \kappa_w\,\|w_d\|$", "#eaf2fb", BLUE)
box(CX[2], ROW_A, BOX_W, BOX_H, "3  Assemble force\nper carrier",
    r"$f = f_{base}$" "\n" r"$+\ G^{\dagger}\,\delta w$" "\n" r"$+\ N\,\delta\lambda$",
    "#ffffff", GREY)
box(CX[0], ROW_B, BOX_W, BOX_H, "4  FMU plant step",
    "desync:\nsensor noise\n+ delay", "#ffffff", GREY)
box(CX[1], ROW_B, BOX_W, BOX_H, "5  Reward",
    "manifold + stall + load\n" r"+ overspeed" "\n" r"+ $\lambda$-overshoot", "#fbecdf", ORANGE)
box(CX[2], ROW_B, BOX_W, BOX_H, "6  Next per-drone\nobservations",
    "load estimate + own carrier state\n+ base forces + Fourier time\nembeddings + history + reference",
    "#eaf2fb", BLUE, fs_s=10)

half = BOX_W / 2
for row in (ROW_A, ROW_B):
    arrow((CX[0] + half, row), (CX[1] - half, row))
    arrow((CX[1] + half, row), (CX[2] - half, row))

# carriage return: box 3 -> box 4
JOIN_Y = 23.5
arrow((CX[2], ROW_A - BOX_H / 2), (CX[2], JOIN_Y), head=False)
arrow((CX[2], JOIN_Y), (CX[0], JOIN_Y), head=False)
arrow((CX[0], JOIN_Y), (CX[0], ROW_B + BOX_H / 2))
label(50, 25.8, "Steps 1–3 run once per carrier, then the full force vector is applied to the plant",
      fontsize=10.5)

fig.savefig("/tmp/claude-1000/-media-hisham-New-Volume-Masters-Internship-and-Thesis-prep-FixedWingsCoopTransportation/6e7632d7-2270-413b-bb70-8394a88bf0b9/scratchpad/mappo_loop.png",
            dpi=200, facecolor="white")
print("done")
