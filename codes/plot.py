import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [1.1, 1]})

# Schedule Data
iterations = np.arange(0, 11)
alpha = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
expert_weight = 1.0 - alpha

# Panel 1: Graph
ax1.plot(iterations, expert_weight, label=r'Expert Weight ($1 - \alpha$)', color='#1f77b4', linewidth=2.5, marker='o', markersize=7)
ax1.plot(iterations, alpha, label=r'Policy Weight ($\alpha$)', color='#d95f02', linewidth=2.5, marker='s', markersize=7)

ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)
ax1.axvspan(-0.5, 0.5, alpha=0.15, color='#1f77b4', label='Phase 1 (1 Iteration)')
ax1.axvspan(0.5, 10.5, alpha=0.15, color='#d95f02', label='Phase 2 (10-Step Injection)')

ax1.set_title(r'Behavioral Cloning Schedule: Phase 1 (Supervised Learning) $\rightarrow$ Phase 2 (DAgger)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Training Iteration', fontsize=11, labelpad=8)
ax1.set_ylabel('Action Weight Ratio', fontsize=11, labelpad=8)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(-0.5, 10.5)
ax1.set_xticks(iterations)
ax1.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

# Panel 2: Text Diagram
ax2.axis('off')
ax2.text(0.5, 0.95, "Mechanics: Phase 1 Supervised Learning vs. Phase 2 DAgger", fontsize=12, fontweight='bold', ha='center', color='#2b2b2b')

p1_text = (
    r"Phase 1: Initial Pass (Iter 0)" + "\n\n" +
    r"• Action: $a_0 = a^*_0$ ($\alpha = 0$)" + "\n" +
    r"• Goal: Pure expert trajectory collection" + "\n" +
    r"  & initial fitting of $\pi_{\theta_0}$." + "\n" +
    r"• Result: Baseline offline policy."
)
ax2.text(0.22, 0.4, p1_text, fontsize=9.5, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#eef5fc', edgecolor='#1f77b4', lw=1.5))

p2_text = (
    r"Phase 2: Policy Action Injection (Iter 1–10)" + "\n\n" +
    r"• Action: $a_t = (1-\alpha)a^*_t + \alpha \pi(s_t)$" + "\n" +
    r"• Visited States: $\alpha$ increases $0.1 \rightarrow 1.0$." + "\n" +
    r"• Goal: Exposes policy to its own mistakes" + "\n" +
    r"  while expert re-labels $s_{t+1}$ with $a^*_{t+1}$." + "\n" +
    r"  Data is collected and added to the Dataset"  + "\n" + 
    r" The Network is retrained on the Aggregated Dataset."
)
ax2.text(0.78, 0.4, p2_text, fontsize=9.5, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fcf0e8', edgecolor='#d95f02', lw=1.5))

ax2.annotate('', xy=(0.53, 0.4), xytext=(0.47, 0.4), arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5, mutation_scale=20))

plt.tight_layout()
plt.show()