import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- FILE CONFIGURATION ---
LOSS_CSV = "dqn_pytorch_loss_log.csv"          # Fig 1: epoch, loss
CONVERGENCE_CSV = "dqn_convergence_log.csv"    # Fig 2: epoch, mean_q_value, mean_expected_return

OUT_DIR = "training_figures"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# --- GLOBAL STYLE: Times New Roman, large clear fonts, no rotation ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white'
})

def smooth(y, weight=0.9):
    s = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        s.append(last)
    return np.array(s)

def add_fig_title(fig, text):
    fig.suptitle(text, y=1.03, fontweight='bold')

print("=========================================")
print("  GENERATING TRAINING CONVERGENCE FIGURES ")
print("=========================================")

# =========================================================
#  FIGURE 1: DQN TD LOSS
# =========================================================
print("Generating Figure 1: TD Loss...")
df_loss = pd.read_csv(LOSS_CSV)

fig, ax = plt.subplots(figsize=(11, 7))
ax.plot(df_loss['epoch'], df_loss['loss'], alpha=0.3, color='#e74c3c', label='Raw TD Loss')
ax.plot(df_loss['epoch'], smooth(df_loss['loss'].values, 0.95), color='#c0392b', lw=2.5, label='Smoothed')
ax.set_yscale('log')
ax.set_xlabel("Epochs")
ax.set_ylabel("Huber Loss (Log Scale)")
ax.tick_params(axis='x', rotation=0)
ax.legend()
add_fig_title(fig, "Figure 1: DQN Temporal Difference Loss Convergence")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig1_TD_Loss.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# =========================================================
#  FIGURE 2: EXPECTED LEARNING CONVERGENCE (B1 & B3)
# =========================================================
print("Generating Figure 2: Expected Learning Convergence...")
df_conv = pd.read_csv(CONVERGENCE_CSV)

fig, ax = plt.subplots(figsize=(11, 7))
ax.plot(df_conv['epoch'], df_conv['mean_expected_return'], color='#e6b8e0', lw=1.5,
        label='Implied Episodic Reward (B1)')
ax.plot(df_conv['epoch'], df_conv['mean_q_value'], color='#2980b9', lw=2.5,
        label='Mean Q-Value Confidence (B3)')
ax.axhline(y=200, color='black', linestyle='--', linewidth=2, label='Max Target (Goal)')
ax.set_xlabel("Gradient Update Steps (Epochs)")
ax.set_ylabel("Predicted Cumulative Reward (Q-Value)")
ax.tick_params(axis='x', rotation=0)
ax.legend()
add_fig_title(fig, "Figure 2: Expected Learning Convergence (Offline Training)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig2_Convergence.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\nDone. Figures saved to:", OUT_DIR)