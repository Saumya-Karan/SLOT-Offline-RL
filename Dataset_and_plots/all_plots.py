import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# =========================================================
#  FOLDER / FILE CONFIGURATION
# =========================================================
DATASET_CSV = "slot_rl_dataset3_with_omni.csv"
LOSS_CSV    = "dqn_pytorch_loss_log.csv"
CONVERGENCE_CSV = "dqn_convergence_log.csv"
DEP_SUMMARY = "deploy_summary.csv"
DEP_STEPS   = "deploy_steps.csv"

OUT_DIR = "plots_for_paper"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# =========================================================
#  PER-FIGURE FONT SIZE CONTROL
#  Change any number below to control that figure only.
#  label     -> x/y axis label size
#  tick      -> tick label size
#  legend    -> legend text size
#  annotate  -> size of any in-plot text (bar values, % labels, etc.)
# =========================================================
FONT_CONFIG = {
    "fig1":  {"label": 35, "tick": 26, "legend": 20, "annotate": 25},
    "fig2":  {"label": 38, "tick": 30, "legend": 20, "annotate": 25},
    "fig3":  {"label": 26, "tick": 20, "legend": 20, "annotate": 25},
    "fig4":  {"label": 40, "tick": 35, "legend": 27, "annotate": 25},
    "fig5":  {"label": 43, "tick": 39, "legend": 27, "annotate": 25},
    "fig6":  {"label": 35, "tick": 30, "legend": 20, "annotate": 25},
    "fig7":  {"label": 40, "tick": 35, "legend": 35, "annotate": 25},
    "fig8":  {"label": 37, "tick": 25, "legend": 20, "annotate": 25},
    "fig9":  {"label": 33, "tick": 31, "legend": 25, "annotate": 25},
    "fig10": {"label": 25, "tick": 20, "legend": 20, "annotate": 20},
}

# =========================================================
#  GLOBAL STYLE: Times New Roman, bold everything (unrelated to sizing above)
# =========================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 25,
    'axes.titlesize': 27,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linewidth': 1.0,
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2.4,
    'xtick.major.width': 2.2,
    'ytick.major.width': 2.2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'lines.linewidth': 3.2,
    'font.weight': 'bold',
})

def style_axes(ax, cfg):
    """Apply reviewer-requested styling to a single Axes, sized per FONT_CONFIG entry."""
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2.4)
    ax.grid(True, alpha=0.5, linewidth=1.0)
    ax.tick_params(axis='both', which='major', width=2.2, length=8, labelsize=cfg["tick"], colors='black')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(cfg["label"])
    ax.yaxis.label.set_fontsize(cfg["label"])

print("=========================================")
print("  GENERATING ALL PAPER FIGURES (1 RUN)    ")
print("=========================================")

# =========================================================
#  LOAD ALL DATA
# =========================================================
df_data = pd.read_csv(DATASET_CSV)
df_data['Action'] = df_data['Action'].astype(str)
df_data_numeric = pd.to_numeric(df_data['Action'], errors='coerce')

df_sum = pd.read_csv(DEP_SUMMARY)
df_steps = pd.read_csv(DEP_STEPS)

try:
    df_loss = pd.read_csv(LOSS_CSV)
except FileNotFoundError:
    df_loss = None
    print(f"[skip] {LOSS_CSV} not found -- TD Loss figure will be skipped.")

try:
    df_conv = pd.read_csv(CONVERGENCE_CSV)
except FileNotFoundError:
    df_conv = None
    print(f"[skip] {CONVERGENCE_CSV} not found -- Convergence figure will be skipped.")

action_dict = {0: 'Forward', 1: 'Left Turn', 2: 'Right Turn', 3: 'Left Walk',
               4: 'Right Walk', 5: 'Crawl', 6: 'Omni'}
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#e67e22']

def smooth(y, weight=0.9):
    s = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        s.append(last)
    return np.array(s)

def add_fig_title(fig, text):
    fig.suptitle(text, y=1.03, fontweight='bold')

def wrap_label(text, max_words_per_line=1):
    words = text.split()
    if len(words) <= max_words_per_line:
        return text
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == max_words_per_line:
            lines.append(' '.join(cur))
            cur = []
    if cur:
        lines.append(' '.join(cur))
    return '\n'.join(lines)


# =========================================================
#  FIGURE 1: DATASET GAIT FREQUENCY
# =========================================================
print("Generating Figure 1: Dataset Gait Frequency...")
cfg = FONT_CONFIG["fig1"]
act_counts = df_data_numeric.value_counts().reindex(range(7), fill_value=0)

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.bar([wrap_label(a) for a in action_dict.values()], act_counts.values, color='#3498db', edgecolor='black', linewidth=1.5)
ax.set_xlabel("Action")
ax.set_ylabel("Frequency")
ax.set_ylim(0, max(act_counts) * 1.15)
for i, v in enumerate(act_counts.values):
    ax.text(i, v + (max(act_counts) * 0.02), str(v), ha='center', fontweight='bold', fontsize=cfg["annotate"])
style_axes(ax, cfg)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig1_Dataset_Gait_Frequency.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
#  FIGURE 2: REWARD DISTRIBUTION
# =========================================================
print("Generating Figure 2: Reward Distribution...")
cfg = FONT_CONFIG["fig2"]
reward_labels = {
    200: "Goal\nReached",
    50: "Waypoint\nCleared",
    15: "Correct\nCrawl\n(Bar Zone)",
    1: "Safe Step",
    -40: "Missed\nCrawl\nPenalty",
    -100: "Collision"
}
reward_order = [1, -40, 15, 50, 200, -100]
reward_counts = df_data["Reward"].value_counts().reindex(reward_order, fill_value=0)

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar([reward_labels[r] for r in reward_order], reward_counts.values, color="#e67e22", edgecolor='black', linewidth=1.5)
ax.set_yscale("log")
ax.set_xlabel("Reward Event")
ax.set_ylabel("Count (Log Scale)")
style_axes(ax, cfg)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig2_Reward_Distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
#  FIGURE 3: TRAJECTORY LENGTH DISTRIBUTION
# =========================================================
print("Generating Figure 3: Trajectory Length Distribution...")
cfg = FONT_CONFIG["fig3"]
lens = df_data.groupby('Traj_ID').size()

fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(lens[lens < 150], bins=15, color='#9b59b6', edgecolor='black', linewidth=1.5)
ax.set_xlabel("Trajectory Length (Steps)")
ax.set_ylabel("Number of Trajectories")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
style_axes(ax, cfg)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig3_Trajectory_Length.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
#  FIGURE 4: TD LOSS
# =========================================================
if df_loss is not None:
    print("Generating Figure 4: TD Loss Convergence...")
    cfg = FONT_CONFIG["fig4"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_loss['epoch'], df_loss['loss'], alpha=0.9, color='#c0392b', linewidth=5.2, label='Raw TD Loss', zorder=1)
    ax.plot(df_loss['epoch'], smooth(df_loss['loss'].values, 0.95), color='#641e16', lw=5.5, label='Smoothed', zorder=2)
    ax.set_yscale('log')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Huber Loss (Log Scale)")

    max_epoch = df_loss['epoch'].max()
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(np.linspace(0, max_epoch, 6))

    # --- Y-axis: hand-picked ticks, no LogLocator sprawl ---
    y_min = min(df_loss['loss'].min(), smooth(df_loss['loss'].values, 0.95).min())
    y_max = max(df_loss['loss'].max(), smooth(df_loss['loss'].values, 0.95).max())

    candidate_ticks = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    yticks = [t for t in candidate_ticks if y_min * 0.85 <= t <= y_max * 1.08]
    yticks = sorted(set(yticks) | {round(y_max, 2)})  # force one tick at the curve's true start

    from matplotlib.ticker import FuncFormatter
    ax.set_ylim(y_min * 0.85, y_max * 1.08)
    ax.set_yticks(yticks)
    ax.set_ylim(y_min * 0.85, y_max * 1.08)  # reassert once, last, after set_yticks
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2g}'))
    ax.yaxis.set_minor_locator(plt.NullLocator())  # no stray minor ticks either

    ax.legend(loc='best', fontsize=cfg["legend"], framealpha=0.9)
    style_axes(ax, cfg)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig4_TD_Loss.png", dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================
#  FIGURE 5: EXPECTED LEARNING CONVERGENCE
# =========================================================
if df_conv is not None:
    print("Generating Figure 5: Expected Learning Convergence...")
    cfg = FONT_CONFIG["fig5"]
    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.plot(df_conv['epoch'], df_conv['mean_expected_return'], color='#9b2fae', lw=5.0,
            label='Implied Episodic Reward (B1)')
    ax.plot(df_conv['epoch'], df_conv['mean_q_value'], color='#1a5276', lw=5.0,
            label='Mean Q-Value Confidence (B3)')
    ax.axhline(y=200, color='black', linestyle='--', linewidth=3.0, label='Reference: Terminal Goal Reward')
    ax.set_xlabel("Gradient Update Steps (Epochs)")
    ax.set_ylabel("Predicted Cumulative\nReward (Q-Value)")

    max_epoch = df_conv['epoch'].max()
    ax.set_xlim(0, max_epoch)
    xticks = np.linspace(0, max_epoch, 6)
    ax.set_xticks(xticks)
    xtick_labels = [str(int(t)) for t in xticks]
    xtick_labels[0] = ''   # blank out the x-axis "0" so it doesn't double up with the y-axis "0"
    ax.set_xticklabels(xtick_labels)

    y_max = max(df_conv['mean_q_value'].max(), df_conv['mean_expected_return'].max(), 200)
    ax.set_ylim(0, y_max * 1.05)

    ax.legend(loc='best', fontsize=cfg["legend"], framealpha=0.9)
    style_axes(ax, cfg)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig5_Convergence.png", dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================
#  FIGURE 6: Q-TRACE (Trajectory 1, Trial 1)
# =========================================================
print("Generating Figure 6: Q-Trace during Deployment...")
cfg = FONT_CONFIG["fig6"]
target_traj = 1
df_trace = df_steps[(df_steps['Traj_ID'] == target_traj) & (df_steps['Trial'] == 1)].copy()
df_trace = df_trace.sort_values('Time_sec')

fig, ax = plt.subplots(figsize=(13, 9))
ax.plot(df_trace['Time_sec'], df_trace['Max_Q'], color='#e74c3c', lw=4.5, label='Max Q-Value (Chosen Action)')

for act_idx, act_name in action_dict.items():
    act_data = df_trace[df_trace['Action'].astype(str).str.upper() == act_name.upper()]
    if not act_data.empty:
        ax.scatter(act_data['Time_sec'], act_data['Max_Q'], color=colors[act_idx], s=140,
                   edgecolor='black', zorder=5, label=f"Acted: {act_name}")

max_time = df_trace['Time_sec'].max()
ax.axvspan(0, max_time * 0.20, color='#2ecc71', alpha=0.15, label='Context: Open Corridor')
ax.axvspan(max_time * 0.20, max_time * 0.60, color='#f1c40f', alpha=0.15, label='Context: Obstacle Detected')
ax.axvspan(max_time * 0.60, max_time, color='#e74c3c', alpha=0.15, label='Context: Critical Proximity')

is_crash = df_sum[(df_sum['Traj_ID'] == target_traj) & (df_sum['Trial'] == 1)]['Result'].values[0] == 'Crash'
if is_crash:
    ax.axvline(x=max_time, color='black', linestyle='--', linewidth=2.5, label='Fatal Collision')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Predicted Expected Return (Q)")
ax.set_xlim(left=0)
if df_trace['Max_Q'].min() >= 0:
    ax.set_ylim(bottom=0)
style_axes(ax, cfg)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=cfg["legend"], framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig6_QTrace_Actions.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
#  FIGURE 7: ACTION SHIFT
# =========================================================
print("Generating Figure 7: Action Shift...")
cfg = FONT_CONFIG["fig7"]
label_order = [a.upper() for a in action_dict.values()]

hum_mapped = df_data_numeric.map(action_dict).str.upper()
hum_counts = hum_mapped.value_counts(normalize=True).reindex(label_order, fill_value=0) * 100

per_trial_props = []
for (traj, trial), group in df_steps.groupby(['Traj_ID', 'Trial']):
    props = group['Action'].str.upper().value_counts(normalize=True).reindex(label_order, fill_value=0)
    per_trial_props.append(props)
ai_counts = pd.concat(per_trial_props, axis=1).T.mean() * 100

x = np.arange(len(action_dict)); w = 0.35
fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(x - w/2, hum_counts.values, w, label='Manual Dataset', color='#7f8c8d', edgecolor='black', linewidth=1.5)
ax.bar(x + w/2, ai_counts.values, w, label='DQN AutoPilot', color='#27ae60', edgecolor='black', linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels([wrap_label(a) for a in action_dict.values()])
ax.set_ylabel("Frequency (%)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=cfg["legend"])
style_axes(ax, cfg)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig7_Action_Shift.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
#  FIGURE 8: SUCCESS HEATMAP
# =========================================================
print("Generating Figure 8: Success Heatmap...")
cfg = FONT_CONFIG["fig8"]
num_traj = int(max(df_sum['Traj_ID']))
hmap = np.full((3, num_traj), np.nan)
counts = {}
for _, row in df_sum.iterrows():
    tid = int(row['Traj_ID'])
    if tid not in counts:
        counts[tid] = 0
    if counts[tid] < 3:
        hmap[counts[tid], tid - 1] = 1.0 if row['Result'] == 'Goal' else 0.0
        counts[tid] += 1

fig, ax = plt.subplots(figsize=(14, 4))
cmap = mcolors.ListedColormap(['#e74c3c', '#2ecc71'])
cmap.set_bad('#bdc3c7')
ax.imshow(hmap, cmap=cmap, aspect='auto', vmin=0, vmax=1)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Trial 1', 'Trial 2', 'Trial 3'])
ax.set_xticks(range(num_traj))
ax.set_xticklabels(range(1, num_traj + 1))
ax.set_xlabel("Trajectory / Obstacle Configuration")
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(2.4)
ax.tick_params(axis='both', which='major', width=2.2, length=8, labelsize=cfg["tick"], colors='black')
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')
ax.xaxis.label.set_fontweight('bold')
ax.xaxis.label.set_fontsize(cfg["label"])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig8_Success_Heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
#  FIGURE 9: Q-VALUE VS. TIME-TO-TERMINAL
# =========================================================
print("Generating Figure 9: Q-Value vs. Time-to-Terminal...")
cfg = FONT_CONFIG["fig9"]

bins = []
qvals = []
outcomes = []

for (traj_id, trial), group in df_steps.groupby(['Traj_ID', 'Trial']):
    res = df_sum[(df_sum['Traj_ID'] == traj_id) & (df_sum['Trial'] == trial)]['Result'].values
    if len(res) == 0:
        continue
    g = group.sort_values('Step_Num').reset_index(drop=True)
    n = len(g)
    if n < 5:
        continue
    steps_from_end = n - 1 - g.index

    for q, sfe in zip(g['Max_Q'], steps_from_end):
        if sfe >= 10:
            b = ">10 steps\n to end"
        elif sfe >= 6:
            b = "6-10 steps\n to end"
        elif sfe >= 3:
            b = "3-5 steps\n to end"
        elif sfe >= 1:
            b = "1-2 steps\n to end"
        else:
            b = "Terminal\n step"
        bins.append(b)
        qvals.append(q)
        outcomes.append(res[0])

df_bins = pd.DataFrame({'bin': bins, 'q': qvals, 'outcome': outcomes})
bin_order = [">10 steps\n to end", "6-10 steps\n to end", "3-5 steps\n to end", "1-2 steps\n to end", "Terminal\n step"]

fig, ax = plt.subplots(figsize=(14, 9))
positions_goal = np.arange(len(bin_order)) * 2.2
positions_crash = positions_goal + 0.9

data_goal = [df_bins[(df_bins.bin == b) & (df_bins.outcome == 'Goal')]['q'].values for b in bin_order]
data_crash = [df_bins[(df_bins.bin == b) & (df_bins.outcome == 'Crash')]['q'].values for b in bin_order]

box_style = dict(linewidth=2.2)
whisker_style = dict(linewidth=2.2)
cap_style = dict(linewidth=2.2)
median_style = dict(linewidth=2.5, color='black')

bp1 = ax.boxplot(data_goal, positions=positions_goal, widths=0.7, patch_artist=True,
                  boxprops=box_style, whiskerprops=whisker_style, capprops=cap_style,
                  medianprops=median_style,
                  flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                                   markerfacecolor='gray', markeredgecolor='none'))
bp2 = ax.boxplot(data_crash, positions=positions_crash, widths=0.7, patch_artist=True,
                  boxprops=box_style, whiskerprops=whisker_style, capprops=cap_style,
                  medianprops=median_style,
                  flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                                   markerfacecolor='gray', markeredgecolor='none'))

for patch in bp1['boxes']:
    patch.set_facecolor('#2ecc71')
for patch in bp2['boxes']:
    patch.set_facecolor('#e74c3c')

from matplotlib.patches import Rectangle

MIN_VISIBLE_HEIGHT = 0.15

for positions, data_list, color in [(positions_goal, data_goal, '#2ecc71'),
                                     (positions_crash, data_crash, '#e74c3c')]:
    for pos, data in zip(positions, data_list):
        if len(data) == 0:
            continue
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        height = q3 - q1
        if height < MIN_VISIBLE_HEIGHT:
            y0 = med - MIN_VISIBLE_HEIGHT / 2
            rect = Rectangle((pos - 0.35, y0), 0.7, MIN_VISIBLE_HEIGHT,
                              facecolor=color, edgecolor='black', linewidth=2.2, zorder=5)
            ax.add_patch(rect)
            ax.plot([pos - 0.35, pos + 0.35], [med, med], color='black', linewidth=2.5, zorder=6)

ax.set_xticks(positions_goal + 0.45)
ax.set_xticklabels(bin_order)
ax.set_ylabel("Predicted Expected Return (Q-Value)")
style_axes(ax, cfg)

ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Goal Trajectories', 'Crash Trajectories'],
          loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=cfg["legend"])

fig.subplots_adjust(top=0.85)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"{OUT_DIR}/Fig9_QDrop_ByTimeToTerminal.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# =========================================================
#  FIGURE 10: OVERALL DEPLOYMENT OUTCOMES
# =========================================================
print("Generating Figure 10: Overall Performance...")
cfg = FONT_CONFIG["fig10"]
goals = len(df_sum[df_sum['Result'] == 'Goal']) / len(df_sum) * 100
crashes = len(df_sum[df_sum['Result'] == 'Crash']) / len(df_sum) * 100

fig, ax = plt.subplots(figsize=(9, 7))
ax.bar(['Goal Reached', 'Proximity-threshold termination'], [goals, crashes], color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
ax.set_ylabel("Percentage of Total Trials (%)")
ax.set_ylim(0, 100)
for i, v in enumerate([goals, crashes]):
    ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=cfg["annotate"])
style_axes(ax, cfg)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig10_Performance.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nDONE. All figures saved to:", OUT_DIR)