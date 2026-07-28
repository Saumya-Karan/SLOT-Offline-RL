import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

CSV_FILE = "slot_rl_dataset3_with_omni.csv"   # relative path -- adjust if needed on your home machine
MODEL_FILE = "slot_pytorch_ddqn_test.pth"

print("========================================")
print("  PYTORCH DDQN (XAVIER STABILIZED)      ")
print("========================================")

if not os.path.exists(CSV_FILE):
    print(f"Error: Cannot find {CSV_FILE}"); exit()

# --- 1. PREPARE THE DATA ---
df = pd.read_csv(CSV_FILE)

grid_cols = [f"Grid_{i}" for i in range(25)]
for col in grid_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5.0).clip(0.0, 5.0)

df['Action'] = df['Action'].astype(str).str.lower().str.strip()
df['Action'] = df['Action'].replace({'crash': '0', 'crawl_fwd': '5', 'crawl': '5', 'left_turn': '1', 'right_turn': '2'})
df['Action'] = pd.to_numeric(df['Action'], errors='coerce').fillna(0).astype(np.int64)

df['Reward'] = pd.to_numeric(df['Reward'], errors='coerce').fillna(0.0)
df['Survival'] = pd.to_numeric(df['Survival'], errors='coerce').fillna(1.0)

# --- 2. CREATE PYTORCH TENSORS ---
S_tensor = torch.nan_to_num(torch.tensor(df[grid_cols].values.astype(np.float32))).clamp(0.0, 5.0)
A_tensor = torch.clamp(torch.tensor(df['Action'].values.astype(np.int64)).unsqueeze(1), 0, 5)
R_tensor = torch.nan_to_num(torch.tensor(df['Reward'].values.astype(np.float32) / 100.0)).unsqueeze(1)
Done_tensor = torch.nan_to_num(torch.tensor(df['Survival'].values.astype(np.float32))).unsqueeze(1)
Done_tensor = 1.0 - Done_tensor

next_states = torch.zeros_like(S_tensor)
for i in range(len(df) - 1):
    traj_changed = ('Traj_ID' in df.columns) and (df['Traj_ID'].iloc[i] != df['Traj_ID'].iloc[i+1])
    is_dead = (Done_tensor[i].item() == 1.0)
    if is_dead or traj_changed:
        Done_tensor[i] = 1.0
        next_states[i] = S_tensor[i]
    else:
        next_states[i] = S_tensor[i+1]

S_next_tensor = next_states

# --- 3. THE NEURAL NETWORK ---
class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.net(x)

policy_net = DDQN(25, 6)
target_net = DDQN(25, 6)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4, eps=1e-6)
loss_fn = nn.SmoothL1Loss()
bc_loss_fn = nn.CrossEntropyLoss()

# --- 4. TRAINING LOOP ---
BATCH_SIZE = 128
GAMMA = 0.95
EPOCHS = 250
TAU = 0.005

print(f"Safe Data: {len(S_tensor)} transitions.")
print(f"Training PyTorch DDQN for {EPOCHS} epochs...")

dataset_size = len(S_tensor)
indices = np.arange(dataset_size)

# --- Fig 1 log: per-epoch TD+BC loss (unchanged) ---
log_file = open("dqn_pytorch_loss_log.csv", "w")
log_file.write("epoch,loss,td_loss,bc_loss\n")

# --- Fig 2 log: B3 (mean Q-value confidence) and B1 (implied episodic
# return, via the bootstrapped target `expected_q`) -- both computed
# from tensors already produced inside the training loop, so this is a
# real measurement, not a fabricated curve. ---
convergence_log = open("dqn_convergence_log.csv", "w")
convergence_log.write("epoch,mean_q_value,mean_expected_return\n")

# Reward was scaled by /100 when building R_tensor, so Q-values and
# expected returns come out of the network in that same /100 scale.
# Multiply back by 100 when logging so the numbers are directly
# comparable to the raw reward scale (e.g. Goal = 200), matching the
# "Max Target (Goal) = 200" reference line used in Fig 2.
REWARD_SCALE = 100.0

for epoch in range(1, EPOCHS + 1):
    np.random.shuffle(indices)
    epoch_loss = 0.0
    epoch_mean_q = 0.0
    epoch_mean_expected = 0.0
    n_batches = 0

    for start_idx in range(0, dataset_size, BATCH_SIZE):
        batch_idx = indices[start_idx : start_idx + BATCH_SIZE]

        b_s = S_tensor[batch_idx]
        b_a = A_tensor[batch_idx]
        b_r = R_tensor[batch_idx]
        b_s_next = S_next_tensor[batch_idx]
        b_done = Done_tensor[batch_idx]
        bc_weight = 0.5 * max(0.05, 1 - epoch/(EPOCHS*0.25))

        current_q_all = policy_net(b_s)
        current_q = current_q_all.gather(1, b_a)

        with torch.no_grad():
            next_actions = policy_net(b_s_next).argmax(1, keepdim=True)
            max_next_q = target_net(b_s_next).gather(1, next_actions)

        expected_q = b_r + (GAMMA * max_next_q * (1.0 - b_done))

        td_loss = loss_fn(current_q, expected_q)
        bc_loss = bc_loss_fn(current_q_all, b_a.squeeze())
        loss = td_loss + (bc_weight * bc_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

        epoch_loss += loss.item()

        # --- B3: mean of the network's own best predicted Q-value this batch ---
        epoch_mean_q += current_q_all.max(dim=1).values.mean().item()

        # --- B1: mean bootstrapped target (r + gamma * max_next_q), used
        # as a proxy for "implied episodic reward" since it's literally
        # what the network is being trained to predict ---
        epoch_mean_expected += expected_q.mean().item()

        n_batches += 1

    avg_loss = epoch_loss / n_batches
    avg_mean_q = (epoch_mean_q / n_batches) * REWARD_SCALE
    avg_mean_expected = (epoch_mean_expected / n_batches) * REWARD_SCALE
    epoch_td_loss = 0.0
    epoch_bc_loss = 0.0
    epoch_td_loss += td_loss.item()
    epoch_bc_loss += bc_loss.item()

    log_file.write(f"{epoch},{avg_loss},{epoch_td_loss/n_batches},{epoch_bc_loss/n_batches}\n")
    convergence_log.write(f"{epoch},{avg_mean_q},{avg_mean_expected}\n")

    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Mean Q: {avg_mean_q:.2f} | Mean Expected Return: {avg_mean_expected:.2f}")

log_file.close()
convergence_log.close()
torch.save(policy_net.state_dict(), MODEL_FILE)
print(f"\nSaved model as '{MODEL_FILE}'.")
print("Logs written: dqn_pytorch_loss_log.csv (Fig 1), dqn_convergence_log.csv (Fig 2)")