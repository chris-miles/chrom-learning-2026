# %% [markdown]
# # 09 -- NRI-lite with typed poles: latent topology inference
#
# This notebook replaces the earlier fixed-channel interaction network with a
# simple latent-graph model in the spirit of Neural Relational Inference (NRI).
#
# The comparison is now genuinely different from the basis-expansion pipeline:
#
# - The edge structure is **latent**, not hand-specified as `xy` vs `xx`.
# - The model sees **short trajectory windows**, not just one snapshot.
# - The model infers discrete edge types with a **variational graph encoder**.
#
# We still keep two practical simplifications relative to the original NRI
# formulation:
#
# 1. The two spindle poles are treated as **typed observed nodes**.
# 2. The decoder predicts only the next-step chromosome velocity, while the
#    poles are used as observed drivers.
#
# That makes this a useful "NRI-lite" comparison for the chromosome data:
# the graph is latent, but the notebook stays compact enough to experiment with.
#
# **Requires:** `pip install torch`

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in dir()
    else Path("..").resolve()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromlearn.io.catalog import load_condition
from chromlearn.io.trajectory import trim_trajectory

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "This notebook requires PyTorch. Install with: pip install torch"
    ) from exc

plt.rcParams["figure.dpi"] = 110

SEED = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def mlp(in_dim, hidden_dim, out_dim, depth=2):
    layers = []
    last_dim = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


# %% [markdown]
# ## Load trimmed trajectories
#
# Same cells and trimming window as the SFI notebooks.

# %%
CONDITION = "rpe18_ctr"
DT = 5.0

cells_raw = load_condition(CONDITION)
cells = [trim_trajectory(c, method="neb_ao_frac") for c in cells_raw]
print(f"Loaded {len(cells)} cells")
for cell in cells:
    T, _, N = cell.chromosomes.shape
    print(f"  {cell.cell_id}: {T} frames, {N} chromosomes")


# %% [markdown]
# ## Build fixed-length graph windows
#
# Each training sample is a short window ending at time `t`:
#
# - history of node positions over `HISTORY` frames
# - next-step chromosome velocity target `(x(t+1) - x(t)) / dt`
#
# Nodes are:
#
# - chromosome slots `0 .. max_chroms-1`
# - pole slots `max_chroms, max_chroms+1`
#
# To support cells with different tracked chromosome counts, we pad to the
# maximum chromosome count and carry explicit node masks.

# %%
HISTORY = 8
MIN_VALID_CHROMS = 2

TYPE_CHROM = 0
TYPE_POLE = 1

MAX_CHROMS = max(cell.chromosomes.shape[2] for cell in cells)
MAX_NODES = MAX_CHROMS + 2
POLE_SLOTS = np.array([MAX_CHROMS, MAX_CHROMS + 1], dtype=np.int64)
NODE_TYPES = np.concatenate(
    [
        np.full(MAX_CHROMS, TYPE_CHROM, dtype=np.int64),
        np.full(2, TYPE_POLE, dtype=np.int64),
    ]
)

print(f"Max chromosome slots: {MAX_CHROMS}")
print(f"Total padded nodes:    {MAX_NODES}")


def extract_windows(cells, history=HISTORY):
    """Create padded trajectory windows with node masks and velocity targets."""
    windows = []
    per_cell_counts = []

    for cell_idx, cell in enumerate(cells):
        chromosomes = cell.chromosomes  # (T, 3, N)
        centrioles = cell.centrioles    # (T, 3, 2)
        T = chromosomes.shape[0]
        N = chromosomes.shape[2]
        n_windows_cell = 0

        for t in range(history - 1, T - 1):
            hist_slice = slice(t - history + 1, t + 1)

            chrom_block = chromosomes[t - history + 1 : t + 2]  # (H+1, 3, N)
            pole_block = centrioles[t - history + 1 : t + 2]    # (H+1, 3, 2)

            if not np.isfinite(pole_block).all():
                continue

            # Require each chromosome to be finite throughout the full window
            # and at the target frame. This keeps the latent graph consistent
            # within a sample.
            valid_chrom = np.isfinite(chrom_block).all(axis=(0, 1))
            valid_indices = np.flatnonzero(valid_chrom)
            if valid_indices.size < MIN_VALID_CHROMS:
                continue

            pos_hist = np.zeros((history, MAX_NODES, 3), dtype=np.float32)
            target_vel = np.zeros((MAX_NODES, 3), dtype=np.float32)
            node_mask = np.zeros(MAX_NODES, dtype=bool)
            target_mask = np.zeros(MAX_NODES, dtype=bool)

            # Anchor each sample to the pole midpoint at the current frame so
            # the model does not spend capacity on absolute translation.
            origin = 0.5 * (centrioles[t, :, 0] + centrioles[t, :, 1])  # (3,)

            chrom_hist = chromosomes[hist_slice].transpose(0, 2, 1) - origin
            chrom_cur = chromosomes[t].T - origin
            chrom_next = chromosomes[t + 1].T - origin
            pole_hist = centrioles[hist_slice].transpose(0, 2, 1) - origin

            pos_hist[:, valid_indices, :] = chrom_hist[:, valid_indices, :]
            pos_hist[:, POLE_SLOTS, :] = pole_hist

            node_mask[valid_indices] = True
            node_mask[POLE_SLOTS] = True
            target_mask[valid_indices] = True

            target_vel[valid_indices] = (
                chrom_next[valid_indices] - chrom_cur[valid_indices]
            ) / cell.dt

            windows.append(
                {
                    "pos_hist": pos_hist,
                    "node_mask": node_mask,
                    "target_vel": target_vel,
                    "target_mask": target_mask,
                    "cell_idx": cell_idx,
                    "cell_id": cell.cell_id,
                    "n_valid_chroms": int(valid_indices.size),
                    "n_cell_chroms": int(N),
                }
            )
            n_windows_cell += 1

        per_cell_counts.append((cell.cell_id, n_windows_cell))

    return windows, per_cell_counts


windows, window_counts = extract_windows(cells, history=HISTORY)
print(f"Built {len(windows)} windows")
for cell_id, count in window_counts:
    print(f"  {cell_id}: {count} windows")

if not windows:
    raise RuntimeError("No valid windows were extracted.")

valid_counts = np.array([w["n_valid_chroms"] for w in windows], dtype=np.int64)
print(
    f"Valid chromosomes per window: mean={valid_counts.mean():.1f}, "
    f"min={valid_counts.min()}, max={valid_counts.max()}"
)


def pack_windows(window_list):
    """Stack a list of windows into dense numpy arrays."""
    if not window_list:
        return {
            "pos_hist": np.zeros((0, HISTORY, MAX_NODES, 3), dtype=np.float32),
            "node_mask": np.zeros((0, MAX_NODES), dtype=bool),
            "target_vel": np.zeros((0, MAX_NODES, 3), dtype=np.float32),
            "target_mask": np.zeros((0, MAX_NODES), dtype=bool),
            "cell_idx": np.zeros(0, dtype=np.int64),
        }

    return {
        "pos_hist": np.stack([w["pos_hist"] for w in window_list]).astype(np.float32),
        "node_mask": np.stack([w["node_mask"] for w in window_list]).astype(bool),
        "target_vel": np.stack([w["target_vel"] for w in window_list]).astype(np.float32),
        "target_mask": np.stack([w["target_mask"] for w in window_list]).astype(bool),
        "cell_idx": np.array([w["cell_idx"] for w in window_list], dtype=np.int64),
    }


all_data = pack_windows(windows)
print(
    "Window tensor shapes:",
    {k: v.shape for k, v in all_data.items() if hasattr(v, "shape")}
)


# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(valid_counts, bins=np.arange(valid_counts.min(), valid_counts.max() + 2) - 0.5,
             color="C0", edgecolor="k", alpha=0.8)
axes[0].set_xlabel("Valid chromosomes in window")
axes[0].set_ylabel("Count")
axes[0].set_title("Window coverage")

window_counts_only = np.array([count for _, count in window_counts], dtype=np.int64)
axes[1].bar(range(len(window_counts)), window_counts_only, color="C1", alpha=0.8)
axes[1].set_xticks(range(len(window_counts)))
axes[1].set_xticklabels([cell_id for cell_id, _ in window_counts], rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("Number of windows")
axes[1].set_title("Windows per cell")

fig.suptitle("NRI-lite dataset construction")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Typed-node NRI-lite model
#
# The latent graph is inferred by an encoder:
#
# $$q(z_{ij} \mid x_{t-H+1:t})$$
#
# over all directed pairs of padded nodes.  Edge types are categorical:
#
# - `0`: null edge
# - `1`: active edge
#
# The decoder receives the sampled soft edge assignments and predicts the next
# chromosome velocity from the current node geometry.  The poles are kept as
# typed observed nodes, so the model can use them as senders even though the
# loss is only applied to chromosomes.

# %%
N_EDGE_TYPES = 2
TYPE_EMBED_DIM = 8
ENC_HIDDEN = 128
DEC_HIDDEN = 128
MSG_HIDDEN = 128
MSG_DIM = 128

LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
N_EPOCHS = 120
KL_WEIGHT = 1e-2
KL_WARMUP_EPOCHS = 40
NULL_EDGE_PRIOR = 0.9
TEMP_START = 1.0
TEMP_END = 0.5
PRINT_EVERY = 20


def directed_edge_index(num_nodes):
    senders = []
    receivers = []
    for sender in range(num_nodes):
        for receiver in range(num_nodes):
            if sender == receiver:
                continue
            senders.append(sender)
            receivers.append(receiver)
    return (
        torch.tensor(senders, dtype=torch.long),
        torch.tensor(receivers, dtype=torch.long),
    )


class TypedNRILite(nn.Module):
    """Minimal typed-node latent-graph model for next-step velocity prediction."""

    def __init__(
        self,
        num_nodes,
        node_types,
        history,
        edge_types=N_EDGE_TYPES,
        type_embed_dim=TYPE_EMBED_DIM,
        enc_hidden=ENC_HIDDEN,
        dec_hidden=DEC_HIDDEN,
        msg_hidden=MSG_HIDDEN,
        msg_dim=MSG_DIM,
        dt=DT,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.history = history
        self.edge_types = edge_types
        self.dt = float(dt)
        self.msg_dim = msg_dim

        senders, receivers = directed_edge_index(num_nodes)
        type_ids = torch.tensor(node_types, dtype=torch.long)

        self.register_buffer("senders", senders)
        self.register_buffer("receivers", receivers)
        self.register_buffer("type_ids", type_ids)
        self.register_buffer("sender_types", type_ids[senders])
        self.register_buffer("receiver_types", type_ids[receivers])

        self.type_embedding = nn.Embedding(2, type_embed_dim)

        enc_pos_dim = history * 3
        enc_vel_dim = (history - 1) * 3
        enc_input_dim = enc_pos_dim + enc_vel_dim + type_embed_dim
        self.encoder_node = mlp(enc_input_dim, enc_hidden, enc_hidden, depth=3)
        self.encoder_edge = mlp(
            2 * enc_hidden + 2 * type_embed_dim,
            enc_hidden,
            edge_types,
            depth=3,
        )

        self.message_mlps = nn.ModuleList(
            [
                mlp(
                    4 * 3 + 2 * 3 + 2 * type_embed_dim,
                    msg_hidden,
                    msg_dim,
                    depth=3,
                )
                for _ in range(edge_types - 1)
            ]
        )
        self.decoder_out = mlp(2 * 3 + type_embed_dim + msg_dim, dec_hidden, 3, depth=3)

    def _edge_mask(self, node_mask):
        return node_mask[:, self.senders] & node_mask[:, self.receivers]

    def encode(self, pos_hist, node_mask):
        B = pos_hist.shape[0]
        type_embed = self.type_embedding(self.type_ids).unsqueeze(0).expand(B, -1, -1)

        vel_hist = (pos_hist[:, 1:] - pos_hist[:, :-1]) / self.dt
        pos_feat = pos_hist.permute(0, 2, 1, 3).reshape(B, self.num_nodes, -1)
        vel_feat = vel_hist.permute(0, 2, 1, 3).reshape(B, self.num_nodes, -1)

        node_input = torch.cat([pos_feat, vel_feat, type_embed], dim=-1)
        node_input = node_input * node_mask.unsqueeze(-1)
        node_hidden = self.encoder_node(node_input)

        sender_hidden = node_hidden[:, self.senders]
        receiver_hidden = node_hidden[:, self.receivers]
        sender_type = type_embed[:, self.senders]
        receiver_type = type_embed[:, self.receivers]
        edge_input = torch.cat(
            [sender_hidden, receiver_hidden, sender_type, receiver_type],
            dim=-1,
        )

        logits = self.encoder_edge(edge_input)
        edge_mask = self._edge_mask(node_mask)

        # Invalid edges should become deterministic null edges.
        logits = logits.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        logits[..., 0] = torch.where(
            edge_mask,
            logits[..., 0],
            torch.zeros_like(logits[..., 0]),
        )
        return logits, edge_mask

    def sample_edges(self, logits, edge_mask, temperature):
        probs = logits.softmax(dim=-1)
        edges = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)

        null_edges = torch.zeros_like(edges)
        null_edges[..., 0] = 1.0
        probs = torch.where(edge_mask.unsqueeze(-1), probs, null_edges)
        edges = torch.where(edge_mask.unsqueeze(-1), edges, null_edges)
        return edges, probs

    def decode(self, pos_hist, node_mask, edges):
        B = pos_hist.shape[0]
        type_embed = self.type_embedding(self.type_ids).unsqueeze(0).expand(B, -1, -1)

        cur_pos = pos_hist[:, -1]
        prev_pos = pos_hist[:, -2]
        cur_vel = (cur_pos - prev_pos) / self.dt

        sender_pos = cur_pos[:, self.senders]
        receiver_pos = cur_pos[:, self.receivers]
        sender_vel = cur_vel[:, self.senders]
        receiver_vel = cur_vel[:, self.receivers]
        sender_type = type_embed[:, self.senders]
        receiver_type = type_embed[:, self.receivers]

        rel_pos = sender_pos - receiver_pos
        rel_vel = sender_vel - receiver_vel
        edge_input = torch.cat(
            [
                sender_pos,
                receiver_pos,
                sender_vel,
                receiver_vel,
                rel_pos,
                rel_vel,
                sender_type,
                receiver_type,
            ],
            dim=-1,
        )

        edge_mask = self._edge_mask(node_mask).unsqueeze(-1)
        agg = torch.zeros(B, self.num_nodes, self.msg_dim, device=pos_hist.device)
        receiver_index = self.receivers.view(1, -1, 1).expand(B, -1, self.msg_dim)

        for edge_type, msg_mlp in enumerate(self.message_mlps, start=1):
            msg = msg_mlp(edge_input)
            msg = msg * edges[..., edge_type : edge_type + 1] * edge_mask
            agg.scatter_add_(1, receiver_index, msg)

        node_input = torch.cat([cur_pos, cur_vel, type_embed, agg], dim=-1)
        node_input = node_input * node_mask.unsqueeze(-1)
        delta_vel = self.decoder_out(node_input)
        pred_vel = cur_vel + delta_vel
        return pred_vel

    def forward(self, pos_hist, node_mask, temperature):
        logits, edge_mask = self.encode(pos_hist, node_mask)
        edges, probs = self.sample_edges(logits, edge_mask, temperature)
        pred_vel = self.decode(pos_hist, node_mask, edges)
        return pred_vel, logits, probs, edge_mask


def masked_velocity_mse(pred_vel, target_vel, target_mask):
    mask = target_mask.unsqueeze(-1).float()
    sq_err = ((pred_vel - target_vel) ** 2) * mask
    denom = mask.sum() * pred_vel.shape[-1]
    if denom <= 0:
        return torch.tensor(float("nan"), device=pred_vel.device)
    return sq_err.sum() / denom


def categorical_kl(probs, edge_mask, null_prior=NULL_EDGE_PRIOR):
    if probs.numel() == 0:
        return torch.tensor(0.0, device=probs.device)

    valid_probs = probs[edge_mask]
    if valid_probs.numel() == 0:
        return torch.tensor(0.0, device=probs.device)

    if valid_probs.shape[-1] == 1:
        return torch.tensor(0.0, device=probs.device)

    prior = torch.full(
        (valid_probs.shape[-1],),
        (1.0 - null_prior) / (valid_probs.shape[-1] - 1),
        device=probs.device,
    )
    prior[0] = null_prior
    prior = prior.clamp_min(1e-8)

    valid_probs = valid_probs.clamp_min(1e-8)
    kl = valid_probs * (torch.log(valid_probs) - torch.log(prior))
    return kl.sum(dim=-1).mean()


def iterate_minibatches(data, batch_size, rng=None, shuffle=False):
    n = data["pos_hist"].shape[0]
    if n == 0:
        return

    indices = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield {
            "pos_hist": torch.tensor(data["pos_hist"][batch_idx], dtype=torch.float32, device=DEVICE),
            "node_mask": torch.tensor(data["node_mask"][batch_idx], dtype=torch.bool, device=DEVICE),
            "target_vel": torch.tensor(data["target_vel"][batch_idx], dtype=torch.float32, device=DEVICE),
            "target_mask": torch.tensor(data["target_mask"][batch_idx], dtype=torch.bool, device=DEVICE),
        }


def summarize_relation_probs(model, probs, edge_mask):
    """Average active-edge posterior by relation class."""
    active_prob = probs[..., 1:].sum(dim=-1)

    pole_to_chrom_mask = (
        edge_mask
        & (model.sender_types == TYPE_POLE).unsqueeze(0)
        & (model.receiver_types == TYPE_CHROM).unsqueeze(0)
    )
    chrom_to_chrom_mask = (
        edge_mask
        & (model.sender_types == TYPE_CHROM).unsqueeze(0)
        & (model.receiver_types == TYPE_CHROM).unsqueeze(0)
    )

    def _mean(mask):
        if mask.any():
            return float(active_prob[mask].mean().item())
        return np.nan

    pole_active = _mean(pole_to_chrom_mask)
    chrom_active = _mean(chrom_to_chrom_mask)
    pole_fraction = (
        pole_active / (pole_active + chrom_active)
        if np.isfinite(pole_active) and np.isfinite(chrom_active) and (pole_active + chrom_active) > 0
        else np.nan
    )

    return {
        "mean_active_pole_to_chrom": pole_active,
        "mean_active_chrom_to_chrom": chrom_active,
        "pole_fraction": pole_fraction,
    }


# %% [markdown]
# ## Leave-one-cell-out training
#
# Each fold:
#
# 1. Holds out one cell's windows
# 2. Trains the latent-graph model on the remaining cells
# 3. Reports held-out velocity MSE
# 4. Summarizes the posterior edge activity for:
#    - pole -> chromosome edges
#    - chromosome -> chromosome edges
#
# If the pole->chromosome posterior is consistently stronger, the neural model
# is independently favoring the poles-driven topology.

# %%
print(
    f"Model: history={HISTORY}, edge_types={N_EDGE_TYPES}, "
    f"epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, "
    f"kl_weight={KL_WEIGHT}, null_prior={NULL_EDGE_PRIOR}"
)


def train_fold(train_windows, test_windows, fold_seed, verbose=True):
    if not train_windows:
        raise ValueError("Train fold has no windows.")
    if not test_windows:
        raise ValueError("Test fold has no windows.")

    set_seed(fold_seed)
    rng = np.random.default_rng(fold_seed)

    model = TypedNRILite(
        num_nodes=MAX_NODES,
        node_types=NODE_TYPES,
        history=HISTORY,
        edge_types=N_EDGE_TYPES,
        dt=DT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_data = pack_windows(train_windows)
    test_data = pack_windows(test_windows)

    temp_decay = (TEMP_END / TEMP_START) ** (1.0 / max(1, N_EPOCHS - 1))

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_batches = 0

        temperature = max(TEMP_END, TEMP_START * (temp_decay ** epoch))
        beta = KL_WEIGHT * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)

        for batch in iterate_minibatches(train_data, BATCH_SIZE, rng=rng, shuffle=True):
            pred_vel, _, probs, edge_mask = model(
                batch["pos_hist"],
                batch["node_mask"],
                temperature=temperature,
            )
            recon = masked_velocity_mse(pred_vel, batch["target_vel"], batch["target_mask"])
            kl = categorical_kl(probs, edge_mask)
            loss = recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += float(recon.item())
            epoch_kl += float(kl.item())
            epoch_batches += 1

        if verbose and (epoch + 1) % PRINT_EVERY == 0:
            print(
                f"  epoch {epoch+1:4d}  "
                f"recon={epoch_recon / max(epoch_batches, 1):.6f}  "
                f"kl={epoch_kl / max(epoch_batches, 1):.6f}  "
                f"temp={temperature:.3f}"
            )

    model.eval()
    mse_num = 0.0
    mse_den = 0
    pole_vals = []
    chrom_vals = []
    frac_vals = []

    with torch.no_grad():
        for batch in iterate_minibatches(test_data, BATCH_SIZE, shuffle=False):
            pred_vel, _, probs, edge_mask = model(
                batch["pos_hist"],
                batch["node_mask"],
                temperature=TEMP_END,
            )

            mask = batch["target_mask"].unsqueeze(-1).float()
            sq_err = ((pred_vel - batch["target_vel"]) ** 2) * mask
            mse_num += float(sq_err.sum().item())
            mse_den += int(mask.sum().item() * pred_vel.shape[-1])

            rel_stats = summarize_relation_probs(model, probs, edge_mask)
            pole_vals.append(rel_stats["mean_active_pole_to_chrom"])
            chrom_vals.append(rel_stats["mean_active_chrom_to_chrom"])
            frac_vals.append(rel_stats["pole_fraction"])

    test_mse = mse_num / mse_den if mse_den > 0 else np.nan

    return {
        "test_mse": test_mse,
        "mean_active_pole_to_chrom": float(np.nanmean(pole_vals)),
        "mean_active_chrom_to_chrom": float(np.nanmean(chrom_vals)),
        "pole_fraction": float(np.nanmean(frac_vals)),
        "model": model,
    }


# %%
n_cells = len(cells)
print(f"Running {n_cells}-fold leave-one-cell-out CV...")

cv_results = []
for fold in range(n_cells):
    held_out_id = cells[fold].cell_id
    print(f"Fold {fold+1}/{n_cells} (held out: {held_out_id})")

    train_windows = [w for w in windows if w["cell_idx"] != fold]
    test_windows = [w for w in windows if w["cell_idx"] == fold]
    print(f"  train windows={len(train_windows)}, test windows={len(test_windows)}")

    result = train_fold(train_windows, test_windows, fold_seed=SEED + fold)
    cv_results.append(result)

    print(
        f"  test MSE={result['test_mse']:.6f}, "
        f"pole->chrom active={result['mean_active_pole_to_chrom']:.3f}, "
        f"chrom->chrom active={result['mean_active_chrom_to_chrom']:.3f}, "
        f"pole fraction={result['pole_fraction']:.3f}\n"
    )


# %% [markdown]
# ## NRI-lite results

# %%
test_mses = np.array([r["test_mse"] for r in cv_results], dtype=np.float64)
pole_active = np.array([r["mean_active_pole_to_chrom"] for r in cv_results], dtype=np.float64)
chrom_active = np.array([r["mean_active_chrom_to_chrom"] for r in cv_results], dtype=np.float64)
pole_fractions = np.array([r["pole_fraction"] for r in cv_results], dtype=np.float64)

print("=== Leave-one-cell-out NRI-lite summary ===")
print(f"Test MSE:             {np.nanmean(test_mses):.6f} +/- {np.nanstd(test_mses):.6f}")
print(f"Pole->chrom active:   {np.nanmean(pole_active):.3f} +/- {np.nanstd(pole_active):.3f}")
print(f"Chrom->chrom active:  {np.nanmean(chrom_active):.3f} +/- {np.nanstd(chrom_active):.3f}")
print(f"Pole fraction:        {np.nanmean(pole_fractions):.3f} +/- {np.nanstd(pole_fractions):.3f}")
print()

if np.nanmean(pole_fractions) > 0.7:
    print("=> The latent graph strongly favors pole->chromosome edges.")
    print("   This is consistent with the SFI poles topology.")
elif np.nanmean(pole_fractions) > 0.5:
    print("=> Pole->chromosome edges are stronger on average, but chrom-chrom edges remain non-trivial.")
else:
    print("=> Chromosome-chromosome latent edges are comparably active.")


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cell_ids = [cell.cell_id for cell in cells]
x = np.arange(n_cells)

ax = axes[0]
ax.bar(x, pole_fractions, color="C0", alpha=0.8)
ax.axhline(np.nanmean(pole_fractions), color="k", linestyle="--", linewidth=1,
           label=f"mean = {np.nanmean(pole_fractions):.2f}")
ax.axhline(0.5, color="0.7", linestyle=":", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(cell_ids, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Pole-edge fraction")
ax.set_title("Held-out pole-edge dominance")
ax.set_ylim(0, 1)
ax.legend()

ax = axes[1]
w = 0.35
ax.bar(x - w / 2, pole_active, w, label="pole -> chrom", color="C0", alpha=0.8)
ax.bar(x + w / 2, chrom_active, w, label="chrom -> chrom", color="C1", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cell_ids, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Mean active-edge posterior")
ax.set_title("Relation-class posterior activity")
ax.legend()

fig.suptitle("NRI-lite latent topology summary")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Comparison with SFI cross-validation
#
# We compare the neural held-out one-step velocity error with the winning SFI
# topology (`poles`) from NB04/05.

# %%
from chromlearn.model_fitting import FitConfig
from chromlearn.model_fitting.fit import cross_validate

sfi_config = FitConfig(
    topology="poles",
    n_basis_xx=10,
    n_basis_xy=10,
    r_min_xx=0.3,
    r_max_xx=15.0,
    r_min_xy=0.3,
    r_max_xy=15.0,
    basis_type="bspline",
    lambda_ridge=1e-3,
    lambda_rough=1e-3,
    basis_eval_mode="ito",
    dt=DT,
)
sfi_cv = cross_validate(cells, sfi_config)

print("=== CV error comparison ===")
print(f"SFI (poles, B-spline): {sfi_cv.mean_error:.6f} +/- {sfi_cv.std_error:.6f}")
print(f"NRI-lite:              {np.nanmean(test_mses):.6f} +/- {np.nanstd(test_mses):.6f}")


# %% [markdown]
# ## Notes
#
# - This notebook is much closer to NRI than the earlier fixed-channel version
#   because the relation graph is now **latent and discrete**.
# - It is still deliberately lighter than the original NRI implementation:
#   next-step prediction only, typed poles, padded masks for variable node
#   counts, and a small decoder.
# - The key readout is not a learned radial kernel but the **posterior edge
#   activity** for pole->chromosome versus chromosome->chromosome relations.
#
# If the held-out pole fraction stays high while the prediction error is
# competitive, that is the neural analogue of the SFI poles-topology result.
