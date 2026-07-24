"""Experiment 128: Vacuity controls for all 6 inductive-bias variants.

Tests whether sparse overcomplete VAE variants pass the noise control where
the standard Structured VAE fails. Corrects critical bugs from exp73:
  1. Uses model's own classifier for IIA (not post-hoc LogReg)
     - Models with choice_head: use choice_head directly
     - PiVAE (no head): posterior-prior log-likelihood scoring
  2. Pairs by evidence_labels (not choice_labels)
  3. Z-scores activity on TRAIN split only, applies to test
  4. Random pair sampling (not sequential)
  5. Filters no-go trials (response != 0)
  6. Shared train/test splits and intervention pairs across models
  7. Multiple matched replicates (N_REPLICATES seeds)

Design: 6 models x 2 data conditions (real/noise) x 2 label conditions
(real/shuffled) x 3 training modes (trained/untrained/recon_only)
x N_REPLICATES seeds, per region-session.

See PREREGISTRATION_exp128_vacuity_ablation.md for full design and predictions.
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results_batch4" / "exp128"

MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0
L1_COEFF = 1e-3
EXPANSION_FACTOR = 8
N_IIA_PAIRS = 100
N_REPLICATES = 3


# ---------------------------------------------------------------------------
# Model classes (copied from exp73 to avoid import issues on Modal)
# ---------------------------------------------------------------------------

class StructuredVAE(nn.Module):
    """Model A: structured split + Gaussian prior + classification head."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim
        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)
        self._has_structured_encode = False

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits

    def loss(self, x, labels, alpha_choice=ALPHA_CHOICE):
        recon, mu, logvar, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * kl + alpha_choice * choice_loss


class PiVAE(nn.Module):
    """Model B: label-conditional prior, no structured split, no classifier."""
    def __init__(self, n_neurons, z_dim=Z_CHOICE_DIM + Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = Z_CHOICE_DIM
        self.z_dim = z_dim
        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.prior_mu = nn.Embedding(2, z_dim)
        self.prior_logvar = nn.Embedding(2, z_dim)
        self._has_structured_encode = False

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, x, labels, alpha_choice=ALPHA_CHOICE):
        recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl = -0.5 * (1 + logvar - prior_lv
                      - ((mu - prior_mu).pow(2) + logvar.exp()) / prior_lv.exp()).mean()
        return recon_loss + BETA_KL * kl


class PiStructuredVAE(nn.Module):
    """Model C: structured split + label-conditional prior on z_choice."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim
        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)
        self.prior_mu = nn.Embedding(2, z_choice_dim)
        self.prior_logvar = nn.Embedding(2, z_choice_dim)
        self._has_structured_encode = False

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits

    def loss(self, x, labels, alpha_choice=ALPHA_CHOICE):
        recon, mu, logvar, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        mu_c, logvar_c = mu[:, :self.z_choice_dim], logvar[:, :self.z_choice_dim]
        mu_o, logvar_o = mu[:, self.z_choice_dim:], logvar[:, self.z_choice_dim:]
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl_c = -0.5 * (1 + logvar_c - prior_lv
                        - ((mu_c - prior_mu).pow(2) + logvar_c.exp()) / prior_lv.exp()).mean()
        kl_o = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * (kl_c + kl_o) + alpha_choice * choice_loss


class SAEStructured(nn.Module):
    """Model D: structured split + Gaussian prior + overcomplete z_choice + L1."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM,
                 expansion=EXPANSION_FACTOR):
        super().__init__()
        self.z_choice_dim = z_choice_dim * expansion
        self.z_other_dim = z_other_dim
        z_dim = self.z_choice_dim + z_other_dim
        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(self.z_choice_dim, 2)
        self._has_structured_encode = False

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits

    def loss(self, x, labels, alpha_choice=ALPHA_CHOICE, l1_coeff=L1_COEFF):
        recon, mu, logvar, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        sparsity = mu[:, :self.z_choice_dim].abs().mean()
        return recon_loss + BETA_KL * kl + alpha_choice * choice_loss + l1_coeff * sparsity


class PiSAEPlain(nn.Module):
    """Model E: label prior + overcomplete + L1, NO structured split."""
    def __init__(self, n_neurons, expansion=EXPANSION_FACTOR):
        super().__init__()
        self.z_choice_dim = Z_CHOICE_DIM * expansion
        self.z_dim = self.z_choice_dim
        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, self.z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, self.z_dim)
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.classifier = nn.Linear(self.z_dim, 2)
        self.prior_mu = nn.Embedding(2, self.z_dim)
        self.prior_logvar = nn.Embedding(2, self.z_dim)
        self._has_structured_encode = False

    def encode(self, x):
        h = self.enc_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits

    def loss(self, x, labels, alpha_choice=ALPHA_CHOICE, l1_coeff=L1_COEFF):
        recon, mu, logvar, logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl = -0.5 * (1 + logvar - prior_lv
                      - ((mu - prior_mu).pow(2) + logvar.exp()) / prior_lv.exp()).mean()
        ce = F.cross_entropy(logits, labels)
        sparsity = mu.abs().mean()
        return recon_loss + BETA_KL * kl + alpha_choice * ce + l1_coeff * sparsity


class PiSAEStructured(nn.Module):
    """Model F: structured split + label prior on z_choice + overcomplete + L1."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM,
                 expansion=EXPANSION_FACTOR):
        super().__init__()
        self.z_choice_dim = z_choice_dim * expansion
        self.z_other_dim = z_other_dim
        z_dim = self.z_choice_dim + z_other_dim
        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.enc_choice_mu = nn.Linear(HIDDEN_DIM, self.z_choice_dim)
        self.enc_choice_logvar = nn.Linear(HIDDEN_DIM, self.z_choice_dim)
        self.enc_other_mu = nn.Linear(HIDDEN_DIM, z_other_dim)
        self.enc_other_logvar = nn.Linear(HIDDEN_DIM, z_other_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(self.z_choice_dim, 2)
        self.prior_mu = nn.Embedding(2, self.z_choice_dim)
        self.prior_logvar = nn.Embedding(2, self.z_choice_dim)
        self._has_structured_encode = True

    def encode(self, x):
        h = self.enc_trunk(x)
        return (self.enc_choice_mu(h), self.enc_choice_logvar(h),
                self.enc_other_mu(h), self.enc_other_logvar(h))

    def forward(self, x):
        mu_c, lv_c, mu_o, lv_o = self.encode(x)
        z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
        z_o = mu_o + torch.exp(0.5 * lv_o) * torch.randn_like(lv_o)
        z = torch.cat([z_c, z_o], dim=-1)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z_c)
        return recon, mu_c, lv_c, mu_o, lv_o, choice_logits

    def loss(self, x, labels, alpha_choice=ALPHA_CHOICE, l1_coeff=L1_COEFF):
        recon, mu_c, lv_c, mu_o, lv_o, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl_c = -0.5 * (1 + lv_c - prior_lv
                        - ((mu_c - prior_mu).pow(2) + lv_c.exp()) / prior_lv.exp()).mean()
        kl_o = -0.5 * torch.mean(1 + lv_o - mu_o.pow(2) - lv_o.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        sparsity = mu_c.abs().mean()
        return recon_loss + BETA_KL * (kl_c + kl_o) + alpha_choice * choice_loss + l1_coeff * sparsity


MODEL_SPECS = {
    "structured_vae": {"cls": StructuredVAE, "has_classifier": True, "sparse": False},
    "pi_vae": {"cls": PiVAE, "has_classifier": False, "sparse": False},
    "pi_structured_vae": {"cls": PiStructuredVAE, "has_classifier": True, "sparse": False},
    "sae_structured": {"cls": SAEStructured, "has_classifier": True, "sparse": True},
    "pi_sae_plain": {"cls": PiSAEPlain, "has_classifier": True, "sparse": True},
    "pi_sae_structured": {"cls": PiSAEStructured, "has_classifier": True, "sparse": True},
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _zscore_fit(activity):
    """Compute z-score parameters from training data."""
    mu = activity.mean(axis=0, keepdims=True)
    std = activity.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mu, std


def _zscore_apply(activity, mu, std):
    """Apply pre-computed z-score parameters."""
    return (activity - mu) / std


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    evidence = cr[:n] - cl[:n]
    labels = np.full(n, -1, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    return labels


def _make_shared_splits(evidence_labels, rng):
    """Create train/test split and intervention pairs shared across all models.

    Returns dict with train_idx, test_idx, pair_left, pair_right.
    All models within a region-session-replicate use these same indices.
    """
    n = len(evidence_labels)
    indices = np.arange(n)

    left_all = np.where(evidence_labels == 0)[0]
    right_all = np.where(evidence_labels == 1)[0]

    # Stratified 50/50 split
    n_left_train = len(left_all) // 2
    n_right_train = len(right_all) // 2

    left_perm = rng.permutation(left_all)
    right_perm = rng.permutation(right_all)

    train_idx = np.concatenate([left_perm[:n_left_train], right_perm[:n_right_train]])
    test_idx = np.concatenate([left_perm[n_left_train:], right_perm[n_right_train:]])

    # Intervention pairs from test set only
    test_evidence = evidence_labels[test_idx]
    test_left = np.where(test_evidence == 0)[0]
    test_right = np.where(test_evidence == 1)[0]

    n_pairs = min(N_IIA_PAIRS, len(test_left), len(test_right))
    if n_pairs < 5:
        return None

    pair_left = rng.choice(test_left, n_pairs, replace=False)
    pair_right = rng.choice(test_right, n_pairs, replace=False)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "pair_left": pair_left,
        "pair_right": pair_right,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_model(model, activity, labels, device, alpha_choice=ALPHA_CHOICE,
                 l1_coeff=L1_COEFF):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=len(dataset) > BATCH_SIZE,
    )
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            kwargs = {}
            if hasattr(model.loss, "__code__") and "alpha_choice" in model.loss.__code__.co_varnames:
                kwargs["alpha_choice"] = alpha_choice
            if hasattr(model.loss, "__code__") and "l1_coeff" in model.loss.__code__.co_varnames:
                kwargs["l1_coeff"] = l1_coeff
            loss = model.loss(xb, yb, **kwargs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Corrected IIA computation
# ---------------------------------------------------------------------------

def _get_z_choice(model, X_tensor):
    """Extract z_choice (mu) from any model architecture."""
    with torch.no_grad():
        if model._has_structured_encode:
            mu_c, _, _, _ = model.encode(X_tensor)
            return mu_c
        else:
            mu, _ = model.encode(X_tensor)
            return mu[:, :model.z_choice_dim]


def _make_predict_fn(model, spec):
    """Return a prediction function using the model's own learned classifier.

    Models A,C,D,F: use choice_head (nn.Linear on z_choice).
    Model E: use classifier (nn.Linear on z_choice).
    Model B (PiVAE): use posterior-prior log-likelihood scoring.
      For z_choice, compute log N(z | prior_mu[y], exp(prior_logvar[y]))
      for y in {0,1} and predict argmax. This is PiVAE's own learned
      discrimination — the conditional prior IS its classifier.
    """
    device = next(model.parameters()).device

    if spec["has_classifier"]:
        head = getattr(model, "choice_head", None) or getattr(model, "classifier", None)

        def predict_fn(z_choice_tensor):
            with torch.no_grad():
                logits = head(z_choice_tensor.to(device))
                return logits.argmax(dim=1).cpu().numpy()

        return predict_fn

    # PiVAE: posterior-prior scoring
    prior_mu_0 = model.prior_mu.weight[0].detach()
    prior_mu_1 = model.prior_mu.weight[1].detach()
    prior_lv_0 = model.prior_logvar.weight[0].detach()
    prior_lv_1 = model.prior_logvar.weight[1].detach()
    z_dim = model.z_choice_dim

    def predict_fn(z_choice_tensor):
        with torch.no_grad():
            z = z_choice_tensor[:, :z_dim].to(device)
            # log N(z | mu, sigma^2) = -0.5 * (logvar + (z-mu)^2/var)
            ll_0 = -0.5 * (prior_lv_0[:z_dim] + (z - prior_mu_0[:z_dim]).pow(2)
                           / prior_lv_0[:z_dim].exp()).sum(dim=1)
            ll_1 = -0.5 * (prior_lv_1[:z_dim] + (z - prior_mu_1[:z_dim]).pow(2)
                           / prior_lv_1[:z_dim].exp()).sum(dim=1)
            return (ll_1 > ll_0).long().cpu().numpy()

    return predict_fn


def _compute_sparsity_metrics(z_choice_np, train_std):
    """Multiple sparsity metrics, threshold-free and threshold-based.

    train_std: per-dimension std from training set for scale-relative threshold.
    """
    abs_z = np.abs(z_choice_np)
    mean_abs = float(np.mean(abs_z))

    # Scale-relative L0: |z_j| > 0.1 * std_j (training set)
    threshold = 0.1 * np.maximum(train_std, 1e-6)
    l0_relative = float(np.mean(abs_z > threshold))

    # Hoyer sparsity: (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    n_dims = z_choice_np.shape[1]
    l1_norms = np.sum(abs_z, axis=1)
    l2_norms = np.sqrt(np.sum(z_choice_np ** 2, axis=1))
    l2_norms = np.maximum(l2_norms, 1e-10)
    hoyer_per_sample = (np.sqrt(n_dims) - l1_norms / l2_norms) / (np.sqrt(n_dims) - 1)
    hoyer = float(np.mean(np.clip(hoyer_per_sample, 0, 1)))

    # Top-k concentration: fraction of total activation in top 3 dims
    sorted_abs = np.sort(abs_z, axis=1)[:, ::-1]
    total_act = np.sum(abs_z, axis=1, keepdims=True)
    total_act = np.maximum(total_act, 1e-10)
    top3_frac = float(np.mean(np.sum(sorted_abs[:, :3], axis=1, keepdims=True) / total_act))

    # Per-feature activation frequency (fraction of samples where feature active)
    feature_freq = np.mean(abs_z > threshold, axis=0)

    return {
        "mean_abs_activation": mean_abs,
        "l0_scale_relative": l0_relative,
        "hoyer_sparsity": hoyer,
        "top3_concentration": top3_frac,
        "n_active_features_mean": float(np.sum(abs_z > threshold, axis=1).mean()),
        "feature_activation_freq": feature_freq.tolist(),
    }


def _compute_iia_corrected(model, spec, activity, evidence_labels, splits,
                           device):
    """Corrected IIA with shared splits, model's own classifier, bidirectional."""
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    z_choice = _get_z_choice(model, X)
    z_choice_np = z_choice.cpu().numpy()

    predict_fn = _make_predict_fn(model, spec)

    test_idx = splits["test_idx"]
    pair_left = splits["pair_left"]
    pair_right = splits["pair_right"]

    eval_z = z_choice[test_idx]
    eval_evidence = evidence_labels[test_idx]

    flips = 0
    total = 0
    for li, ri in zip(pair_left, pair_right):
        z_l = eval_z[li].unsqueeze(0)
        z_r = eval_z[ri].unsqueeze(0)

        orig_pred_l = predict_fn(z_l)[0]
        orig_pred_r = predict_fn(z_r)[0]
        swap_pred_l = predict_fn(z_r)[0]
        swap_pred_r = predict_fn(z_l)[0]

        if swap_pred_l != orig_pred_l:
            flips += 1
        if swap_pred_r != orig_pred_r:
            flips += 1
        total += 2

    iia = flips / total if total > 0 else float("nan")

    # Classification accuracy on test set
    all_preds = predict_fn(eval_z)
    acc = float(np.mean(all_preds == eval_evidence))

    # Reconstruction MSE (on test set)
    X_test = X[test_idx]
    with torch.no_grad():
        if model._has_structured_encode:
            mu_c, _, mu_o, _ = model.encode(X_test)
            z = torch.cat([mu_c, mu_o], dim=-1)
        else:
            mu, _ = model.encode(X_test)
            z = mu
        recon = model.decoder(z)
        mse = F.mse_loss(recon, X_test).item()

    # Sparsity metrics — threshold relative to z_choice std on train set
    train_idx = splits["train_idx"]
    train_z_std = np.std(z_choice_np[train_idx], axis=0)
    sparsity = _compute_sparsity_metrics(z_choice_np[test_idx], train_z_std)

    # Diversity: 1 - mean cosine similarity within each class
    cos_sims = []
    for label_val in [0, 1]:
        mask = eval_evidence == label_val
        if mask.sum() < 2:
            continue
        z_class = eval_z[mask].cpu().numpy()
        norms = np.linalg.norm(z_class, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        z_normed = z_class / norms
        sim_matrix = z_normed @ z_normed.T
        n = sim_matrix.shape[0]
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        cos_sims.extend(upper_tri.tolist())
    diversity = 1.0 - float(np.mean(cos_sims)) if cos_sims else float("nan")

    metrics = {
        "classification_accuracy": acc,
        "reconstruction_mse": mse,
        "latent_cosine_spread": diversity,
        **{f"sparsity_{k}": v for k, v in sparsity.items()
           if k != "feature_activation_freq"},
    }
    return iia, metrics


# ---------------------------------------------------------------------------
# Data loading (RAW — z-scoring happens per-split)
# ---------------------------------------------------------------------------

def _load_region_data(sessions):
    """Load per-region data with evidence labels, excluding no-go trials.

    Returns RAW activity (not z-scored). Z-scoring is done per train/test
    split to prevent information leakage.
    """
    region_data = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        response = sess["response"]
        n_spks = sess["spks"].shape[2]
        n = min(n_spks, len(response))

        evidence_labels = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue

        # Filter no-go trials (response == 0) BEFORE truncation
        valid_response = response[:n] != 0
        valid_evidence = evidence_labels[:n] >= 0
        valid = valid_response & valid_evidence

        if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
            continue

        choice_labels = (response[:n] > 0).astype(int)
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            act_n = min(act.shape[0], n)
            activity = act[:act_n, :, TIME_WINDOW].mean(axis=2)
            valid_n = valid[:act_n]

            # Store RAW activity — z-score later per split
            activity_valid = activity[valid_n]
            ev_valid = evidence_labels[:act_n][valid_n]
            ch_valid = choice_labels[:act_n][valid_n]

            if len(np.unique(ev_valid)) < 2:
                continue
            left_count = (ev_valid == 0).sum()
            right_count = (ev_valid == 1).sum()
            if left_count < MIN_TRIALS_PER_CONDITION or right_count < MIN_TRIALS_PER_CONDITION:
                continue

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "activity_raw": activity_valid,
                "evidence_labels": ev_valid,
                "choice_labels": ch_valid,
                "n_neurons": int(activity_valid.shape[1]),
                "mouse": mouse,
                "session_idx": sess_idx,
            })

    return region_data


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

DATA_CONDITIONS = ["real", "noise"]
LABEL_CONDITIONS = ["real", "shuffled"]
TRAINING_MODES = ["trained", "untrained", "recon_only"]


def _run_single_condition(active_specs, activity_raw, evidence_labels,
                          data_cond, label_cond, training_mode,
                          splits, device, l1_coeff=L1_COEFF):
    """Run ALL models on one (data x label x mode) condition with shared splits.

    Z-scores on train split, applies to test. Same splits/pairs for all models.
    Returns dict of model_name -> result.
    """
    # Prepare data condition
    if data_cond == "noise":
        input_raw = np.random.randn(*activity_raw.shape).astype(np.float32)
    else:
        input_raw = activity_raw

    if label_cond == "shuffled":
        ev = evidence_labels.copy()
        np.random.shuffle(ev)
    else:
        ev = evidence_labels

    # Z-score: fit on TRAIN split only
    train_idx = splits["train_idx"]
    zs_mu, zs_std = _zscore_fit(input_raw[train_idx])
    input_zscored = _zscore_apply(input_raw, zs_mu, zs_std)

    train_activity = input_zscored[train_idx]
    train_labels = ev[train_idx]

    results = {}
    for model_name, spec in active_specs.items():
        n_neurons = input_zscored.shape[1]
        model = spec["cls"](n_neurons).to(device)

        if training_mode == "trained":
            _train_model(model, train_activity, train_labels, device,
                         l1_coeff=l1_coeff)
        elif training_mode == "recon_only":
            _train_model(model, train_activity, train_labels, device,
                         alpha_choice=0.0, l1_coeff=l1_coeff)

        iia, metrics = _compute_iia_corrected(
            model, spec, input_zscored, ev, splits, device,
        )
        results[model_name] = {
            "iia": float(iia), **{k: float(v) if isinstance(v, (int, float)) else v
                                  for k, v in metrics.items()},
        }

    return results


def _hierarchical_bootstrap(region_results, region_mouse_map, key_real, key_noise,
                            n_bootstrap=10000):
    """Bootstrap clustered by mouse. Returns 95% CI on IIA gap."""
    regions = sorted(set(region_results.keys()))
    mice = sorted(set(region_mouse_map[r] for r in regions
                      if r in region_mouse_map))
    mouse_to_regions = {}
    for r in regions:
        m = region_mouse_map.get(r, "unknown")
        if m not in mouse_to_regions:
            mouse_to_regions[m] = []
        mouse_to_regions[m].append(r)

    gaps = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        boot_mice = rng.choice(mice, size=len(mice), replace=True)
        boot_gaps = []
        for m in boot_mice:
            m_regions = mouse_to_regions.get(m, [])
            if not m_regions:
                continue
            boot_regions = rng.choice(m_regions, size=len(m_regions), replace=True)
            for r in boot_regions:
                rr = region_results.get(r, {})
                real_iia = rr.get(key_real, {}).get("iia")
                noise_iia = rr.get(key_noise, {}).get("iia")
                if real_iia is not None and noise_iia is not None:
                    if not (np.isnan(real_iia) or np.isnan(noise_iia)):
                        boot_gaps.append(real_iia - noise_iia)
        if boot_gaps:
            gaps.append(np.mean(boot_gaps))

    if not gaps:
        return {"ci_lower": None, "ci_upper": None, "n_bootstrap": 0}
    gaps = np.array(gaps)
    return {
        "ci_lower": float(np.percentile(gaps, 2.5)),
        "ci_upper": float(np.percentile(gaps, 97.5)),
        "mean": float(np.mean(gaps)),
        "n_bootstrap": len(gaps),
    }


def run(max_sessions=None, model_filter=None, l1_sweep=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_dir = RESULTS_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting exp128 vacuity ablation "
                f"on {device} with {len(sessions)} sessions, "
                f"{N_REPLICATES} replicates per condition")

    region_data = _load_region_data(sessions)
    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    if model_filter:
        active_specs = {model_filter: MODEL_SPECS[model_filter]}
    else:
        active_specs = MODEL_SPECS

    l1_values = [1e-4, 1e-3, 1e-2] if l1_sweep else [L1_COEFF]

    region_mouse_map = {}
    for region, measurements in region_data.items():
        mice = [m["mouse"] for m in measurements]
        region_mouse_map[region] = mice[0] if mice else "unknown"

    incremental_path = RESULTS_DIR / "exp128_incremental.jsonl"
    all_region_results = {}
    n_fits_total = 0
    fit_times = []

    for region in tqdm(sorted(region_data.keys()), desc="Regions"):
        measurements = region_data[region]
        region_result = {}

        for sess_i, m_data in enumerate(measurements):
            activity_raw = m_data["activity_raw"]
            evidence_labels = m_data["evidence_labels"]

            for rep in range(N_REPLICATES):
                # Fixed seed per replicate: deterministic, reproducible
                rep_seed = hash((region, m_data["session_idx"], rep)) % (2**31)
                rng = np.random.default_rng(rep_seed)

                # Shared splits for ALL models in this replicate
                splits = _make_shared_splits(evidence_labels, rng)
                if splits is None:
                    continue

                # Save audit trail for first replicate of first session
                if sess_i == 0 and rep == 0:
                    audit_record = {
                        "region": region,
                        "session_idx": m_data["session_idx"],
                        "replicate": rep,
                        "seed": rep_seed,
                        "n_trials": len(evidence_labels),
                        "train_idx": splits["train_idx"].tolist(),
                        "test_idx": splits["test_idx"].tolist(),
                        "pair_left": splits["pair_left"].tolist(),
                        "pair_right": splits["pair_right"].tolist(),
                        "evidence_labels": evidence_labels.tolist(),
                    }
                    with open(audit_dir / f"{region}_audit.json", "w") as f:
                        json.dump(audit_record, f, indent=2)

                # Set torch seed for model init reproducibility
                torch.manual_seed(rep_seed)

                for data_cond in DATA_CONDITIONS:
                    for label_cond in LABEL_CONDITIONS:
                        for mode in TRAINING_MODES:
                            for l1_val in l1_values:
                                # Skip non-sparse L1 variations
                                if l1_val != L1_COEFF and not l1_sweep:
                                    continue

                                t0 = time.time()

                                # Set per-condition seed for noise/shuffle
                                cond_seed = hash((rep_seed, data_cond, label_cond)) % (2**31)
                                np.random.seed(cond_seed)

                                cond_results = _run_single_condition(
                                    active_specs, activity_raw,
                                    evidence_labels, data_cond, label_cond,
                                    mode, splits, device, l1_coeff=l1_val,
                                )

                                dt = time.time() - t0
                                fit_times.append(dt)
                                n_fits_total += len(active_specs)

                                for model_name, result in cond_results.items():
                                    sparse = active_specs[model_name]["sparse"]
                                    if not sparse and l1_val != L1_COEFF:
                                        continue

                                    key = f"{model_name}__{data_cond}__{label_cond}__{mode}"
                                    if l1_sweep and sparse:
                                        key += f"__l1={l1_val}"

                                    if key not in region_result:
                                        region_result[key] = []
                                    region_result[key].append(result)

        # Average across sessions x replicates for this region
        region_avg = {}
        for key, results_list in region_result.items():
            iias = [r["iia"] for r in results_list if not np.isnan(r["iia"])]
            accs = [r["classification_accuracy"] for r in results_list
                    if not np.isnan(r["classification_accuracy"])]
            mses = [r["reconstruction_mse"] for r in results_list]
            divs = [r["latent_cosine_spread"] for r in results_list
                    if not np.isnan(r["latent_cosine_spread"])]
            region_avg[key] = {
                "iia": float(np.mean(iias)) if iias else float("nan"),
                "iia_std": float(np.std(iias)) if len(iias) > 1 else float("nan"),
                "classification_accuracy": float(np.mean(accs)) if accs else float("nan"),
                "reconstruction_mse": float(np.mean(mses)) if mses else float("nan"),
                "latent_cosine_spread": float(np.mean(divs)) if divs else float("nan"),
                "n_replicates": len(results_list),
            }
            # Aggregate sparsity if present
            hoyers = [r.get("sparsity_hoyer_sparsity", float("nan"))
                      for r in results_list]
            hoyers = [h for h in hoyers if not np.isnan(h)]
            if hoyers:
                region_avg[key]["hoyer_sparsity"] = float(np.mean(hoyers))
                region_avg[key]["l0_scale_relative"] = float(np.mean(
                    [r.get("sparsity_l0_scale_relative", 0) for r in results_list]
                ))
                region_avg[key]["mean_abs_activation"] = float(np.mean(
                    [r.get("sparsity_mean_abs_activation", 0) for r in results_list]
                ))

        all_region_results[region] = region_avg

        with open(incremental_path, "a") as f:
            f.write(json.dumps({"region": region, **region_avg}, default=str) + "\n")

        trained_keys = [k for k in region_avg if "__trained" in k
                        and "__real__real" in k and "l1=" not in k]
        iia_str = " | ".join(
            f"{k.split('__')[0]}={region_avg[k]['iia']:.3f}" for k in sorted(trained_keys)
        )
        logger.info(f"{datetime.now().isoformat()} {region}: {iia_str}")

    # --- Precondition checks ---
    preconditions = {}
    for model_name in active_specs:
        untrained_key = f"{model_name}__real__real__untrained"
        recon_key = f"{model_name}__real__real__recon_only"

        untrained_iias = [
            all_region_results[r][untrained_key]["iia"]
            for r in all_region_results
            if untrained_key in all_region_results[r]
            and not np.isnan(all_region_results[r][untrained_key]["iia"])
        ]
        recon_iias = [
            all_region_results[r][recon_key]["iia"]
            for r in all_region_results
            if recon_key in all_region_results[r]
            and not np.isnan(all_region_results[r][recon_key]["iia"])
        ]

        preconditions[model_name] = {
            "untrained_mean_iia": float(np.mean(untrained_iias)) if untrained_iias else None,
            "untrained_pass": (float(np.mean(untrained_iias)) < 0.6) if untrained_iias else None,
            "recon_only_mean_iia": float(np.mean(recon_iias)) if recon_iias else None,
            "recon_only_pass": (float(np.mean(recon_iias)) < 0.65) if recon_iias else None,
        }

    # --- Primary analysis: vacuity tests ---
    primary_tests = {}
    for model_name in active_specs:
        real_key = f"{model_name}__real__real__trained"
        noise_key = f"{model_name}__noise__real__trained"
        shuffled_key = f"{model_name}__real__shuffled__trained"

        regions = sorted(all_region_results.keys())
        real_iias = []
        noise_iias = []
        shuffled_iias = []

        for r in regions:
            rr = all_region_results[r]
            if real_key in rr and noise_key in rr:
                ri = rr[real_key]["iia"]
                ni = rr[noise_key]["iia"]
                if not np.isnan(ri) and not np.isnan(ni):
                    real_iias.append(ri)
                    noise_iias.append(ni)
            if real_key in rr and shuffled_key in rr:
                ri2 = rr[real_key]["iia"]
                si = rr[shuffled_key]["iia"]
                if not np.isnan(ri2) and not np.isnan(si):
                    shuffled_iias.append(si)

        test_result = {"n_regions": len(real_iias)}

        if len(real_iias) >= 5:
            gaps = np.array(real_iias) - np.array(noise_iias)
            mean_gap = float(np.mean(gaps))
            std_gap = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 1e-10
            cohens_d = mean_gap / std_gap if std_gap > 0 else 0.0

            try:
                w_stat, w_p = wilcoxon(gaps, alternative="greater")
            except Exception:
                w_stat, w_p = None, None

            untrained_ref = preconditions.get(model_name, {}).get("untrained_mean_iia", 0.5)
            noise_near_untrained = (
                abs(float(np.mean(noise_iias)) - (untrained_ref or 0.5)) < 0.05
            )

            non_vacuous = (
                (w_p is not None and w_p < 0.001)
                and (mean_gap >= 0.10 or abs(cohens_d) >= 0.5)
                and noise_near_untrained
            )

            test_result["vacuity"] = {
                "real_mean": float(np.mean(real_iias)),
                "noise_mean": float(np.mean(noise_iias)),
                "mean_gap": mean_gap,
                "cohens_d": cohens_d,
                "wilcoxon_p": float(w_p) if w_p is not None else None,
                "noise_near_untrained": noise_near_untrained,
                "non_vacuous": non_vacuous,
            }

            bootstrap = _hierarchical_bootstrap(
                all_region_results, region_mouse_map, real_key, noise_key,
            )
            test_result["vacuity"]["bootstrap_ci"] = bootstrap

            # TOST: is noise IIA equivalent to untrained baseline?
            # Two one-sided tests with margin delta=0.05
            tost_delta = 0.05
            noise_arr = np.array(noise_iias)
            tost_ref = untrained_ref or 0.5
            noise_shifted_hi = noise_arr - (tost_ref + tost_delta)
            noise_shifted_lo = (tost_ref - tost_delta) - noise_arr
            try:
                _, p_upper = wilcoxon(noise_shifted_hi, alternative="less")
                _, p_lower = wilcoxon(noise_shifted_lo, alternative="less")
                tost_p = max(float(p_upper), float(p_lower))
                tost_equivalent = tost_p < 0.05
            except Exception:
                tost_p, tost_equivalent = None, None
            test_result["vacuity"]["tost"] = {
                "delta": tost_delta,
                "reference": tost_ref,
                "p_value": tost_p,
                "equivalent_to_chance": tost_equivalent,
            }

        if shuffled_iias and len(shuffled_iias) >= 5:
            real_for_shuffled = real_iias[:len(shuffled_iias)]
            gaps_s = np.array(real_for_shuffled) - np.array(shuffled_iias)
            mean_gap_s = float(np.mean(gaps_s))
            std_gap_s = float(np.std(gaps_s, ddof=1)) if len(gaps_s) > 1 else 1e-10
            cohens_d_s = mean_gap_s / std_gap_s if std_gap_s > 0 else 0.0

            try:
                w_stat_s, w_p_s = wilcoxon(gaps_s, alternative="greater")
            except Exception:
                w_stat_s, w_p_s = None, None

            learns_structure = (
                (w_p_s is not None and w_p_s < 0.001)
                and (mean_gap_s >= 0.10 or abs(cohens_d_s) >= 0.5)
            )

            test_result["shuffled_label"] = {
                "real_mean": float(np.mean(real_for_shuffled)),
                "shuffled_mean": float(np.mean(shuffled_iias)),
                "mean_gap": mean_gap_s,
                "cohens_d": cohens_d_s,
                "wilcoxon_p": float(w_p_s) if w_p_s is not None else None,
                "learns_structure": learns_structure,
            }

        primary_tests[model_name] = test_result

    # --- Save summary ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(all_region_results),
        "n_replicates": N_REPLICATES,
        "n_fits_total": n_fits_total,
        "mean_seconds_per_condition_batch": (
            float(np.mean(fit_times)) if fit_times else None
        ),
        "preconditions": preconditions,
        "primary_tests": primary_tests,
        "per_region": all_region_results,
        "config": {
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "expansion_factor": EXPANSION_FACTOR,
            "l1_coeff": L1_COEFF,
            "n_epochs": N_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "alpha_choice": ALPHA_CHOICE,
            "beta_kl": BETA_KL,
            "n_iia_pairs": N_IIA_PAIRS,
            "n_replicates": N_REPLICATES,
            "l1_sweep": l1_sweep,
            "l1_values": l1_values,
        },
        "corrections": [
            "Model's own classifier for IIA (PiVAE: posterior-prior scoring)",
            "Pairs by evidence_labels (not choice_labels)",
            "Z-score fit on train split only, applied to all",
            "Shared train/test splits and intervention pairs across models",
            "Random pair sampling (not sequential)",
            "Filters no-go trials (response != 0)",
            "Bidirectional flips (both A->B and B->A)",
            f"{N_REPLICATES} matched replicates with fixed seeds",
            "Scale-relative sparsity threshold + Hoyer + top-k concentration",
            "Audit trail saved per region (splits, pairs, seeds)",
        ],
    }

    out_path = RESULTS_DIR / f"exp128_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Saved summary to {out_path}")

    print("\n" + "=" * 70)
    print("EXP128 HEADLINE RESULTS")
    print("=" * 70)
    print(f"\n{n_fits_total} model fits across {len(all_region_results)} regions "
          f"x {N_REPLICATES} replicates")
    if fit_times:
        print(f"Mean {np.mean(fit_times):.1f}s per condition batch, "
              f"total ~{sum(fit_times)/60:.0f} min")

    print("\nPreconditions:")
    for name, pc in preconditions.items():
        u = pc.get("untrained_mean_iia")
        r = pc.get("recon_only_mean_iia")
        print(f"  {name}: untrained={u:.3f if u else 'N/A'} "
              f"({'PASS' if pc.get('untrained_pass') else 'FAIL'}), "
              f"recon_only={r:.3f if r else 'N/A'} "
              f"({'PASS' if pc.get('recon_only_pass') else 'FAIL'})")

    print("\nVacuity tests:")
    for name, test in primary_tests.items():
        v = test.get("vacuity", {})
        if v:
            verdict = "NON-VACUOUS" if v.get("non_vacuous") else "VACUOUS"
            print(f"  {name}: {verdict} "
                  f"(real={v['real_mean']:.3f}, noise={v['noise_mean']:.3f}, "
                  f"gap={v['mean_gap']:.3f}, d={v['cohens_d']:.2f}, "
                  f"p={v['wilcoxon_p']:.1e})")
            bs = v.get("bootstrap_ci", {})
            if bs.get("ci_lower") is not None:
                print(f"    bootstrap 95% CI: [{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}]")
            tost = v.get("tost", {})
            if tost.get("p_value") is not None:
                eq = "YES" if tost["equivalent_to_chance"] else "NO"
                print(f"    TOST noise~chance: {eq} (p={tost['p_value']:.3f}, "
                      f"delta={tost['delta']}, ref={tost['reference']:.3f})")

    print("\nShuffled-label tests:")
    for name, test in primary_tests.items():
        s = test.get("shuffled_label", {})
        if s:
            verdict = "LEARNS STRUCTURE" if s.get("learns_structure") else "NO STRUCTURE"
            print(f"  {name}: {verdict} "
                  f"(real={s['real_mean']:.3f}, shuffled={s['shuffled_mean']:.3f}, "
                  f"gap={s['mean_gap']:.3f}, p={s['wilcoxon_p']:.1e})")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--l1-sweep", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run(max_sessions=args.max_sessions, model_filter=args.model,
        l1_sweep=args.l1_sweep)
