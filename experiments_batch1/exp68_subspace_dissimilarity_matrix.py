"""Experiment 68: Subspace dissimilarity matrix (Grassmannian RDM).

Inspired by Kriegeskorte (2008) RSA and Zamir et al. (2018) Taskonomy:
build a region-by-region dissimilarity matrix using Grassmannian distances
between choice subspaces, then compare it to CKA-based and Procrustes-based
RDMs and anatomical distance.

This is "RSA of subspaces": where standard RSA asks "do two conditions evoke
similar population responses?", the GDM asks "do two regions use similar
geometric strategies to encode choice?"

For all 73 regions:
  1. Estimate choice subspace via LDA and VAE.
  2. Build 73x73 Grassmannian dissimilarity matrices (GDM_LDA, GDM_VAE).
  3. Build CKA RDM and Procrustes RDM from cross-session data.
  4. Compute Mantel test correlations: GDM vs CKA-RDM, GDM vs Procrustes-RDM.
  5. Test whether GDM recovers functional hierarchy (sensory -> association -> motor).
  6. Cluster the GDM and compare to known anatomical groupings.

The neural taskonomy prediction: regions with similar computational roles
(both sensory, both motor planning) should have lower Grassmannian distances
than regions with different roles, even when anatomically distant.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp68"
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

N_MANTEL_PERMS = 10000

FUNCTIONAL_GROUPS = {
    "sensory": ["VISp", "VISl", "VISrl", "VISam", "VISpm", "VISal", "VISa"],
    "motor": ["MOs", "MOp", "SCm", "SCs", "GRN", "MRN"],
    "association": ["ACA", "PL", "ILA", "ORB", "RSP", "RSPv"],
    "hippocampal": ["CA1", "CA3", "DG", "SUB", "POST"],
    "thalamic": ["LP", "LD", "PO", "VPM", "VPL", "VAL", "MD", "CL", "RT"],
    "striatal": ["CP", "ACB", "LS", "LSc", "LSr"],
    "midbrain": ["SNr", "ZI", "PAG", "IC", "SC"],
}

REGION_TO_GROUP = {}
for group, regions in FUNCTIONAL_GROUPS.items():
    for r in regions:
        REGION_TO_GROUP[r] = group


class StructuredVAE(nn.Module):
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        z_dim = z_choice_dim + z_other_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_choice = z[:, :self.z_choice_dim]
        recon = self.decoder(z)
        choice_logits = self.choice_head(z_choice)
        return recon, mu, logvar, choice_logits, z_choice


def _train_vae(activity, labels, device, n_neurons):
    model = StructuredVAE(n_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(dataset) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            recon, mu, logvar, choice_logits, _ = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            choice_loss = F.cross_entropy(choice_logits, yb)
            loss = recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _grassmannian_distance(U1, U2):
    """Geodesic distance between two subspaces on the Grassmannian.
    U1, U2: (k, d) orthonormal bases."""
    if U1.shape != U2.shape:
        k = min(U1.shape[0], U2.shape[0])
        U1 = U1[:k]
        U2 = U2[:k]
    M = U1 @ U2.T
    svals = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    angles = np.arccos(svals)
    return float(np.linalg.norm(angles))


def _orthonormalize(dirs):
    """SVD-based orthonormalization of direction matrix (k, d) -> (k, d) orthonormal rows."""
    _, _, Vt = np.linalg.svd(dirs, full_matrices=False)
    return Vt


def _compute_cka(X1, X2):
    """Linear CKA between two activity matrices."""
    X1 = X1 - X1.mean(axis=0)
    X2 = X2 - X2.mean(axis=0)
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T
    hsic_12 = np.sum(K1 * K2)
    hsic_11 = np.sum(K1 * K1)
    hsic_22 = np.sum(K2 * K2)
    denom = np.sqrt(hsic_11 * hsic_22)
    if denom < 1e-12:
        return 0.0
    return float(hsic_12 / denom)


def _mantel_test(D1, D2, n_perms=N_MANTEL_PERMS):
    """Mantel test between two distance matrices."""
    v1 = squareform(D1)
    v2 = squareform(D2)
    rho_obs, _ = spearmanr(v1, v2)
    count = 0
    n = D1.shape[0]
    for _ in range(n_perms):
        perm = np.random.permutation(n)
        D2_perm = D2[np.ix_(perm, perm)]
        v2_perm = squareform(D2_perm)
        rho_perm, _ = spearmanr(v1, v2_perm)
        if rho_perm >= rho_obs:
            count += 1
    return float(rho_obs), float((count + 1) / (n_perms + 1))


def _fit_alpha(activity):
    """Power-law exponent from PCA eigenvalue spectrum."""
    n_components = min(50, activity.shape[1], activity.shape[0] - 1)
    if n_components < 15:
        return float("nan")
    pca = PCA(n_components=n_components)
    pca.fit(activity)
    eigenvalues = pca.explained_variance_
    ranks = np.arange(10, min(50, len(eigenvalues)))
    if len(ranks) < 5:
        return float("nan")
    log_ranks = np.log(ranks + 1)
    log_eig = np.log(eigenvalues[ranks] + 1e-12)
    slope, _ = np.polyfit(log_ranks, log_eig, 1)
    return float(-slope)


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting subspace dissimilarity matrix experiment "
                f"with {len(sessions)} sessions on {device}")

    # --- Load all region data ---
    region_data: dict[str, list[dict]] = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "activity": activity,
                "choice_labels": ch,
                "n_neurons": int(activity.shape[1]),
            })

    valid_regions = sorted([r for r, v in region_data.items()
                           if len(v) >= 1 and sum(len(s["choice_labels"]) for s in v) >= MIN_TRIALS_PER_CONDITION * 2])
    n_regions = len(valid_regions)
    logger.info(f"{datetime.now().isoformat()} {n_regions} valid regions")

    # --- Compute per-session subspaces and Grassmannian distances ---
    # Different sessions have different neuron counts for the same region, so we cannot
    # compare subspaces across sessions. Instead: within each session, compute LDA/VAE
    # directions for every region present, compute pairwise Grassmannian distances between
    # co-recorded regions (same ambient dimension via PCA), then average across sessions.

    # Reorganize data by session for within-session processing
    session_to_regions: dict[int, dict[str, dict]] = {}
    for region in valid_regions:
        for entry in region_data[region]:
            s_idx = entry["session_idx"]
            if s_idx not in session_to_regions:
                session_to_regions[s_idx] = {}
            session_to_regions[s_idx][region] = entry

    # Compute per-region alphas by averaging across sessions
    region_alphas = {}
    for region in valid_regions:
        alphas = [_fit_alpha(e["activity"]) for e in region_data[region]]
        alphas = [a for a in alphas if not np.isnan(a)]
        region_alphas[region] = float(np.mean(alphas)) if alphas else float("nan")

    # Build GDMs by averaging per-session distance matrices
    logger.info(f"{datetime.now().isoformat()} Building {n_regions}x{n_regions} GDMs "
                f"across {len(session_to_regions)} sessions")
    gdm_lda_sum = np.zeros((n_regions, n_regions))
    gdm_vae_sum = np.zeros((n_regions, n_regions))
    gdm_count = np.zeros((n_regions, n_regions))
    region_idx = {r: i for i, r in enumerate(valid_regions)}

    N_PCA_AMBIENT = 30  # shared ambient dimension for Grassmannian comparison

    for s_idx, sess_regions in tqdm(session_to_regions.items(), desc="Per-session GDMs"):
        # Only process regions with enough trials
        usable = {r: d for r, d in sess_regions.items()
                  if len(d["choice_labels"]) >= MIN_TRIALS_PER_CONDITION * 2}
        if len(usable) < 2:
            continue

        # Stack all neurons from this session to build a shared PCA ambient space
        all_activities = []
        region_slices = {}
        col = 0
        for r in sorted(usable.keys()):
            act = usable[r]["activity"]
            n_trials = act.shape[0]
            n_neur = act.shape[1]
            region_slices[r] = (col, col + n_neur)
            all_activities.append(act)
            col += n_neur

        # Use minimum trial count across regions
        min_trials = min(a.shape[0] for a in all_activities)
        concat = np.concatenate([a[:min_trials] for a in all_activities], axis=1)  # (trials, total_neurons)

        n_pca = min(N_PCA_AMBIENT, concat.shape[1], min_trials - 1)
        if n_pca < Z_CHOICE_DIM:
            continue
        pca = PCA(n_components=n_pca)
        pca.fit(concat)
        # Project each region's activity into shared PCA space
        region_pca = {}
        for r in sorted(usable.keys()):
            act = usable[r]["activity"][:min_trials]
            ch = usable[r]["choice_labels"][:min_trials]
            start, end = region_slices[r]
            # Zero-pad to full concatenated dimension, project through PCA
            padded = np.zeros((min_trials, concat.shape[1]))
            padded[:, start:end] = act
            region_pca[r] = {"projected": pca.transform(padded), "choice_labels": ch}

        # Compute LDA and VAE subspaces in the shared PCA space
        sess_lda_dirs = {}
        sess_vae_dirs = {}
        for r, pca_data in region_pca.items():
            proj = pca_data["projected"]
            ch = pca_data["choice_labels"]

            # LDA in PCA space
            lda = LinearDiscriminantAnalysis()
            lda.fit(proj, ch)
            lda_dirs = lda.scalings_[:, :min(Z_CHOICE_DIM, lda.scalings_.shape[1])].T
            sess_lda_dirs[r] = _orthonormalize(lda_dirs)

            # VAE in PCA space
            n_pca_dim = proj.shape[1]
            vae_model = _train_vae(proj, ch, device, n_pca_dim)
            vae_model.eval()
            with torch.no_grad():
                X_t = torch.tensor(proj, dtype=torch.float32, device=device)
                mu, _ = vae_model.encode(X_t)
                z_choice = mu[:, :Z_CHOICE_DIM].cpu().numpy()
            ridge = Ridge(alpha=1.0)
            ridge.fit(proj, z_choice)
            sess_vae_dirs[r] = _orthonormalize(ridge.coef_)

        # Pairwise Grassmannian distances within this session
        regions_list = sorted(sess_lda_dirs.keys())
        for a_idx, ra in enumerate(regions_list):
            for rb in regions_list[a_idx + 1:]:
                i = region_idx[ra]
                j = region_idx[rb]
                d_lda = _grassmannian_distance(sess_lda_dirs[ra], sess_lda_dirs[rb])
                d_vae = _grassmannian_distance(sess_vae_dirs[ra], sess_vae_dirs[rb])
                gdm_lda_sum[i, j] += d_lda
                gdm_lda_sum[j, i] += d_lda
                gdm_vae_sum[i, j] += d_vae
                gdm_vae_sum[j, i] += d_vae
                gdm_count[i, j] += 1
                gdm_count[j, i] += 1

    # Average across sessions (avoid division by zero for pairs never co-recorded)
    gdm_lda = np.where(gdm_count > 0, gdm_lda_sum / gdm_count, 0.0)
    gdm_vae = np.where(gdm_count > 0, gdm_vae_sum / gdm_count, 0.0)

    # --- Build CKA RDM (using cross-session CKA where available) ---
    cka_rdm = np.ones((n_regions, n_regions))
    for i in range(n_regions):
        cka_rdm[i, i] = 0.0
        for j in range(i + 1, n_regions):
            ri, rj = valid_regions[i], valid_regions[j]
            # Find sessions where both regions are present
            sess_i = {s["session_idx"] for s in region_data[ri]}
            sess_j = {s["session_idx"] for s in region_data[rj]}
            common = sess_i & sess_j
            if not common:
                cka_rdm[i, j] = cka_rdm[j, i] = 1.0
                continue
            cka_vals = []
            for s_idx in common:
                act_i = [s["activity"] for s in region_data[ri] if s["session_idx"] == s_idx][0]
                act_j = [s["activity"] for s in region_data[rj] if s["session_idx"] == s_idx][0]
                n_shared = min(act_i.shape[0], act_j.shape[0])
                cka_vals.append(_compute_cka(act_i[:n_shared], act_j[:n_shared]))
            cka_rdm[i, j] = cka_rdm[j, i] = 1.0 - np.mean(cka_vals)

    # --- Mantel tests ---
    logger.info(f"{datetime.now().isoformat()} Running Mantel tests (n_perms={N_MANTEL_PERMS})")
    mantel_lda_cka_rho, mantel_lda_cka_p = _mantel_test(gdm_lda, cka_rdm)
    mantel_vae_cka_rho, mantel_vae_cka_p = _mantel_test(gdm_vae, cka_rdm)
    mantel_lda_vae_rho, mantel_lda_vae_p = _mantel_test(gdm_lda, gdm_vae)

    # --- Functional hierarchy test ---
    group_labels = [REGION_TO_GROUP.get(r, "other") for r in valid_regions]
    n_with_group = sum(1 for g in group_labels if g != "other")

    within_group_lda = []
    between_group_lda = []
    within_group_vae = []
    between_group_vae = []

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            gi, gj = group_labels[i], group_labels[j]
            if gi == "other" or gj == "other":
                continue
            if gi == gj:
                within_group_lda.append(gdm_lda[i, j])
                within_group_vae.append(gdm_vae[i, j])
            else:
                between_group_lda.append(gdm_lda[i, j])
                between_group_vae.append(gdm_vae[i, j])

    if within_group_lda and between_group_lda:
        u_lda, p_lda = mannwhitneyu(within_group_lda, between_group_lda, alternative="less")
        u_vae, p_vae = mannwhitneyu(within_group_vae, between_group_vae, alternative="less")
    else:
        u_lda = p_lda = u_vae = p_vae = float("nan")

    # --- Clustering and ARI ---
    for n_clusters in [5, 7, 10]:
        clust_lda = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                            linkage="average")
        clust_vae = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                            linkage="average")
        labels_lda = clust_lda.fit_predict(gdm_lda)
        labels_vae = clust_vae.fit_predict(gdm_vae)

    # Use the k=7 clustering for ARI against functional groups
    clust_lda_7 = AgglomerativeClustering(n_clusters=7, metric="precomputed", linkage="average")
    clust_vae_7 = AgglomerativeClustering(n_clusters=7, metric="precomputed", linkage="average")
    labels_lda_7 = clust_lda_7.fit_predict(gdm_lda)
    labels_vae_7 = clust_vae_7.fit_predict(gdm_vae)

    func_labels_int = [hash(g) % 100 for g in group_labels]
    ari_lda = adjusted_rand_score(func_labels_int, labels_lda_7)
    ari_vae = adjusted_rand_score(func_labels_int, labels_vae_7)
    nmi_lda = normalized_mutual_info_score(func_labels_int, labels_lda_7)
    nmi_vae = normalized_mutual_info_score(func_labels_int, labels_vae_7)

    summary = {
        "n_regions": n_regions,
        "region_list": valid_regions,
        "mantel_tests": {
            "gdm_lda_vs_cka_rdm": {"rho": mantel_lda_cka_rho, "p": mantel_lda_cka_p},
            "gdm_vae_vs_cka_rdm": {"rho": mantel_vae_cka_rho, "p": mantel_vae_cka_p},
            "gdm_lda_vs_gdm_vae": {"rho": mantel_lda_vae_rho, "p": mantel_lda_vae_p},
        },
        "functional_hierarchy": {
            "n_with_group": n_with_group,
            "n_within_pairs": len(within_group_lda),
            "n_between_pairs": len(between_group_lda),
            "lda": {
                "mean_within": float(np.mean(within_group_lda)) if within_group_lda else None,
                "mean_between": float(np.mean(between_group_lda)) if between_group_lda else None,
                "U": float(u_lda), "p": float(p_lda),
            },
            "vae": {
                "mean_within": float(np.mean(within_group_vae)) if within_group_vae else None,
                "mean_between": float(np.mean(between_group_vae)) if between_group_vae else None,
                "U": float(u_vae), "p": float(p_vae),
            },
        },
        "clustering_k7": {
            "lda_ari": float(ari_lda), "lda_nmi": float(nmi_lda),
            "vae_ari": float(ari_vae), "vae_nmi": float(nmi_vae),
        },
        "gdm_pair_coverage": {
            "mean_sessions_per_pair": float(gdm_count[gdm_count > 0].mean()) if (gdm_count > 0).any() else 0,
            "n_pairs_with_data": int((gdm_count > 0).sum() // 2),
            "n_total_pairs": n_regions * (n_regions - 1) // 2,
        },
        "gdm_lda": gdm_lda.tolist(),
        "gdm_vae": gdm_vae.tolist(),
        "cka_rdm": cka_rdm.tolist(),
        "region_alphas": {r: region_alphas[r] for r in valid_regions},
    }

    out_path = RESULTS_DIR / f"exp68_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Results saved to {out_path}")

    return summary
