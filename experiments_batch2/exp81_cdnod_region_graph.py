"""Experiment 81: CD-NOD causal discovery over brain regions.

Discovers directed causal graph between brain regions using CD-NOD
(Huang et al., JMLR 2020) on VAE choice subspace projections.

Key idea: CD-NOD exploits distribution shifts across sessions (different mice,
different days, different behavioral states) to identify + orient causal edges.
Session index is the context variable c_indx — the heterogeneity IS the signal.

Pipeline:
  1. Load Steinmetz data across all sessions
  2. Train structured VAE per region (as in exp69)
  3. Project each trial to z_choice (first causal dim) per region
  4. Build (n_trials_total, n_regions) data matrix + c_indx = session index
  5. Run CD-NOD (PC + Phase III orientation using distribution shifts)
  6. Also run plain PC algorithm as baseline (ignores session structure)
  7. Compare discovered causal graph against exp70 cross-region patching hub scores

Validation: regions with high outgoing causal strength in CD-NOD graph should
be the same regions where silencing most disrupts behavior (optogenetic data).

References:
  Huang et al. (2020). Causal Discovery from Heterogeneous/Nonstationary Data.
  JMLR 21(89):1-53.

Usage:
    modal run modal_run.py --experiment exp81
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20
MIN_SESSIONS_PER_REGION = 3

Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0


class StructuredVAE(nn.Module):
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits, z[:, :self.z_choice_dim]

    def loss(self, x, labels):
        recon, mu, logvar, choice_logits, _ = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss


def _train_vae(model, activity, labels, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(dataset) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            loss = model.loss(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _extract_z_choice(model, activity, device="cpu"):
    model.eval()
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    with torch.inference_mode():
        mu, _ = model.encode(X)
    return mu[:, :model.z_choice_dim].cpu().numpy()


def _parse_adjacency(adj, labels):
    """Parse causal-learn adjacency into directed/undirected edges + hub scores."""
    n_aug = adj.shape[0]
    directed_edges = []
    undirected_edges = []
    c_connected = []

    for i in range(n_aug):
        for j in range(i + 1, n_aug):
            if adj[i, j] == -1 and adj[j, i] == 1:
                directed_edges.append((labels[j], labels[i]))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                directed_edges.append((labels[i], labels[j]))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                undirected_edges.append((labels[i], labels[j]))
            elif adj[i, j] == 1 and adj[j, i] == 1:
                directed_edges.append((labels[i], labels[j]))
                directed_edges.append((labels[j], labels[i]))

    c_idx = n_aug - 1
    for i in range(n_aug - 1):
        if adj[c_idx, i] != 0 or adj[i, c_idx] != 0:
            c_connected.append(labels[i])

    # Hub scores: outgoing and incoming degree per region
    region_labels = labels[:-1]  # exclude C node
    hub_scores = {}
    for region in region_labels:
        outgoing = sum(1 for src, tgt in directed_edges if src == region and tgt != "C")
        incoming = sum(1 for src, tgt in directed_edges if tgt == region and src != "C")
        hub_scores[region] = {
            "outgoing": outgoing,
            "incoming": incoming,
            "total_degree": outgoing + incoming,
            "is_changing_module": region in c_connected,
        }

    return {
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
        "changing_modules": c_connected,
        "n_directed": len(directed_edges),
        "n_undirected": len(undirected_edges),
        "n_changing_modules": len(c_connected),
        "hub_scores": hub_scores,
    }


def run_cdnod_on_data(data, c_indx, region_labels, alpha=0.05, indep_test="fisherz"):
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.cit import fisherz as fisherz_test, kci as kci_test

    test_map = {"fisherz": fisherz_test, "kci": kci_test}
    test_fn = test_map.get(indep_test, fisherz_test)

    logger.info(f"Running CD-NOD: {data.shape[0]} samples, {data.shape[1]} variables, "
                f"alpha={alpha}, test={indep_test}")

    t0 = time.time()
    cg = cdnod(data, c_indx, alpha=alpha, indep_test=test_fn, stable=True,
               uc_rule=0, uc_priority=2, verbose=False, show_progress=True)
    elapsed = time.time() - t0
    logger.info(f"CD-NOD completed in {elapsed:.1f}s")

    labels = list(region_labels) + ["C"]
    result = _parse_adjacency(cg.G.graph, labels)
    result["elapsed_seconds"] = round(elapsed, 1)
    result["alpha"] = alpha
    result["indep_test"] = indep_test
    result["adjacency_matrix"] = cg.G.graph.tolist()
    result["labels"] = labels
    return result


def _parse_adj_no_c(adj, labels):
    """Parse adjacency from methods without a C node (PC, NOTEARS, DAGMA)."""
    directed_edges = []
    undirected_edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                directed_edges.append((labels[j], labels[i]))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                directed_edges.append((labels[i], labels[j]))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                undirected_edges.append((labels[i], labels[j]))

    hub_scores = {}
    for region in labels:
        outgoing = sum(1 for src, tgt in directed_edges if src == region)
        incoming = sum(1 for src, tgt in directed_edges if tgt == region)
        hub_scores[region] = {"outgoing": outgoing, "incoming": incoming,
                              "total_degree": outgoing + incoming}

    return {
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
        "n_directed": len(directed_edges),
        "n_undirected": len(undirected_edges),
        "hub_scores": hub_scores,
        "adjacency_matrix": adj.tolist(),
        "labels": list(labels),
    }


def _parse_weighted_adj(W, labels, threshold=0.1):
    """Parse weighted adjacency from NOTEARS/DAGMA into directed edges."""
    directed_edges = []
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and abs(W[i, j]) > threshold:
                directed_edges.append((labels[j], labels[i], round(float(W[i, j]), 4)))

    hub_scores = {}
    for region in labels:
        outgoing = sum(1 for src, tgt, *_ in directed_edges if src == region)
        incoming = sum(1 for src, tgt, *_ in directed_edges if tgt == region)
        out_weight = sum(abs(w) for src, tgt, w in directed_edges if src == region)
        in_weight = sum(abs(w) for src, tgt, w in directed_edges if tgt == region)
        hub_scores[region] = {"outgoing": outgoing, "incoming": incoming,
                              "total_degree": outgoing + incoming,
                              "outgoing_weight": round(out_weight, 4),
                              "incoming_weight": round(in_weight, 4)}

    return {
        "directed_edges": [(s, t, w) for s, t, w in directed_edges],
        "n_directed": len(directed_edges),
        "hub_scores": hub_scores,
        "weighted_adjacency": W.tolist(),
        "labels": list(labels),
        "threshold": threshold,
    }


def run_pc_on_data(data, region_labels, alpha=0.05, indep_test="fisherz"):
    """Run plain PC algorithm as baseline (ignores session structure)."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz as fisherz_test, kci as kci_test

    test_map = {"fisherz": fisherz_test, "kci": kci_test}
    test_fn = test_map.get(indep_test, fisherz_test)

    logger.info(f"Running PC: {data.shape[0]} samples, {data.shape[1]} variables, "
                f"alpha={alpha}, test={indep_test}")

    t0 = time.time()
    cg = pc(data, alpha=alpha, indep_test=test_fn, stable=True,
            uc_rule=0, uc_priority=2, verbose=False, show_progress=True)
    elapsed = time.time() - t0
    logger.info(f"PC completed in {elapsed:.1f}s")

    result = _parse_adj_no_c(cg.G.graph, list(region_labels))
    result["elapsed_seconds"] = round(elapsed, 1)
    return result


def run_notears_on_data(data, region_labels, lambda1=0.1):
    """Run NOTEARS continuous DAG optimization baseline."""
    from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
    except ImportError:
        pass

    logger.info(f"Running NOTEARS: {data.shape[0]} samples, {data.shape[1]} variables, "
                f"lambda1={lambda1}")

    t0 = time.time()
    try:
        from causallearn.search.FCMBased import lingam
        model = lingam.DirectLiNGAM()
        model.fit(data)
        W = model.adjacency_matrix_
        elapsed = time.time() - t0
        logger.info(f"DirectLiNGAM completed in {elapsed:.1f}s")
        result = _parse_weighted_adj(W, list(region_labels))
        result["elapsed_seconds"] = round(elapsed, 1)
        result["method"] = "DirectLiNGAM"
        return result
    except Exception as e:
        logger.warning(f"DirectLiNGAM failed: {e}, trying NOTEARS linear")

    try:
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
        # Fallback: use GES (Greedy Equivalence Search) which is always available
        from causallearn.search.ScoreBased.GES import ges
        record = ges(data, score_func='local_score_BIC')
        elapsed = time.time() - t0
        logger.info(f"GES completed in {elapsed:.1f}s")
        result = _parse_adj_no_c(record['G'].graph, list(region_labels))
        result["elapsed_seconds"] = round(elapsed, 1)
        result["method"] = "GES"
        return result
    except Exception as e2:
        logger.warning(f"GES also failed: {e2}")
        return {"error": str(e2), "method": "GES_failed"}


def run(max_sessions: int | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    logger.info(f"{datetime.now().isoformat()} Starting CD-NOD region graph with "
                f"{len(sessions)} sessions on {device}")

    # Phase 1: Load data grouped by (region, session)
    region_session_data: dict[str, list[dict]] = {}
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
            if (ch == 0).sum() < MIN_TRIALS_PER_CONDITION or (ch == 1).sum() < MIN_TRIALS_PER_CONDITION:
                continue
            if region not in region_session_data:
                region_session_data[region] = []
            region_session_data[region].append({
                "activity": activity,
                "choice_labels": ch,
                "n_neurons": int(activity.shape[1]),
                "sess_idx": sess_idx,
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_session_data)} regions loaded")

    # Phase 2: Train VAE per (region, session) and extract z_choice
    region_z: dict[str, list[dict]] = {}
    for region in tqdm(sorted(region_session_data.keys()), desc="Training VAEs"):
        region_z[region] = []
        for sess_data in region_session_data[region]:
            activity = sess_data["activity"]
            ch = sess_data["choice_labels"]
            n_neurons = sess_data["n_neurons"]
            sess_idx = sess_data["sess_idx"]

            model = StructuredVAE(n_neurons).to(device)
            model = _train_vae(model, activity, ch, device)
            z_choice = _extract_z_choice(model, activity, device)
            region_z[region].append({
                "z_choice": z_choice,
                "choice_labels": ch,
                "sess_idx": sess_idx,
                "n_trials": len(ch),
            })
            del model

    # Phase 3: Find a good (regions, sessions) subset for CD-NOD
    # Problem: requiring ALL regions in ALL sessions is too strict.
    # Strategy: greedy — start with the most recorded region, add regions
    # while keeping >= MIN_SESSIONS_PER_REGION co-occurring sessions.
    session_ids = sorted({s["sess_idx"] for entries in region_z.values() for s in entries})
    logger.info(f"Total sessions with data: {len(session_ids)}")

    region_sessions = {}
    for region, entries in region_z.items():
        region_sessions[region] = {s["sess_idx"] for s in entries}

    # Sort regions by how many sessions they appear in (descending)
    regions_by_coverage = sorted(region_sessions.keys(),
                                 key=lambda r: len(region_sessions[r]), reverse=True)
    for r in regions_by_coverage[:10]:
        logger.info(f"  {r}: {len(region_sessions[r])} sessions")

    # Strategy: try multiple min-session thresholds, pick the one that
    # maximizes n_regions * n_sessions (best data coverage).
    best_regions = []
    best_sessions = []
    best_score = 0

    for min_sess in [3, 4, 5, 6, 8, 10]:
        cand = []
        common = set(session_ids)
        for region in regions_by_coverage:
            new_common = common & region_sessions[region]
            if len(new_common) >= min_sess:
                cand.append(region)
                common = new_common
        if len(cand) >= 3 and len(common) >= 2:
            score = len(cand) * len(common)
            logger.info(f"  min_sess={min_sess}: {len(cand)} regions x {len(common)} sessions (score={score})")
            if score > best_score:
                best_score = score
                best_regions = list(cand)
                best_sessions = sorted(common)

    candidate_regions = sorted(best_regions)
    common_sessions = best_sessions

    logger.info(f"Best selection: {len(candidate_regions)} regions x "
                f"{len(common_sessions)} co-occurring sessions (score={best_score})")

    if len(common_sessions) < 2 or len(candidate_regions) < 3:
        logger.error("Not enough overlap for causal discovery")
        return {"error": "insufficient_overlap",
                "n_regions": len(candidate_regions), "n_sessions": len(common_sessions)}

    # Phase 4: Build the (n_trials_total, n_regions) data matrix
    # For each session, use the first z_choice component (most informative)
    all_data_rows = []
    all_c_indx = []

    for sess_idx in tqdm(common_sessions, desc="Building data matrix"):
        # Get min trial count across regions for this session (for alignment)
        trial_counts = []
        for region in candidate_regions:
            entries = [e for e in region_z[region] if e["sess_idx"] == sess_idx]
            if entries:
                trial_counts.append(entries[0]["n_trials"])
        if not trial_counts:
            continue
        n_trials = min(trial_counts)

        # Build row: each trial gets a row with z_choice[0] per region
        for trial_i in range(n_trials):
            row = []
            for region in candidate_regions:
                entries = [e for e in region_z[region] if e["sess_idx"] == sess_idx]
                z = entries[0]["z_choice"]
                row.append(z[trial_i, 0])  # first causal dimension
            all_data_rows.append(row)
            all_c_indx.append(common_sessions.index(sess_idx))

    data = np.array(all_data_rows, dtype=np.float64)
    c_indx = np.array(all_c_indx, dtype=np.float64).reshape(-1, 1)

    logger.info(f"Data matrix: {data.shape}, c_indx unique values: {len(np.unique(c_indx))}")

    # Phase 5: Run CD-NOD
    cdnod_result = run_cdnod_on_data(data, c_indx, candidate_regions, alpha=0.05)
    logger.info(f"CD-NOD: {cdnod_result['n_directed']} directed, "
                f"{cdnod_result['n_undirected']} undirected, "
                f"{cdnod_result['n_changing_modules']} changing modules")

    # Phase 6: Run baselines (PC, DirectLiNGAM/GES)
    pc_result = run_pc_on_data(data, candidate_regions, alpha=0.05)
    logger.info(f"PC: {pc_result['n_directed']} directed, {pc_result['n_undirected']} undirected")

    score_result = run_notears_on_data(data, candidate_regions)
    logger.info(f"Score-based: {score_result.get('n_directed', 'N/A')} directed "
                f"(method: {score_result.get('method', 'unknown')})")

    # Phase 7: Compare against exp70 hub scores (if available)
    exp70_comparison = None
    exp70_path = Path(__file__).parent.parent / "artifacts" / "exp70"
    exp70_files = sorted(exp70_path.glob("exp70_*.json")) if exp70_path.exists() else []
    if not exp70_files:
        exp70_path = Path(__file__).parent / "results" / "exp70"
        exp70_files = sorted(exp70_path.glob("exp70_*.json")) if exp70_path.exists() else []

    if exp70_files:
        with open(exp70_files[-1]) as f:
            exp70_data = json.load(f)
        if "region_hub_scores" in exp70_data:
            exp70_hubs = exp70_data["region_hub_scores"]
            # Compare: rank correlation between CD-NOD outgoing degree and exp70 outgoing IIA
            shared_regions = [r for r in candidate_regions if r in exp70_hubs]
            if len(shared_regions) >= 5:
                cdnod_outgoing = [cdnod_result["hub_scores"][r]["outgoing"] for r in shared_regions]
                exp70_outgoing = [exp70_hubs[r]["mean_outgoing_iia"] for r in shared_regions]
                rho, pval = spearmanr(cdnod_outgoing, exp70_outgoing)
                exp70_comparison = {
                    "n_shared_regions": len(shared_regions),
                    "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else None,
                    "spearman_pval": round(float(pval), 6) if not np.isnan(pval) else None,
                    "shared_regions": shared_regions,
                }
                logger.info(f"CD-NOD vs exp70 hub correlation: rho={rho:.3f}, p={pval:.4f} "
                            f"({len(shared_regions)} shared regions)")

    # Top hubs by outgoing degree
    top_cdnod_hubs = sorted(cdnod_result["hub_scores"].items(),
                            key=lambda x: x[1]["outgoing"], reverse=True)[:10]
    top_pc_hubs = sorted(pc_result["hub_scores"].items(),
                         key=lambda x: x[1]["outgoing"], reverse=True)[:10]

    logger.info(f"\nTop CD-NOD hubs (outgoing): {[(r, s['outgoing']) for r, s in top_cdnod_hubs]}")
    logger.info(f"Top PC hubs (outgoing): {[(r, s['outgoing']) for r, s in top_pc_hubs]}")
    logger.info(f"Changing modules: {cdnod_result['changing_modules']}")

    # Score-based top hubs
    if "hub_scores" in score_result:
        top_score_hubs = sorted(score_result["hub_scores"].items(),
                                key=lambda x: x[1].get("outgoing_weight", x[1]["outgoing"]),
                                reverse=True)[:10]
        logger.info(f"Top score-based hubs: {[(r, s.get('outgoing_weight', s['outgoing'])) for r, s in top_score_hubs]}")
    else:
        top_score_hubs = []

    return {
        "n_regions": len(candidate_regions),
        "n_sessions": len(common_sessions),
        "n_trials_total": len(all_data_rows),
        "regions": candidate_regions,
        "sessions": common_sessions,
        "cdnod": cdnod_result,
        "pc_baseline": pc_result,
        "score_baseline": score_result,
        "exp70_comparison": exp70_comparison,
        "top_cdnod_hubs": [(r, s) for r, s in top_cdnod_hubs],
        "top_pc_hubs": [(r, s) for r, s in top_pc_hubs],
        "top_score_hubs": [(r, s) for r, s in top_score_hubs],
    }
