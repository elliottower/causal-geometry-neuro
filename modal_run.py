"""Modal runner for neuro-causal-geometry experiments.

Usage:
    modal run modal_run.py --experiment exp7 --detach
    modal run modal_run.py --experiment exp7,exp8,exp9 --small --detach
    modal run modal_run.py --experiment all --small --detach
    modal run modal_run.py --experiment exp1 --max-sessions 5 --detach
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import modal

SA_KEY_PATH = Path.home() / ".config" / "gcloud" / "factorization-circuits" / "sa.json"
sa_key_content = SA_KEY_PATH.read_text() if SA_KEY_PATH.exists() else "{}"

gcs_secret = modal.Secret.from_dict({"GCS_SA_JSON": sa_key_content})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "pkg-config")
    .pip_install("setuptools<71", "wheel", "Cython", "numpy>=1.24,<2")
    .pip_install(
        "scipy>=1.11",
        "torch>=2.1,<2.3",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "pandas>=2.1",
        "scikit-learn>=1.3",
        "tqdm>=4.66",
        "requests>=2.31",
        "google-cloud-storage>=2.14",
    )
    .pip_install("ripser>=0.6", "persim>=0.3", "umap-learn>=0.5")
    .pip_install("causal-learn>=0.1.3")
    .pip_install("slicetca>=0.1")
    .pip_install("ONE-api>=2.7", "ibllib>=2.36")
    .pip_install("h5py>=3.9")
    .pip_install("allensdk>=2.16")
    .add_local_dir("geometry", "/root/repo/geometry")
    .add_local_dir("data", "/root/repo/data")
    .add_local_dir("experiments", "/root/repo/experiments")
    .add_local_dir("batch2_reviewer_fixes", "/root/repo/batch2_reviewer_fixes")
    .add_local_file("pyproject.toml", "/root/repo/pyproject.toml")
)

app = modal.App("neuro-causal-geometry")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

EXPERIMENT_MAP = {
    "exp1": "experiments.exp1_cross_animal_stability",
    "exp2": "experiments.exp2_cka_vs_grassmannian",
    "exp3": "experiments.exp3_sheaf_cohomology_ibl",
    "exp4": "experiments.exp4_stimulus_transportability",
    "exp5": "experiments.exp5_behavioral_state_modulation",
    "exp6": "experiments.exp6_gauge_correction",
    "exp7": "experiments.exp7_sheaf_cohomology",
    "exp7_controls": "experiments.exp7_controls",
    "exp8": "experiments.exp8_holonomy",
    "exp9": "experiments.exp9_multiple_realization",
    "exp10": "experiments.exp10_direction_vs_subspace",
    "exp11": "experiments.exp11_linear_vs_nonlinear",
    "exp12": "experiments.exp12_topology_vs_geometry",
    "exp13": "experiments.exp13_static_vs_dynamic",
    "exp14": "experiments.exp14_geometric_type_classifier",
    "exp15": "experiments.exp15_communication_subspace_sheaf",
    "exp16": "experiments.exp16_sae_spike_trains",
    "exp17": "experiments.exp17_neural_factor_bank",
    "exp18": "experiments.exp18_grassmannian_parcellation",
    "exp19": "experiments.exp19_latent_causal_discovery",
    "exp20": "experiments.exp20_jpca_rotation",
    "exp15b": "experiments.exp15b_communication_subspace_sheaf_ibl",
    "exp21": "experiments.exp21_baseline_comparisons",
    "exp22": "experiments.exp22_geometric_type_prediction",
    "exp23": "experiments.exp23_subsample_control",
    "exp24": "experiments.exp24_robustness_controls",
    "exp25": "experiments.exp25_cortical_hierarchy",
    "exp26": "experiments.exp26_cka_denoising",
    "exp27": "experiments.exp27_tucker_decomposition",
    "exp28": "experiments.exp28_invariant_causal_prediction",
    "exp29": "experiments.exp29_prior_block_modulation",
    "exp30": "experiments.exp30_communication_subspace_grassmannian",
    "exp31": "experiments.exp31_causal_representation_learning",
    "exp32": "experiments.exp32_causal_discovery_interregion",
    "exp33": "experiments.exp33_causal_abstraction_geometry",
    "exp34": "experiments.exp34_distributional_invariance",
    "exp35": "experiments.exp35_score_causal_discovery",
    "exp36": "experiments.exp36_wasserstein_distance",
    "exp37": "experiments.exp37_spd_riemannian",
    "exp38": "experiments.exp38_parallel_transport",
    "exp39": "experiments.exp39_fisher_rao",
    "exp40": "experiments.exp40_das_biological",
    "exp41": "experiments.exp41_cross_paper_bridge",
    "exp42": "experiments.exp42_real_iia",
    "exp43": "experiments.exp43_clustering_validation",
    "exp44": "experiments.exp44_iia_causal_direction",
    "exp45": "experiments.exp45_allen_atlas_validation",
    "exp46": "experiments.exp46_ibl_cross_dataset",
    "exp47": "experiments.exp47_silencing_validation",
    "exp47b": "experiments.exp47b_silencing_real_data",
    "exp48": "experiments.exp48_sufficiency",
    "exp49": "experiments.exp49_specificity",
    "exp50": "experiments.exp50_double_dissociation",
    "exp51": "experiments.exp51_confound_control",
    "exp52": "experiments.exp52_multi_method_intervention",
    "exp53": "experiments.exp53_graded_response",
    "exp54": "experiments.exp54_baseline_separation",
    "exp57": "experiments.exp57_structured_vae",
    "exp57b": "experiments.exp57b_vae_k1",
    "exp57c": "experiments.exp57c_vae_k2",
    "exp58": "experiments.exp58_multi_task",
    "exp58_moe": "experiments.exp58_moe_vae",
    "exp59": "experiments.exp59_grassmannian_vae",
    "exp59_sutter": "experiments.exp59_sutter_dilemma",
    "exp60": "experiments.exp60_cross_task_grassmannian",
    "exp61": "experiments.exp61_engagement_subspace",
    "exp62": "experiments.exp62_shuffled_label_control",
    "exp63": "experiments.exp63_linear_vae_ablation",
    "exp64": "experiments.exp64_hierarchical_silencing",
    "exp65": "experiments.exp65_temporal_iia",
    "exp66": "experiments.exp66_per_mouse_silencing",
    "exp67": "experiments.exp67_potent_null_space",
    "exp68": "experiments.exp68_subspace_dissimilarity_matrix",
    "exp69": "experiments.exp69_pivae_hybrid",
    "exp70": "experiments.exp70_cross_region_patching",
    "exp71": "experiments.exp71_vae_causal_circuits",
    "exp72": "experiments.exp72_sutter_continuous",
    "exp73": "experiments.exp73_structured_pi_sae",
    "exp74": "batch2_reviewer_fixes.exp74_debiased_cka",
    "exp75": "batch2_reviewer_fixes.exp75_ccf_coordinate_matching",
    "exp76": "batch2_reviewer_fixes.exp76_umap_stochasticity",
    "exp77": "batch2_reviewer_fixes.exp77_alpha_bias_robustness",
    "exp78": "batch2_reviewer_fixes.exp78_optogenetic_power_analysis",
    "exp80": "batch2_reviewer_fixes.exp80_ivae_verification",
    "exp81": "batch2_reviewer_fixes.exp81_cdnod_region_graph",
    "debug_ibl": "experiments.debug_ibl",
}

SMALL_SESSIONS = {
    "exp1": 3, "exp2": 3, "exp3": 3,
    "exp4": 5, "exp5": 5, "exp6": 5,
    "exp7": 5, "exp7_controls": 5, "exp8": 5, "exp9": 5,
    "exp10": 5,
    "exp11": 5, "exp12": 5, "exp13": 5, "exp14": 5,
    "exp15": 5, "exp16": 5, "exp17": 5, "exp18": 5, "exp19": 5, "exp20": 5,
    "exp15b": 3,
    "exp21": 5,
    "exp22": 5,
    "exp23": 5,
    "exp24": 5,
    "exp25": 5,
    "exp26": 5,
    "exp27": 5,
    "exp28": 5,
    "exp29": 5,
    "exp30": 5,
    "exp31": 5,
    "exp32": 5,
    "exp33": 5,
    "exp34": 5,
    "exp35": 5,
    "exp36": 5,
    "exp37": 5,
    "exp38": 5,
    "exp39": 5,
    "exp40": 5,
    "exp41": 5,
    "exp42": 5,
    "exp43": 5,
    "exp44": 5,
    "exp45": 5,
    "exp46": 5,
    "exp47": 5,
    "exp47b": 5,
    "exp48": 5,
    "exp49": 5,
    "exp50": 5,
    "exp51": 5,
    "exp52": 5,
    "exp53": 5,
    "exp54": 5,
    "exp57": 5,
    "exp58": 5,
    "exp58_moe": 5,
    "exp59": 5,
    "exp59_sutter": 5,
    "exp60": 5,
    "exp61": 5,
    "exp62": 5,
    "exp63": 5,
    "exp64": 5,
    "exp65": 5,
    "exp66": 5,
    "exp67": 5,
    "exp68": 5,
    "exp69": 5,
    "exp70": 5,
    "exp71": 5,
    "exp72": 5,
    "exp73": 5,
    "exp74": 5,
    "exp75": 5,
    "exp76": 5,
    "exp77": 5,
    "exp78": 5,
    "exp80": 5,
    "exp81": 5,
}

IBL_EXPERIMENTS = {"exp1", "exp2", "exp3", "exp15b"}


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600 * 4,
    volumes={"/results": volume},
    secrets=[gcs_secret],
)
def run_experiment(experiment: str, max_sessions: int | None = None, model_filter: str | None = None):
    import importlib
    import logging
    import traceback

    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")

    sa_json = os.environ.get("GCS_SA_JSON", "{}")
    if sa_json != "{}":
        sa_path = Path("/secrets/gcs-sa.json")
        sa_path.parent.mkdir(parents=True, exist_ok=True)
        sa_path.write_text(sa_json)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("modal_run")

    results_dir = Path("/results") / experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{experiment}.jsonl"

    def _log(event, **extra):
        entry = {"ts": datetime.now().isoformat(), "experiment": experiment, "event": event, **extra}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        volume.commit()
        logger.info(f"[{entry['ts']}] {event}" + (f" — {extra}" if extra else ""))

    if experiment not in EXPERIMENT_MAP:
        _log("error", msg=f"Unknown experiment: {experiment}")
        raise ValueError(f"Unknown experiment: {experiment}")

    module_name = EXPERIMENT_MAP[experiment]
    _log("started", module=module_name, max_sessions=max_sessions)

    try:
        mod = importlib.import_module(module_name)
        kwargs = {}
        if max_sessions is not None:
            kwargs["max_sessions"] = max_sessions
        if model_filter is not None:
            kwargs["model_filter"] = model_filter

        results = mod.run(**kwargs)
        _log("run_complete", n_keys=len(results) if isinstance(results, dict) else None)
    except Exception as e:
        _log("run_failed", error=str(e), traceback=traceback.format_exc())
        raise

    try:
        out_path = results_dir / f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        volume.commit()
        _log("saved_volume", path=str(out_path))
    except Exception as e:
        _log("save_failed", error=str(e), traceback=traceback.format_exc())

    local_artifacts = Path("/root/repo/artifacts")
    if local_artifacts.exists():
        for artifact_file in local_artifacts.rglob("*.json*"):
            dest = results_dir / artifact_file.name
            dest.write_text(artifact_file.read_text())
        volume.commit()
        _log("copied_artifacts")

    try:
        from data.gcs_cache import upload_results
        upload_results(experiment, results)
        _log("uploaded_gcs")
    except Exception as e:
        _log("gcs_failed", error=str(e))

    _log("done")
    return results


@app.local_entrypoint()
def main(
    experiment: str = "exp7",
    max_sessions: int | None = None,
    small: bool = False,
    model_filter: str | None = None,
):
    experiments = sorted(EXPERIMENT_MAP) if experiment == "all" else [e.strip() for e in experiment.split(",")]

    for exp in experiments:
        if exp not in EXPERIMENT_MAP:
            print(f"Unknown experiment: {exp}. Choose from: {sorted(EXPERIMENT_MAP)}")
            return

    if len(experiments) == 1:
        ms = max_sessions or (SMALL_SESSIONS.get(experiments[0]) if small else None)
        tag = f" (small, max_sessions={ms})" if ms else ""
        mf_tag = f" model={model_filter}" if model_filter else ""
        print(f"Running {experiments[0]}{tag}{mf_tag}...")
        result = run_experiment.remote(experiments[0], ms, model_filter)
        print(json.dumps(result, indent=2, default=str)[:3000])
    else:
        handles = []
        for exp in experiments:
            ms = max_sessions or (SMALL_SESSIONS.get(exp) if small else None)
            tag = f" (max_sessions={ms})" if ms else ""
            print(f"Spawning {exp}{tag}...")
            handles.append(run_experiment.spawn(exp, ms, model_filter))
        print(f"\nLaunched {len(handles)} experiments. Results on Modal volume + GCS.")
        print("Check results: modal volume get neuro-causal-geometry-results <exp>/")
