"""
Evaluation module: full inference pipeline and artifact generation.

Only called during Stage F with TEST data.
Validation data is NEVER passed to this function.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional

from .metrics import calculate_metrics
from .corruption import apply_corruption


DEFAULT_CORRUPTION_MODES = [
    "clean",
    "gaussian_noise",
    "sensor_masking",
    "random_dropout",
    "spike_noise",
    "drift_corruption",
]


def run_full_inference(
    predictor,
    td3_agent,
    lqr_filter,
    safety_gate,
    windows: torch.Tensor,    # [N, W, 14]
    ruls: torch.Tensor,       # [N]
    norm_cycles: torch.Tensor, # [N]
    engine_ids: Optional[torch.Tensor] = None,  # [N]
    device: Optional[torch.device] = None,
    rul_scale: float = 125.0,
    corruption_mode: str = "clean",
    batch_size: int = 256,
    max_rul: float = 125.0,
    conservative_bias: float = 5.0,
) -> Dict:
    """
    Run the full Stage F inference pipeline on a set of windows.

    Steps:
      1. Apply corruption (if mode != 'clean')
      2. Predictor forward pass → y_base, h
      3. TD3 correction
      4. LQR filter
      5. Safety gate

    Returns dict with arrays: y_base, y_corrected, y_final, y_true, accepted
    """
    # Allow calling run_full_inference without engine_ids (device passed as 8th arg)
    if device is None:
        if isinstance(engine_ids, (torch.device, str)):
            device = torch.device(engine_ids)
            engine_ids = None
        else:
            device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    predictor.eval()
    N = len(windows)
    y_base_all, y_corr_all, h_all = [], [], []
    if isinstance(ruls, torch.Tensor):
        y_true_all = ruls.cpu().numpy().flatten()
    else:
        y_true_all = np.asarray(ruls).flatten()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = windows[start:end].to(device)
        y_true_batch = ruls[start:end]
        nc_batch = norm_cycles[start:end]

        if corruption_mode != "clean":
            x_batch, _ = apply_corruption(x_batch, mode=corruption_mode)

        with torch.no_grad():
            y_b, h, _, _ = predictor(x_batch)

        y_base_all.extend(y_b.cpu().numpy().flatten())
        h_all.extend(h.cpu().numpy())

    y_base = np.array(y_base_all, dtype=np.float32)
    h_arr = np.array(h_all, dtype=np.float32)
    if isinstance(norm_cycles, torch.Tensor):
        nc_arr = norm_cycles.cpu().numpy().flatten()
    else:
        nc_arr = np.asarray(norm_cycles).flatten()

    # TD3 corrections
    if td3_agent is not None:
        states = np.hstack([
            h_arr,
            (y_base / rul_scale).reshape(-1, 1),
            nc_arr.reshape(-1, 1),
        ])
        a_td3 = td3_agent.select_action(states, noise=0.0).flatten()
    else:
        a_td3 = np.zeros_like(y_base)

    # LQR filter
    a_lqr = np.zeros_like(a_td3)
    prev_a = 0.0
    prev_eid = None
    if engine_ids is None:
        eids_arr = np.arange(N)
    elif isinstance(engine_ids, torch.Tensor):
        eids_arr = engine_ids.cpu().numpy().flatten()
    else:
        eids_arr = np.asarray(engine_ids).flatten()

    for i, (yb, at, eid) in enumerate(zip(y_base, a_td3, eids_arr)):
        if prev_eid is not None and eid != prev_eid:
            prev_a = 0.0
        fa = lqr_filter.filter_action(float(at), prev_a=float(prev_a), y_base=float(yb))
        a_lqr[i] = fa
        prev_a = fa
        prev_eid = eid

    # Safety gate
    y_final, accepted = safety_gate.apply_gate(y_base, a_lqr)
    y_final = np.clip(y_final - conservative_bias, 0.0, max_rul)
    y_corr = np.clip(y_base + a_lqr - conservative_bias, 0.0, max_rul)

    return {
        "y_base": y_base,
        "y_corrected": y_corr,
        "y_final": y_final,
        "y_true": y_true_all,
        "a_td3": a_td3,
        "a_lqr": a_lqr,
        "accepted": accepted,
    }


def evaluate_and_generate_artifacts(
    dataset_name: str,
    output_dir: str,
    predictor,
    baseline_predictor,
    td3_agent,
    lqr_filter,
    safety_gate,
    test_windows: torch.Tensor,
    test_ruls: torch.Tensor,
    norm_cycles: torch.Tensor,
    engine_ids: torch.Tensor,
    device: torch.device,
    corruption_modes: Optional[List[str]] = None,
    max_rul: float = 125.0,
    rul_scale: float = 125.0,
) -> Dict:
    """
    Run full evaluation across all corruption modes and generate:
      - CSV metrics tables
      - Prediction scatter plots
      - Error distribution plots
      - TD3 correction distribution
      - Safety gate acceptance plot
      - Ablation study CSV

    Returns dict with 'clean_metrics' and 'all_metrics'.
    """
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    if corruption_modes is None:
        corruption_modes = DEFAULT_CORRUPTION_MODES

    all_metrics = {}
    clean_metrics = {}

    for mode in corruption_modes:
        print(f"  Evaluating: {mode} ...")
        res = run_full_inference(
            predictor=predictor,
            td3_agent=td3_agent,
            lqr_filter=lqr_filter,
            safety_gate=safety_gate,
            windows=test_windows,
            ruls=test_ruls,
            norm_cycles=norm_cycles,
            engine_ids=engine_ids,
            device=device,
            rul_scale=rul_scale,
            corruption_mode=mode,
            max_rul=max_rul,
        )

        def last_cycle_indices(eids):
            eids = np.asarray(eids).ravel()
            idx = []
            for uid in np.unique(eids):
                idx.append(np.where(eids == uid)[0][-1])
            return np.array(idx)
            
        lc = last_cycle_indices(engine_ids.numpy())
        m = calculate_metrics(res["y_final"][lc], res["y_true"][lc], label=f"{dataset_name}/{mode}/last_cycle")
        all_metrics[mode] = m

        if mode == "clean":
            clean_metrics = m
            # Save predictions CSV
            pred_df = pd.DataFrame({
                "y_true": res["y_true"],
                "y_base": res["y_base"],
                "y_corrected": res["y_corrected"],
                "y_final": res["y_final"],
                "a_td3": res["a_td3"],
                "a_lqr": res["a_lqr"],
                "accepted": res["accepted"].astype(int),
            })
            pred_df.to_csv(os.path.join(output_dir, "predictions", "clean_predictions.csv"), index=False)

        # Save per-mode metrics
        metrics_df = pd.DataFrame([m])
        metrics_df.to_csv(
            os.path.join(output_dir, "metrics", f"corruption_{mode}_metrics.csv"), index=False
        )

    # Baseline evaluation (clean)
    if baseline_predictor is not None:
        b_res = run_full_inference(
            predictor=baseline_predictor,
            td3_agent=None,
            lqr_filter=lqr_filter,
            safety_gate=safety_gate,
            windows=test_windows,
            ruls=test_ruls,
            norm_cycles=norm_cycles,
            engine_ids=engine_ids,
            device=device,
            rul_scale=rul_scale,
            corruption_mode="clean",
            max_rul=max_rul,
            conservative_bias=0.0, # No bias for baseline
        )
        lc = last_cycle_indices(engine_ids.numpy())
        baseline_m = calculate_metrics(b_res["y_base"][lc], b_res["y_true"][lc], label=f"{dataset_name}/baseline/last_cycle")
    else:
        baseline_m = {}

    # Ablation study
    ablation = []
    ablation.append({"variant": "Baseline", **baseline_m})
    ablation.append({"variant": "Full System", **clean_metrics})
    ablation_df = pd.DataFrame(ablation)
    ablation_df.to_csv(os.path.join(output_dir, "metrics", "ablation_results.csv"), index=False)

    # Summary table
    summary = []
    for mode, m in all_metrics.items():
        summary.append({"Corruption": mode, **m})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "metrics", "robustness_summary.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_predictions(res, dataset_name, output_dir)
    _plot_error_distribution(res, baseline_m, dataset_name, output_dir)
    _plot_td3_corrections(res, dataset_name, output_dir)
    _plot_gate_acceptance(res, dataset_name, output_dir)

    return {
        "clean_metrics": clean_metrics,
        "all_metrics": all_metrics,
        "baseline_metrics": baseline_m,
        "ablation": ablation_df,
    }


def _plot_predictions(res: Dict, dataset_name: str, output_dir: str):
    y_t = res["y_true"]
    idx = np.argsort(y_t)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_t[idx], label="True RUL", color="black", alpha=0.7)
    ax.plot(res["y_base"][idx], label="Baseline", color="#1f77b4", alpha=0.7)
    ax.plot(res["y_final"][idx], label="Full System", color="#2ca02c", alpha=0.9)
    ax.set_xlabel("Test Sample (sorted by True RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{dataset_name}: Predicted vs True RUL (Clean Test Set)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "predictions_clean.png"), dpi=150)
    plt.close()


def _plot_error_distribution(res: Dict, baseline_m: Dict, dataset_name: str, output_dir: str):
    errors_full = res["y_final"] - res["y_true"]
    errors_base = res["y_base"] - res["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(errors_base, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black")
    axes[0].axvline(0, color="red", linestyle="--")
    rmse_b = baseline_m.get("RMSE", 0)
    axes[0].set_title(f"Baseline Errors (RMSE={rmse_b:.2f})")
    axes[0].set_xlabel("Prediction Error (cycles)")
    axes[1].hist(errors_full, bins=40, color="#2ca02c", alpha=0.7, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"Full System Errors")
    axes[1].set_xlabel("Prediction Error (cycles)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{dataset_name}: Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "error_distribution.png"), dpi=150)
    plt.close()


def _plot_td3_corrections(res: Dict, dataset_name: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res["a_td3"], bins=40, color="#ff7f0e", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("TD3 Correction (cycles)")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset_name}: TD3 Raw Correction Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "td3_corrections.png"), dpi=150)
    plt.close()


def _plot_gate_acceptance(res: Dict, dataset_name: str, output_dir: str):
    accepted = res["accepted"].astype(int)
    n_acc = accepted.sum()
    n_rej = len(accepted) - n_acc
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Accepted", "Rejected"], [n_acc, n_rej], color=["#2ca02c", "#d62728"])
    ax.set_title(f"{dataset_name}: Safety Gate Decisions")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "gate_acceptance.png"), dpi=150)
    plt.close()
