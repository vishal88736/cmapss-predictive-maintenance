"""Serializable configuration shared by the notebook and website."""

FORMAT_VERSION = 7
ARCHITECTURE = "validated-detector-linear-identity-gated-cnn-gru-transformer-residual-v7"
SUPPORTED_ARCHITECTURES = {
    5: "detector-residual-bilstm-wgan-gated-cnn-gru-transformer-residual-v5",
    6: "asinh-balanced-recovery-gated-cnn-gru-transformer-residual-v6",
    FORMAT_VERSION: ARCHITECTURE,
}
CORRUPTION_PROTOCOL = "structured-sensor-faults-v4"
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = SETTINGS + SENSORS
COLUMNS = ["unit", "cycle"] + FEATURE_COLUMNS
SENSOR_NAMES = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf-dmd", "PCNfR-dmd", "W31", "W32"]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_CONFIG = {
    "device": "cuda", "validation_fraction": 0.20,
    "calibration_fraction_of_training": 0.10,
    "subsets": SUBSETS, "sequence_length": 70, "local_length": 30,
    "target_scale": 125.0, "train_rul_cap": 125.0, "generator_hidden": 64,
    "sensor_transform": "asinh", "sensor_transform_scale": 1.0,
    "repair_loss": "smooth_l1", "repair_huber_beta": 1.0,
    "detector_loss": "balanced", "validate_bypass": True,
    "identity_loss_space": "original_linear", "detach_repair_gate": True,
    "finite_repair_gate": "validated_threshold", "detector_training": "separate",
    "detector_refinement_epochs": 15, "detector_refinement_learning_rate": 0.001,
    "detector_refinement_patience": 5, "detector_max_false_positive_rate": 0.05,
    "detector_refinement_min_improvement": 0.0001,
    "generator_layers": 2, "cnn_channels": 32, "gru_hidden": 64,
    "transformer_dim": 64, "transformer_heads": 4, "transformer_layers": 2,
    "dropout": 0.2, "correction_bound": 15.0, "correction_penalty": 0.02,
    "correction_bounds": [5.0, 10.0, 15.0, 30.0],
    "interval_alpha": 0.1, "calibration_policy": "engine_prefix_max",
    "mc_passes": 20, "batch_size": 128,
    "predictor_epochs": 100, "generator_epochs": 50, "joint_epochs": 20,
    "correction_epochs": 50, "patience": 15, "learning_rate": 0.001,
    "repair_learning_rate": 0.0001, "joint_learning_rate": 0.0001,
    "critic_steps": 1, "gradient_penalty": 10.0, "gradient_penalty_batch_size": 16,
    "adversarial_warmup": 5,
    "oof_folds": 3, "window_stride": 1, "clean_fraction": 0.25,
    "fault_start_horizon": 100, "drift_ramp_cycles": 100, "block_gap_cycles": 50,
    "fault_noise_autocorrelation": .9, "fault_transition_gain": 1., "intermittent_mean_burst": 4,
    "curriculum_epochs": 10, "train_corrupted_predictor": True,
    "split_seed": 42, "validation_seed": 31415, "validation_prefixes": 3,
    "correction_prefixes": 6, "engine_balanced_sampling": True,
    "rul_loss_weight": 1.0, "reconstruction_loss_weight": 1.0,
    "generator_reconstruction_loss_weight": 0.25, "identity_loss_weight": 0.5,
    "detection_loss_weight": 0.2, "detector_positive_weight_cap": 5.0,
    "trend_loss_weight": 0.1, "adversarial_loss_weight": 0.01,
    "consistency_loss_weight": 0.1, "branch_loss_weight": 0.1,
    "clean_rul_loss_weight": 0.25,
    "bypass_rul_loss_weight": 0.5,
    "lr_factor": 0.5, "lr_patience": 5, "minimum_learning_rate": 0.000001,
    "max_clean_rmse_increase": 0.01, "max_intact_repair_rmse": 0.15,
    "max_clean_nasa_increase": 0.01,
    "max_reconstruction_rmse": 1.5,
    "max_linear_reconstruction_ratio": 1.0,
    "predictor_family": "gated", "predictor_branch": "both", "equal_fusion": False,
    "train_repair": True,
    "tune_before_training": True, "run_matched_benchmarks": False,
    "run_extended_evaluation": False, "benchmark_tune": True,
    "benchmark_variants": ["paper_A_cnn_gru", "paper_B_repair_cnn_lstm",
                           "paper_C_transformer_corrected", "grace_without_adversarial",
                           "grace_without_trend", "grace_without_repair", "grace_local",
                           "grace_global", "grace_equal_fusion", "grace_without_correction",
                           "pure_transformer"],
    "bootstrap_samples": 1000,
    "seeds": [42, 123, 2026, 314, 777], "seed": 42, "quick_run": False,
    "maximum_batches": None, "maximum_engines": None, "maximum_windows": None,
    "background_size": 48, "explanation_samples": 256, "explanation_examples": 12,
    "explanation_repeats": 3, "cpu_threads": 2,
    "progress_interval_seconds": 30.0,
}


def quick_config(config):
    """A visibly labeled execution check, never a scientific benchmark."""
    return {**config, "quick_run": True, "subsets": ["FD001"], "seeds": [42],
            "predictor_epochs": 1, "generator_epochs": 1, "joint_epochs": 1,
            "correction_epochs": 2, "maximum_batches": 2, "maximum_engines": 18,
            "detector_refinement_epochs": 1, "detector_refinement_patience": 1,
            "maximum_windows": 48, "batch_size": 8, "generator_hidden": 16,
            "cnn_channels": 8, "gru_hidden": 16, "transformer_dim": 32,
            "adversarial_warmup": 0, "background_size": 3, "mc_passes": 4,
            "explanation_examples": 2, "explanation_repeats": 1,
            "explanation_samples": 64, "window_stride": 5,
            "validation_prefixes": 2, "correction_prefixes": 3,
            "correction_bounds": [15.0], "tune_before_training": False,
            "run_matched_benchmarks": False, "run_extended_evaluation": True,
            "benchmark_variants": ["paper_A_cnn_gru", "paper_B_repair_cnn_lstm",
                                   "paper_C_transformer_corrected", "grace_without_repair"],
            "bootstrap_samples": 100}
