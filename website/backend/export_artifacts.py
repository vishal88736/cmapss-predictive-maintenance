"""Export existing trained objects from a completed notebook; this does not train."""
import inspect
import json
from pathlib import Path
import joblib
import torch
from .project import CANONICAL_14_SENSORS, SUBSETS
from src.evaluation import run_full_inference


def export_bundle(destination, subset, predictor, td3_agent, lqr_filter, safety_gate, data_module, *, window_selection):
    """Use 'last_window' for the current notebook, 'all_windows' for the .py run."""
    if subset not in SUBSETS or window_selection not in ("all_windows", "last_window"):
        raise ValueError("Record the actual subset and window-selection policy of the trained run.")
    if data_module.dataset_name != subset:
        raise ValueError("The exported subset must match the trained data module.")
    if data_module.preprocessor.kmeans is None:
        raise ValueError("Export the fitted training preprocessor, not a fresh instance.")
    root = Path(destination) / subset
    root.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.state_dict(), root / "predictor.pt")
    td3_agent.save(str(root / "td3_actor.pt"), str(root / "td3_critic.pt"))
    joblib.dump(data_module.preprocessor, root / "preprocessor.joblib")
    safety_gate.save(str(root / "safety_gate.json"))
    layer = predictor.transformer.layers[0]
    signature = inspect.signature(run_full_inference)
    manifest = {
        "schemaVersion": 1, "subset": subset,
        "modelName": "DM-TCAE + CNN–Transformer + TD3 / LQR / Safety Gate",
        "features": CANONICAL_14_SENSORS,
        "windowLength": data_module.window_length,
        "windowSelection": window_selection,
        "predictorParameters": {
            "in_channels": predictor.in_channels, "latent_channels": predictor.latent_channels,
            "d_model": layer.self_attn.embed_dim, "n_heads": layer.self_attn.num_heads,
            "n_layers": len(predictor.transformer.layers), "ffn_dim": layer.linear1.out_features,
            "dropout": layer.dropout.p, "max_rul": predictor.max_rul,
        },
        "td3Parameters": {"state_dim": td3_agent.actor.net[0].in_features, "action_dim": td3_agent.actor.net[-2].out_features, "max_action": td3_agent.max_action},
        "lqrParameters": {name: getattr(lqr_filter, name) for name in ("Q", "R", "max_correction", "max_rate", "max_rul")},
        "inferenceParameters": {name: signature.parameters[name].default for name in ("rul_scale", "batch_size", "max_rul", "conservative_bias")},
        "source": "Exported existing trained objects; no training executed by the web adapter",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root
