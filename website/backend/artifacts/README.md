# Trained model placeholder

Place exported, **actually trained** bundles in `FD001/`, `FD002/`, `FD003/`, or `FD004/` here, or point `CMAPSS_MODEL_DIR` at your own artifact directory.

Each subset directory needs `manifest.json`, `predictor.pt`, `td3_actor.pt`, `td3_critic.pt`, `preprocessor.joblib`, and `safety_gate.json`.

The API remains model-unavailable until the complete bundle exists and can be loaded. No weights, predictions, or fitted statistics are generated as placeholders.

See `backend/export_artifacts.py` and the website README for exporting the objects from your completed training session.
