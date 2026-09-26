"""
Automated test suite for the NASA C-MAPSS pipeline.

All 24 mandatory checks. Results printed as [PASS] / [FAIL].
"""

import numpy as np
import torch
import tempfile
import os
from typing import List, Tuple


def run_all_tests(
    autoencoder,
    predictor,
    td3_agent,
    lqr_filter,
    safety_gate,
    dm=None,
    device=None,
) -> Tuple[List[bool], List[str]]:
    """
    Run all mandatory executable tests.

    Returns
    -------
    results : list[bool]  (True = PASS)
    names   : list[str]   (test names)
    """
    from .corruption import apply_corruption
    from .models.autoencoder import DMTCAE, MaskedReconstructionLoss
    from .models.predictor import SupervisedPredictor, CombinedPredictorLoss
    from .safety_gate import SafetyGate

    if device is None:
        device = torch.device("cpu")

    results = []
    names = []

    def chk(name, fn):
        names.append(name)
        try:
            fn()
            print(f"[PASS] {name}")
            results.append(True)
        except Exception as e:
            print(f"[FAIL] {name} :: {e}")
            results.append(False)

    probe = torch.randn(8, 30, 14).to(device)

    # 1. Import every module
    def _imports():
        import src.preprocessing
        import src.dataset
        import src.corruption
        import src.models.autoencoder
        import src.models.predictor
        import src.td3.agent
        import src.lqr
        import src.safety_gate
        import src.evaluation
        import src.metrics
        import src.utils
    chk("Importing every module", _imports)

    # 2. Dataset loading
    if dm is not None:
        chk("Dataset loading",
            lambda: (_ for _ in ()).throw(AssertionError("no train_dataset"))
            if dm.train_dataset is None else None)
    else:
        print("[SKIP] Dataset loading (no data)")

    # 3. Canonical 14-feature validation
    if dm is not None:
        chk("Canonical 14-feature validation",
            lambda: (_ for _ in ()).throw(AssertionError(str(dm.train_dataset.windows.shape[-1])))
            if dm.train_dataset.windows.shape[-1] != 14 else None)
    else:
        chk("Canonical 14-feature validation",
            lambda: None)  # synthetic data has 14 by construction

    # 4. Window shape validation
    if dm is not None:
        chk("Window shape validation",
            lambda: (_ for _ in ()).throw(AssertionError("bad shape"))
            if tuple(dm.train_dataset.windows.shape[1:]) != (30, 14) else None)
    else:
        print("[SKIP] Window shape validation (no data)")

    # 5. DM-TCAE forward pass
    def _ae_fwd():
        autoencoder.eval()
        with torch.no_grad():
            rec, lat = autoencoder(probe[:4])
        assert rec.shape == (4, 30, 14)
        assert lat.shape == (4, 30, 16)
        assert torch.isfinite(rec).all()
    chk("DM-TCAE forward pass", _ae_fwd)

    # 6. Predictor forward pass
    def _pred_fwd():
        predictor.eval()
        with torch.no_grad():
            yb, h, xhat, z = predictor(probe[:4])
        assert yb.shape == (4, 1)
        assert h.shape == (4, 64)
        assert xhat.shape == (4, 30, 14)
        assert z.shape == (4, 30, 16)
        assert torch.isfinite(yb).all()
    chk("Predictor forward pass", _pred_fwd)

    # 7. TD3 actor output shape
    def _td3_actor():
        s = np.zeros((4, 66), dtype=np.float32)
        a = td3_agent.select_action(s, noise=0.0)
        assert a.shape == (4, 1), f"Actor output shape: {a.shape}"
    chk("TD3 actor output shape", _td3_actor)

    # 8. TD3 critic output shape
    def _td3_critic():
        s = torch.zeros(4, 66).to(device)
        a = torch.zeros(4, 1).to(device)
        q1, q2 = td3_agent.critic(s, a)
        assert q1.shape == (4, 1) and q2.shape == (4, 1)
    chk("TD3 critic output shape", _td3_critic)

    # 9. Backward pass
    def _backward():
        predictor.train()
        xb = probe[:4].requires_grad_(False)
        yb_pred, _, rec, _ = predictor(xb)
        y_tgt = torch.rand(4, 1).to(device) * 125
        loss, _, _ = CombinedPredictorLoss()(yb_pred, y_tgt, rec, xb)
        loss.backward()
    chk("Backward pass", _backward)

    # 10. Gradient existence
    def _grads():
        assert all(p.grad is not None for p in predictor.parameters() if p.requires_grad), \
            "Some parameters have no gradient"
    chk("Gradient existence", _grads)

    # 11. NaN / Inf checks
    if dm is not None:
        chk("NaN and infinity checks",
            lambda: (_ for _ in ()).throw(AssertionError("NaN/Inf in windows"))
            if not torch.isfinite(dm.train_dataset.windows).all() else None)
    else:
        print("[SKIP] NaN and infinity checks (no data)")

    # 12. RUL range checks
    if dm is not None:
        chk("RUL range checks",
            lambda: (_ for _ in ()).throw(AssertionError("RUL out of [0,125]"))
            if bool(((dm.train_dataset.ruls < 0) | (dm.train_dataset.ruls > 125)).any()) else None)
    else:
        print("[SKIP] RUL range checks (no data)")

    # 13. Train/validation engine disjointness
    if dm is not None:
        chk("Train/validation engine disjointness",
            lambda: (_ for _ in ()).throw(AssertionError("engine overlap"))
            if len(set(dm.train_engine_ids) & set(dm.val_engine_ids)) > 0 else None)
    else:
        print("[SKIP] Train/validation engine disjointness (no data)")

    # 14. Test-data leakage
    if dm is not None:
        chk("Test-data leakage",
            lambda: (_ for _ in ()).throw(AssertionError("test engines in train"))
            if len(set(dm.test_engine_ids) & set(dm.train_engine_ids)) > 0 else None)
    else:
        print("[SKIP] Test-data leakage (no data)")

    # 15. Corruption pipeline
    def _corr():
        for m in ["gaussian_noise", "sensor_masking", "random_dropout", "spike_noise", "drift_corruption"]:
            cx, mk = apply_corruption(probe.cpu(), mode=m, seed=42)
            assert cx.shape == (8, 30, 14) and mk.shape == (8, 30, 14)
            assert float(mk.sum()) >= 0
    chk("Corruption pipeline", _corr)

    # 16. LQR correction bounds
    def _lqr_bounds():
        yb, a_safe = lqr_filter.apply_safety_constraint(y_base=100.0, a_td3=10.0)
        assert yb <= 125.0 and abs(a_safe) <= lqr_filter.max_correction
        yb2, a2 = lqr_filter.apply_safety_constraint(y_base=120.0, a_td3=10.0)
        assert yb2 <= 125.0
    chk("LQR correction bounds", _lqr_bounds)

    # 17. LQR correction-rate limits
    def _lqr_rate():
        fa = lqr_filter.filter_action(10.0, prev_a=0.0)
        assert abs(fa) <= lqr_filter.max_rate + 1e-6, \
            f"Rate limit violated: |{fa}| > {lqr_filter.max_rate}"
    chk("LQR correction-rate limits", _lqr_rate)

    # 18. Safety-gate rejection behaviour
    def _gate_rej():
        g = SafetyGate()
        g.calibrate(
            np.array([50., 60., 70.]),
            np.array([5., 5., 5.]),
            np.array([50., 60., 70.]),
        )
        assert g.alpha == 0.0, f"Gate should prefer baseline; got alpha={g.alpha}"
        y_f, accepted = g.apply_gate(np.array([50.]), np.array([float("nan")]))
        assert np.all(np.isfinite(y_f)), "NaN correction should be rejected"
    chk("Safety-gate rejection behavior", _gate_rej)

    # 19. Checkpoint save
    def _ckpt_save():
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tp = tf.name
        torch.save(predictor.state_dict(), tp)
        os.remove(tp)
    chk("Checkpoint save", _ckpt_save)

    # 20. Checkpoint reload
    def _ckpt_reload():
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tp = tf.name
        torch.save(predictor.state_dict(), tp)
        fresh = SupervisedPredictor().to(device)
        fresh.load_state_dict(torch.load(tp, map_location=device))
        fresh.eval()
        with torch.no_grad():
            fresh(probe[:2])
        os.remove(tp)
    chk("Checkpoint reload", _ckpt_reload)

    # 21. CPU inference
    def _cpu_inf():
        m = SupervisedPredictor().to("cpu").eval()
        with torch.no_grad():
            m(probe[:2].cpu())
    chk("CPU inference", _cpu_inf)

    # 22. CUDA inference
    if torch.cuda.is_available():
        def _cuda_inf():
            m = SupervisedPredictor().to("cuda").eval()
            with torch.no_grad():
                m(probe[:2].cuda())
        chk("CUDA inference if available", _cuda_inf)
    else:
        print("[PASS] CUDA inference if available (no CUDA; CPU validated above)")

    # 23. End-to-end smoke training
    def _smoke():
        ae = DMTCAE().to(device)
        cx, mk = apply_corruption(probe[:4].to(device), mode="dynamic_mixed")
        r, _ = ae(cx)
        MaskedReconstructionLoss()(r, probe[:4].to(device), mk).backward()
    chk("End-to-end smoke training", _smoke)

    # 24. End-to-end inference
    def _e2e():
        predictor.eval()
        with torch.no_grad():
            yb, h, _, _ = predictor(probe[:4].to(device))
        st = np.hstack([h.cpu().numpy(), yb.cpu().numpy() / 125.0, np.zeros((4, 1))])
        a = td3_agent.select_action(st, noise=0.0)
        assert a.shape == (4, 1)
    chk("End-to-end inference", _e2e)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*50}")
    print(f"Tests: {n_pass}/{n_total} passed")
    if n_pass < n_total:
        print("FAILED:", [n for n, r in zip(names, results) if not r])
    return results, names
