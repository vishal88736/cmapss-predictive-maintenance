# @title 🚀 One-Click Colab / Kaggle Setup (Auto-Bootstrap `src` and `data`)
# Run this cell first on Google Colab or Kaggle.
import os, sys, zlib, json, base64, urllib.request, subprocess, shutil

CWD = os.getcwd()
for p in [CWD]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import optuna
except ImportError:
    print('Installing optuna for hyperparameter optimization...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'optuna'])
    import optuna



data_dir = '/home/vishal/D drive/minor/Data'
os.makedirs(data_dir, exist_ok=True)
base_url = 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/'
all_files = [f"{p}_{ds}.txt" for ds in ['FD001', 'FD002', 'FD003', 'FD004'] for p in ['train', 'test', 'RUL']]

def download_file(url, dest):
    try:
        subprocess.run(['curl', '-sL', '-o', dest, url], timeout=15, check=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 0: return True
    except: pass
    try:
        with urllib.request.urlopen(url, timeout=15) as r, open(dest, 'wb') as f: shutil.copyfileobj(r, f)
        return True
    except: return False

missing = [f for f in all_files if not os.path.exists(os.path.join(data_dir, f)) or os.path.getsize(os.path.join(data_dir, f)) == 0]
if missing:
    print('Downloading', len(missing), 'missing C-MAPSS dataset files...')
    for fname in missing:
        if download_file(base_url + fname, os.path.join(data_dir, fname)):
            print('  Fetched', fname)
else:
    print('✅ C-MAPSS data files ready.')
print('✅ Environment ready!')

# -------------------

import os, sys, time, copy, warnings
import torch
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from src.dataset import CMAPSSDataModule
from src.corruption import apply_corruption
from src.models.autoencoder import DMTCAE, MaskedReconstructionLoss
from src.models.predictor import SupervisedPredictor, CombinedPredictorLoss
from src.td3.agent import TD3Agent
from src.lqr import LQRSafetyFilter
from src.safety_gate import SafetyGate
from src.evaluation import run_full_inference
from src.metrics import calculate_metrics
from src.utils import set_seed, get_device

# --- SOTA CONFIGURATION FOR < 15 AVERAGE RMSE ---
SEED = 42
ALL_DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']

# Architectural tweaks to push boundaries
WINDOW_LENGTH = 40        # Increased from 30 for richer historical context
MAX_RUL = 125.0

# HPO settings massively upgraded
N_OPTUNA_TRIALS = 20      # More trials to find perfect architecture
EPOCHS_B_HPO = 15         # Longer evaluation during tuning
EPOCHS_B_FINAL = 50       # Extreme depth training (50 epochs) with OneCycleLR
EPOCHS_A = 15             # Better Autoencoder pretraining
TD3_STEPS = 1500          # 1500 RL steps for deep convergence on residuals
MAX_ACTION_RL = 12.0      # Widen bounds to let TD3 correct severe FD002/FD004 errors

set_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    raise RuntimeError("GPU requested but neither CUDA nor MPS is available in this environment. Please ensure PyTorch is installed with GPU support.")
print(f"Using Device: {DEVICE}")

# -------------------

final_results = {}

def print_optuna_callback(study, trial):
    best = study.best_value
    print(f"  [Trial {trial.number+1:02d}/{N_OPTUNA_TRIALS}] Val RMSE: {trial.value:.3f} | Best so far: {best:.3f} | Params: {trial.params}")

optuna.logging.set_verbosity(optuna.logging.WARNING)

for dataset_name in ALL_DATASETS:
    print("\n" + "="*75)
    print(f"🚀 EXTREME TRAINING PHASE: {dataset_name}")
    print("="*75)
    
    # 1. Load Data
    dm = CMAPSSDataModule(data_dir=data_dir, dataset_name=dataset_name, window_length=WINDOW_LENGTH, seed=SEED, max_rul=MAX_RUL)
    dm.load_and_preprocess(verbose=False)
    train_loader, val_loader, test_loader = dm.get_loaders(batch_size=64)
    print(f"Data Loaded -> Train Windows: {len(dm.train_dataset)}, Val Windows: {len(dm.val_dataset)}")
    
    # 2. Stage A: DM-TCAE Pretraining
    print("\n--- STAGE A: Deep DM-TCAE Pretraining ---")
    autoencoder = DMTCAE(14, 16).to(DEVICE)
    opt_ae = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    crit_ae = MaskedReconstructionLoss()
    
    for ep in range(EPOCHS_A):
        autoencoder.train()
        losses = []
        for x, y, _, _, _ in train_loader:
            x = x.to(DEVICE)
            cx, mk = apply_corruption(x, mode='dynamic_mixed')
            rec, _ = autoencoder(cx)
            loss = crit_ae(rec, x, mk)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
            losses.append(loss.item())
        if (ep+1) % 5 == 0:
            print(f"  Epoch {ep+1}/{EPOCHS_A} | TCAE Loss: {np.mean(losses):.4f}")
            
    # 3. Stage B: Optuna HPO
    print("\n--- STAGE B: Deep HPO Search ---")
    def objective(trial):
        lr_b     = trial.suggest_float('lr_b', 1e-4, 2e-3, log=True)
        batch_sz = trial.suggest_categorical('batch_size', [32, 64])
        dropout  = trial.suggest_float('dropout', 0.05, 0.25)
        lam_rec  = trial.suggest_float('lambda_rec', 0.05, 0.3)
        n_layers = trial.suggest_int('n_layers', 2, 4) # Deeper networks allowed
        n_heads  = trial.suggest_categorical('n_heads', [4, 8])
        ffn_dim  = trial.suggest_categorical('ffn_dim', [64, 128, 256])
        weight_d = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        t_ld, v_ld, _ = dm.get_loaders(batch_size=batch_sz)
        
        ae_clone = copy.deepcopy(autoencoder)
        model = SupervisedPredictor(
            in_channels=14, latent_channels=16, d_model=64, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
            dropout=dropout, max_rul=MAX_RUL, pretrained_autoencoder=ae_clone
        ).to(DEVICE)
        
        opt = torch.optim.AdamW(model.parameters(), lr=lr_b, weight_decay=weight_d)
        crit = CombinedPredictorLoss(loss_type='huber', lambda_rec=lam_rec)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_B_HPO)
        
        best_val = float('inf')
        patience = 0
        
        for ep in range(EPOCHS_B_HPO):
            model.train()
            for x, y, _, _, _ in t_ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                cx, _ = apply_corruption(x, mode='dynamic_mixed')
                yb, _, rec, _ = model(cx)
                loss, _, _ = crit(yb, y, rec, x)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            
            model.eval()
            preds, trues, eids_list = [], [], []
            with torch.no_grad():
                for x, y, _, eid, _ in v_ld:
                    yb, _, _, _ = model(x.to(DEVICE))
                    preds.extend(yb.cpu().numpy().flatten())
                    trues.extend(y.numpy().flatten())
                    eids_list.extend(eid.numpy())
            
            from src.metrics import cmapss_score
            def get_lc(eids):
                eids = np.asarray(eids).ravel()
                idx = []
                for uid in np.unique(eids):
                    idx.append(np.where(eids == uid)[0][-1])
                return np.array(idx)
            
            lc_idx = get_lc(eids_list)
            val_score = cmapss_score(np.array(preds)[lc_idx], np.array(trues)[lc_idx])
            trial.report(val_score, ep)
            
            if val_score < best_val:
                best_val = val_score
                patience = 0
            else:
                patience += 1
                
            if patience >= 4 or trial.should_prune():
                break
        return best_val

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, callbacks=[print_optuna_callback])
    
    bp = study.best_params
    print(f"\n🏆 Best HPO Val Score: {study.best_value:.3f}")
    
    # 4. Final Stage B Training with OneCycleLR
    print("\n--- STAGE B: 50-Epoch Extreme Training with OneCycleLR ---")
    t_ld, v_ld, _ = dm.get_loaders(batch_size=bp['batch_size'])
    predictor = SupervisedPredictor(
        in_channels=14, latent_channels=16, d_model=64, n_heads=bp['n_heads'], n_layers=bp['n_layers'], ffn_dim=bp['ffn_dim'],
        dropout=bp['dropout'], max_rul=MAX_RUL, pretrained_autoencoder=copy.deepcopy(autoencoder)
    ).to(DEVICE)
    
    opt_f = torch.optim.AdamW(predictor.parameters(), lr=bp['lr_b'], weight_decay=bp['weight_decay'])
    crit_f = CombinedPredictorLoss(loss_type='huber', lambda_rec=bp['lambda_rec'])
    
    # Super-convergence scheduler
    sched_f = torch.optim.lr_scheduler.OneCycleLR(opt_f, max_lr=bp['lr_b'] * 1.5, steps_per_epoch=len(t_ld), epochs=EPOCHS_B_FINAL)
    
    best_final_rmse = float('inf')
    best_state = None
    for ep in range(EPOCHS_B_FINAL):
        predictor.train()
        for x, y, _, _, _ in t_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            cx, _ = apply_corruption(x, mode='dynamic_mixed')
            yb, _, rec, _ = predictor(cx)
            loss, _, _ = crit_f(yb, y, rec, x)
            opt_f.zero_grad(); loss.backward(); opt_f.step()
            sched_f.step()
        
        predictor.eval()
        preds, trues, eids_list = [], [], []
        with torch.no_grad():
            for x, y, _, eid, _ in v_ld:
                yb, _, _, _ = predictor(x.to(DEVICE))
                preds.extend(yb.cpu().numpy().flatten())
                trues.extend(y.numpy().flatten())
                eids_list.extend(eid.numpy())
        
        from src.metrics import cmapss_score
        def get_lc(eids):
            eids = np.asarray(eids).ravel()
            idx = []
            for uid in np.unique(eids):
                idx.append(np.where(eids == uid)[0][-1])
            return np.array(idx)
            
        lc_idx = get_lc(eids_list)
        val_score = cmapss_score(np.array(preds)[lc_idx], np.array(trues)[lc_idx])
        if val_score < best_final_rmse:
            best_final_rmse = val_score
            best_state = copy.deepcopy(predictor.state_dict())
            
        if (ep+1) % 5 == 0 or (ep+1) == EPOCHS_B_FINAL:
            print(f"  Epoch {ep+1:02d}/{EPOCHS_B_FINAL} | Val Score: {val_score:.3f}")
            
    predictor.load_state_dict(best_state)
    print(f"Final Model Loaded. Best Val Score: {best_final_rmse:.3f}")
    
    # 5. Stage C-E: TD3 + LQR + Gate (Relaxed Bounds for Max Impact)
    print("\n--- STAGE C-E: Unlocked TD3 Offline Correction ---")
    td3_agent = TD3Agent(state_dim=66, action_dim=1, max_action=MAX_ACTION_RL, device=DEVICE)
    
    from src.td3.replay_buffer import ReplayBuffer
    from src.td3.reward import compute_residual_reward
    buffer = ReplayBuffer(66, 1, max_size=80000)
    predictor.eval()
    with torch.no_grad():
        for x, y, nc, _, _ in t_ld:
            yb, h, _, _ = predictor(x.to(DEVICE))
            yb_cpu, y_cpu = yb.cpu().numpy(), y.numpy()
            s = np.hstack([h.cpu().numpy(), yb_cpu / MAX_RUL, nc.numpy()])
            err = y_cpu - yb_cpu
            a = np.clip(err, -MAX_ACTION_RL, MAX_ACTION_RL)
            r = compute_residual_reward(torch.tensor(yb_cpu), torch.tensor(a), torch.tensor(y_cpu), max_rul=MAX_RUL).numpy()
            for i in range(len(s)): buffer.add(s[i], a[i], float(r[i, 0]), s[i], True)
            
    for step in range(TD3_STEPS):
        td3_agent.train_step(buffer, batch_size=128)
        
    lqr_filter = LQRSafetyFilter(Q=1.0, R=2.0, max_correction=MAX_ACTION_RL, max_rate=MAX_ACTION_RL / 2.0)
    safety_gate = SafetyGate(max_allowed_correction=MAX_ACTION_RL)
    
    val_y_base, val_a_lqr, val_y_true = [], [], []
    with torch.no_grad():
        for x, y, nc, _, _ in v_ld:
            yb, h, _, _ = predictor(x.to(DEVICE))
            s = np.hstack([h.cpu().numpy(), yb.cpu().numpy() / MAX_RUL, nc.numpy()])
            a_td3 = td3_agent.select_action(s, noise=0.0)
            for i in range(len(yb)):
                val_y_base.append(yb[i].item())
                val_a_lqr.append(lqr_filter.filter_action(a_td3[i].item(), y_base=yb[i].item()))
                val_y_true.append(y[i].item())
    
    stats = safety_gate.calibrate(np.array(val_y_base), np.array(val_a_lqr), np.array(val_y_true))
    print(f"Gate Calibrated | Alpha: {stats['best_alpha']:.3f} | Base Val Score: {stats['validation_base_score']:.3f} -> Gated Val Score: {stats['validation_calibrated_score']:.3f}")
    
    # 6. Stage F: Final Evaluation on CLEAN TEST SET
    print("\n--- STAGE F: Test Set Evaluation ---")
    res = run_full_inference(
        predictor, td3_agent, lqr_filter, safety_gate, 
        dm.test_dataset.windows, dm.test_dataset.ruls, dm.test_dataset.cycles_norm, 
        torch.tensor(dm.test_dataset.engine_ids),
        DEVICE, corruption_mode='clean'
    )
    m = calculate_metrics(res['y_final'], res['y_true'])
    m_base = calculate_metrics(res['y_base'], res['y_true'])
    
    print(f"🏆 {dataset_name} FINAL TEST RMSE (Baseline): {m_base['RMSE']:.3f}")
    print(f"🏆 {dataset_name} FINAL TEST RMSE (Full Sys): {m['RMSE']:.3f}")
    
    final_results[dataset_name] = {
        'Baseline RMSE': m_base['RMSE'],
        'Full System RMSE': m['RMSE'],
        'Full System Score': m['CMAPSS_Score']
    }

# -------------------

print("="*60)
print(f"{'Dataset':<10} | {'Baseline RMSE':<15} | {'Full System RMSE':<18} | {'NASA Score'}")
print("-" * 60)
avg_rmse = 0
for ds in ALL_DATASETS:
    r = final_results.get(ds, {'Baseline RMSE': 0, 'Full System RMSE': 0, 'Full System Score': 0})
    avg_rmse += r['Full System RMSE']
    print(f"{ds:<10} | {r['Baseline RMSE']:<15.3f} | {r['Full System RMSE']:<18.3f} | {r['Full System Score']:.1f}")
print("="*60)
print(f"🔥 FINAL AVERAGE RMSE ACROSS ALL DATASETS: {avg_rmse / len(ALL_DATASETS):.3f} 🔥")

# -------------------

