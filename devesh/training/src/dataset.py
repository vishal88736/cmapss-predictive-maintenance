"""
CMAPSSDataModule: sliding-window dataset builder for NASA C-MAPSS.

Provides engine-disjoint 80/20 train/validation split.
Test data is loaded separately and never touches training/validation logic.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
from .preprocessing import RegimeAwarePreprocessor


class CMAPSSWindowDataset(Dataset):
    """
    Torch Dataset of sliding windows [N, window_length, 14] with RUL labels.

    Each item returns:
      x          : [window_length, 14]  normalised sensor window (float32)
      y          : [1]                  RUL label (float32)
      cycle_norm : [1]                  normalised cycle position ∈ [0, 1]
      engine_id  : scalar int           engine unit ID
      cycle      : scalar int           absolute cycle number
    """

    def __init__(
        self,
        windows: torch.Tensor,       # [N, W, 14]
        ruls: torch.Tensor,          # [N]
        engine_ids: torch.Tensor,    # [N]
        cycle_norms: torch.Tensor,   # [N]
    ):
        assert windows.shape[-1] == 14, f"Expected 14 features, got {windows.shape[-1]}"
        assert windows.shape[1] >= 1, "Window length must be ≥ 1"
        self.windows = windows
        self.ruls = ruls
        self.engine_ids = engine_ids
        self.cycles_norm = cycle_norms

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            self.windows[idx],                     # [W, 14]
            self.ruls[idx : idx + 1].float(),      # [1]
            self.cycles_norm[idx : idx + 1].float(), # [1]
            self.engine_ids[idx],
            idx,
        )


def _build_windows(
    features: np.ndarray,   # [N_rows, 14]
    ruls: np.ndarray,       # [N_rows]
    unit_ids: np.ndarray,   # [N_rows]
    window_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sliding windows per engine.

    Returns (windows, labels, engine_ids, cycle_norms).
    """
    win_list, rul_list, eid_list, cnorm_list = [], [], [], []

    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        feat = features[mask]    # [T, 14]
        rul = ruls[mask]         # [T]
        T = len(feat)

        if T < window_length:
            # Pad with first row to reach window_length
            pad = np.tile(feat[:1], (window_length - T, 1))
            feat = np.concatenate([pad, feat], axis=0)
            rul = np.concatenate([rul[:1].repeat(window_length - T), rul], axis=0)
            T = window_length

        for start in range(T - window_length + 1):
            end = start + window_length
            win_list.append(feat[start:end])
            rul_list.append(rul[end - 1])
            eid_list.append(uid)
            cnorm_list.append((end - 1) / max(T - 1, 1))

    windows = np.stack(win_list, axis=0).astype(np.float32)     # [N, W, 14]
    labels = np.array(rul_list, dtype=np.float32)                # [N]
    engine_ids = np.array(eid_list, dtype=np.int32)              # [N]
    cycle_norms = np.array(cnorm_list, dtype=np.float32)         # [N]

    assert windows.shape[-1] == 14, f"Window feature count is {windows.shape[-1]}, expected 14"
    return windows, labels, engine_ids, cycle_norms


class CMAPSSDataModule:
    """
    Orchestrates: load → preprocess → split → windowing → DataLoaders.

    Engine-disjoint 80/20 split: 80% of training engines → train set,
    remaining 20% → validation set. Random split is seeded for reproducibility.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "FD001",
        window_length: int = 30,
        n_regimes: int = 6,
        seed: int = 42,
        max_rul: Optional[float] = 125.0,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.window_length = window_length
        self.seed = seed
        self.max_rul = max_rul

        self.preprocessor = RegimeAwarePreprocessor(
            n_regimes=n_regimes, max_rul=max_rul, seed=seed
        )

        self.train_dataset: Optional[CMAPSSWindowDataset] = None
        self.val_dataset: Optional[CMAPSSWindowDataset] = None
        self.test_dataset: Optional[CMAPSSWindowDataset] = None

        self.train_engine_ids: List[int] = []
        self.val_engine_ids: List[int] = []
        self.test_engine_ids: List[int] = []

    def load_and_preprocess(self, verbose: bool = False) -> None:
        train_path = os.path.join(self.data_dir, f"train_{self.dataset_name}.txt")
        test_path = os.path.join(self.data_dir, f"test_{self.dataset_name}.txt")
        rul_path = os.path.join(self.data_dir, f"RUL_{self.dataset_name}.txt")

        if verbose:
            print(f"\n  [DataModule] Loading {self.dataset_name}")
            print(f"    train: {train_path}")
            print(f"    test : {test_path}")
            print(f"    rul  : {rul_path}")

        # ── Fit on training data ──────────────────────────────────────────────
        feat_all, rul_all, _, uid_all = self.preprocessor.fit_transform(
            train_path, verbose=verbose
        )
        assert feat_all.shape[1] == 14

        # ── Engine-disjoint 80/20 split ───────────────────────────────────────
        all_engines = sorted(np.unique(uid_all).tolist())
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_engines := list(all_engines))
        n_train = max(1, int(np.round(0.8 * len(all_engines))))
        train_eids = all_engines[:n_train]
        val_eids = all_engines[n_train:]
        self.train_engine_ids = sorted(train_eids)
        self.val_engine_ids = sorted(val_eids)

        # Verify disjoint
        assert len(set(self.train_engine_ids) & set(self.val_engine_ids)) == 0, \
            "Engine overlap between train and val!"

        # ── Build windows ─────────────────────────────────────────────────────
        tr_mask = np.isin(uid_all, train_eids)
        val_mask = np.isin(uid_all, val_eids)

        tr_win, tr_rul, tr_eid, tr_cn = _build_windows(
            feat_all[tr_mask], rul_all[tr_mask], uid_all[tr_mask], self.window_length
        )
        v_win, v_rul, v_eid, v_cn = _build_windows(
            feat_all[val_mask], rul_all[val_mask], uid_all[val_mask], self.window_length
        )

        self.train_dataset = CMAPSSWindowDataset(
            torch.from_numpy(tr_win),
            torch.from_numpy(tr_rul),
            torch.from_numpy(tr_eid.astype(np.int64)),
            torch.from_numpy(tr_cn),
        )
        self.val_dataset = CMAPSSWindowDataset(
            torch.from_numpy(v_win),
            torch.from_numpy(v_rul),
            torch.from_numpy(v_eid.astype(np.int64)),
            torch.from_numpy(v_cn),
        )

        # ── Test set (from test file + RUL file) ──────────────────────────────
        t_feat, t_rul, _, t_uid = self.preprocessor.transform(
            test_path, test_rul_path=rul_path, verbose=verbose
        )
        assert t_feat.shape[1] == 14

        t_win, t_rul_w, t_eid, t_cn = _build_windows(
            t_feat, t_rul, t_uid, self.window_length
        )
        self.test_dataset = CMAPSSWindowDataset(
            torch.from_numpy(t_win),
            torch.from_numpy(t_rul_w),
            torch.from_numpy(t_eid.astype(np.int64)),
            torch.from_numpy(t_cn),
        )
        self.test_engine_ids = sorted(np.unique(t_uid).tolist())

        if verbose:
            print(f"    Train windows: {tr_win.shape} | Val windows: {v_win.shape} | Test windows: {t_win.shape}")
            print(f"    Train engines: {len(self.train_engine_ids)} | Val engines: {len(self.val_engine_ids)} | Test engines: {len(self.test_engine_ids)}")

    def get_loaders(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_weights: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if use_weights:
            from torch.utils.data import WeightedRandomSampler
            weights = torch.where(self.train_dataset.ruls < 70, 2.0, 1.0).squeeze()
            sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
