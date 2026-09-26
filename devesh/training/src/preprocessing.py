"""
Regime-Aware Preprocessing for NASA C-MAPSS.

Canonical 14 informative sensors (operating settings excluded from features).
Operating settings are used ONLY for K-Means regime identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Column layout of raw C-MAPSS text files ─────────────────────────────────
# Col 1-2: unit_id, cycle
# Col 3-5: setting_1 (altitude), setting_2 (Mach), setting_3 (TRA)
# Col 6-26: sensor_1 … sensor_21
RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# ─── Canonical 14 informative sensors ────────────────────────────────────────
# Selected by literature consensus (Saxena et al. 2008) + near-zero-variance drop.
# Operating settings are EXCLUDED from this list.
CANONICAL_14_SENSORS: List[str] = [
    "sensor_2",   # Total temperature at LPC outlet (T24)
    "sensor_3",   # Total temperature at HPC outlet (T30)
    "sensor_4",   # Total temperature at LPT outlet (T50)
    "sensor_7",   # Total pressure at HPC outlet (P30)
    "sensor_8",   # Physical fan speed (Nf)
    "sensor_9",   # Physical core speed (Nc)
    "sensor_11",  # Static pressure at HPC outlet (Ps30)
    "sensor_12",  # Ratio of fuel flow to Ps30 (phi)
    "sensor_13",  # Corrected fan speed (NRf)
    "sensor_14",  # Corrected core speed (NRc)
    "sensor_15",  # Bypass ratio (BPR)
    "sensor_17",  # Bleed enthalpy (htBleed)
    "sensor_20",  # High-pressure turbine coolant bleed (W31)
    "sensor_21",  # Low-pressure turbine coolant bleed (W32)
]

CANONICAL_14_DESCRIPTIONS: List[str] = [
    "Total temperature at LPC outlet (T24)",
    "Total temperature at HPC outlet (T30)",
    "Total temperature at LPT outlet (T50)",
    "Total pressure at HPC outlet (P30)",
    "Physical fan speed (Nf)",
    "Physical core speed (Nc)",
    "Static pressure at HPC outlet (Ps30)",
    "Ratio of fuel flow to Ps30 (phi)",
    "Corrected fan speed (NRf)",
    "Corrected core speed (NRc)",
    "Bypass ratio (BPR)",
    "Bleed enthalpy (htBleed)",
    "HP turbine coolant bleed (W31)",
    "LP turbine coolant bleed (W32)",
]

assert len(CANONICAL_14_SENSORS) == 14, "Must be exactly 14 canonical sensors"
assert len(CANONICAL_14_DESCRIPTIONS) == 14


class RegimeAwarePreprocessor:
    """
    Full preprocessing pipeline:
      1. Load raw C-MAPSS text files
      2. Select canonical 14 sensors
      3. K-Means (k=6) on operating settings → regime labels
      4. Regime-specific Z-score normalisation (fit on TRAIN only)
      5. RUL computation and optional clipping at 125
    """

    def __init__(
        self,
        n_regimes: int = 6,
        max_rul: Optional[float] = 125.0,
        seed: int = 42,
    ):
        self.n_regimes = n_regimes
        self.max_rul = max_rul
        self.seed = seed

        # Fitted artefacts (populated by fit_transform)
        self.kmeans: Optional[KMeans] = None
        self.regime_means: Dict[int, np.ndarray] = {}
        self.regime_stds: Dict[int, np.ndarray] = {}
        self.global_mean: Optional[np.ndarray] = None
        self.global_std: Optional[np.ndarray] = None
        self.train_engine_ids: List[int] = []

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_raw(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
        # Drop any trailing NaN columns
        df = df.dropna(axis=1, how="all")
        return df

    @staticmethod
    def _compute_rul_train(df: pd.DataFrame, max_rul: Optional[float]) -> pd.Series:
        """Compute RUL for TRAINING data (each engine ends at failure)."""
        max_cycles = df.groupby("unit_id")["cycle"].transform("max")
        rul = max_cycles - df["cycle"]
        if max_rul is not None:
            rul = rul.clip(upper=max_rul)
        return rul.astype(np.float32)

    def _assign_regimes(self, settings: np.ndarray) -> np.ndarray:
        """Assign operating-regime labels using the fitted KMeans."""
        assert self.kmeans is not None, "Call fit_transform before _assign_regimes"
        return self.kmeans.predict(settings)

    def _normalise(self, features: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """
        Apply regime-specific Z-score normalisation row-by-row.
        Falls back to global stats for any unseen regime.
        """
        out = np.empty_like(features, dtype=np.float32)
        for k in range(self.n_regimes):
            idx = regimes == k
            if idx.any():
                mu = self.regime_means.get(k, self.global_mean)
                sd = self.regime_stds.get(k, self.global_std)
                out[idx] = (features[idx] - mu) / sd
        # Handle any regime not seen during training
        unknown = ~np.isin(regimes, list(self.regime_means.keys()))
        if unknown.any():
            out[unknown] = (features[unknown] - self.global_mean) / self.global_std
        return out

    # ── public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self, train_path: str, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load + fit on training data. Returns:
          features [N_rows, 14]  (float32, normalised)
          ruls     [N_rows]      (float32)
          regimes  [N_rows]      (int)
          unit_ids [N_rows]      (int)
        """
        df = self._load_raw(train_path)
        self.train_engine_ids = sorted(df["unit_id"].unique().tolist())

        # 1. Extract canonical 14 features
        features_raw = df[CANONICAL_14_SENSORS].values.astype(np.float32)

        # 2. Fit K-Means on operating settings (NOT on sensor data)
        settings = df[["setting_1", "setting_2", "setting_3"]].values
        self.kmeans = KMeans(
            n_clusters=self.n_regimes, random_state=self.seed, n_init=10
        )
        regime_labels = self.kmeans.fit_predict(settings).astype(int)

        # 3. Fit regime-specific statistics (on raw features)
        self.global_mean = features_raw.mean(0)
        raw_std = features_raw.std(0)
        raw_std[raw_std < 1e-8] = 1.0
        self.global_std = raw_std

        for k in range(self.n_regimes):
            idx = regime_labels == k
            if idx.sum() < 2:
                self.regime_means[k] = self.global_mean
                self.regime_stds[k] = self.global_std
            else:
                mu = features_raw[idx].mean(0)
                sd = features_raw[idx].std(0)
                sd[sd < 1e-8] = 1.0
                self.regime_means[k] = mu
                self.regime_stds[k] = sd

        # 4. Normalise
        features_norm = self._normalise(features_raw, regime_labels)

        # 5. Compute RUL
        ruls = self._compute_rul_train(df, self.max_rul).values

        unit_ids = df["unit_id"].values.astype(int)

        assert features_norm.shape[1] == 14, f"Expected 14 features, got {features_norm.shape[1]}"

        if verbose:
            print(f"  [Preprocessor] Train rows: {len(df)} | Engines: {len(self.train_engine_ids)}")
            print(f"  [Preprocessor] Regimes found: {np.unique(regime_labels)}")
            print(f"  [Preprocessor] RUL: min={ruls.min():.1f} max={ruls.max():.1f} mean={ruls.mean():.1f}")

        return features_norm, ruls, regime_labels, unit_ids

    def transform(
        self, path: str, test_rul_path: Optional[str] = None, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform test/val data using fitted statistics.

        If test_rul_path is given, per-engine last-cycle RUL labels from the
        RUL file are used (C-MAPSS test convention).
        Otherwise the sequence is treated as a partial training sequence and
        RUL is computed from max_cycle - cycle (val set usage).
        """
        assert self.kmeans is not None, "Must call fit_transform first"

        df = self._load_raw(path)
        features_raw = df[CANONICAL_14_SENSORS].values.astype(np.float32)
        settings = df[["setting_1", "setting_2", "setting_3"]].values
        regime_labels = self._assign_regimes(settings).astype(int)
        features_norm = self._normalise(features_raw, regime_labels)

        if test_rul_path is not None:
            # C-MAPSS test: last-cycle RUL per engine supplied in separate file
            rul_file = pd.read_csv(test_rul_path, header=None, names=["rul"])
            rul_map = {i + 1: v for i, v in enumerate(rul_file["rul"].values)}
            # Build per-row RUL: only the last cycle of each engine gets real RUL;
            # For windowing we need per-row labels → compute backwards
            ruls_list = []
            for _, grp in df.groupby("unit_id", sort=True):
                uid = int(grp["unit_id"].iloc[0])
                base_rul = float(rul_map.get(uid, 0))
                n_cycles = len(grp)
                # Last row = base_rul, second-to-last = base_rul+1, etc.
                r = base_rul + np.arange(n_cycles - 1, -1, -1, dtype=np.float32)
                ruls_list.append(r)
            ruls = np.concatenate(ruls_list, axis=0)
        else:
            ruls = self._compute_rul_train(df, self.max_rul).values

        unit_ids = df["unit_id"].values.astype(int)

        assert features_norm.shape[1] == 14, f"Expected 14 features, got {features_norm.shape[1]}"

        if verbose:
            print(f"  [Preprocessor] Test/Val rows: {len(df)} | Engines: {df['unit_id'].nunique()}")
            print(f"  [Preprocessor] RUL: min={ruls.min():.1f} max={ruls.max():.1f} mean={ruls.mean():.1f}")

        return features_norm, ruls, regime_labels, unit_ids
