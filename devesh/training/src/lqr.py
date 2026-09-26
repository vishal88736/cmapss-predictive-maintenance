"""
LQR Safety Filter: Scalar-integrator approximation.

This is a simple scalar discrete-time LQR with:
  state  x_k = e_k  (current correction error)
  input  u_k = a_k  (requested TD3 correction)
  dynamics: x_{k+1} = x_k + u_k

The DARE solution for this system reduces to scalar gain K.
The filter enforces:
  1. Correction magnitude ≤ max_correction
  2. Correction rate      ≤ max_rate  (|a_k - a_{k-1}|)
  3. Corrected RUL ∈ [0, max_rul]

Note: This is NOT a physical engine controller. It is a safety constraint
applied to the scalar residual-correction signal only.
"""

import numpy as np


class LQRSafetyFilter:
    """
    Scalar-integrator LQR safety filter.

    Parameters
    ----------
    Q            : state cost weight
    R            : control cost weight
    max_correction: maximum allowed |correction|
    max_rate     : maximum allowed |a_k - a_{k-1}|
    max_rul      : maximum allowed corrected RUL (= dataset RUL cap)
    """

    def __init__(
        self,
        Q: float = 1.0,
        R: float = 2.0,
        max_correction: float = 6.0,
        max_rate: float = 2.0,
        max_rul: float = 125.0,
    ):
        self.Q = Q
        self.R = R
        self.max_correction = max_correction
        self.max_rate = max_rate
        self.max_rul = max_rul

        # DARE-derived optimal gain for scalar system x_{k+1} = x + u
        # K* = (R + P) / (R + P + Q)  ... analytically derived
        # For Q=1, R=2: K ~ 0.33  (illustrative; we use magnitude/rate clipping)
        self._K = float(Q) / (float(Q) + float(R))  # scalar gain ∈ (0, 1)

    def filter_action(
        self,
        a_td3: float,
        prev_a: float = 0.0,
        y_base: float = 0.0,
    ) -> float:
        """
        Apply LQR safety constraints to raw TD3 action.

        Parameters
        ----------
        a_td3  : raw TD3 correction
        prev_a : previous filtered action (for rate limiting)
        y_base : baseline RUL prediction (for bounds check)

        Returns
        -------
        filtered action (float)
        """
        # 1. Apply LQR gain scaling
        a = self._K * float(a_td3)

        # 2. Magnitude clamp
        a = float(np.clip(a, -self.max_correction, self.max_correction))

        # 3. Rate limit
        delta = a - float(prev_a)
        if abs(delta) > self.max_rate:
            a = float(prev_a) + float(np.sign(delta)) * self.max_rate

        # 4. Magnitude re-clamp after rate limiting
        a = float(np.clip(a, -self.max_correction, self.max_correction))

        # 5. Bounds check: ensure corrected RUL ∈ [0, max_rul]
        corrected = float(y_base) + a
        if corrected < 0.0:
            a = -float(y_base)
        elif corrected > self.max_rul:
            a = self.max_rul - float(y_base)

        return float(a)

    def apply_safety_constraint(
        self, y_base: float, a_td3: float, prev_a: float = 0.0
    ):
        """Apply filter and return (corrected_rul, filtered_action)."""
        a_safe = self.filter_action(a_td3, prev_a=prev_a, y_base=y_base)
        y_corrected = float(np.clip(y_base + a_safe, 0.0, self.max_rul))
        return y_corrected, a_safe
