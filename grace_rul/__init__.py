"""Shared training and inference implementation for the finalized GRACE-RUL design."""

from .config import DEFAULT_CONFIG, FORMAT_VERSION
__all__ = ["DEFAULT_CONFIG", "FORMAT_VERSION", "GraceRUL", "TemporalCritic"]


def __getattr__(name):
    # Data-independent sensor simulation/configuration does not need PyTorch.
    if name in {"GraceRUL", "TemporalCritic"}:
        from .model import GraceRUL, TemporalCritic
        return {"GraceRUL": GraceRUL, "TemporalCritic": TemporalCritic}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
