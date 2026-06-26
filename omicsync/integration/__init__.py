"""Integration methods for multi-omics data fusion."""

from omicsync.integration.concat import simple_concat, weighted_concat, pca_concat
from omicsync.integration.sklearn_compat import OmicsSyncTransformer

__all__ = [
    "simple_concat",
    "weighted_concat",
    "pca_concat",
    "OmicsSyncTransformer",
]
