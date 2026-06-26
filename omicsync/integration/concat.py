"""Simple concatenation strategies for multi-omics integration."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from omicsync.core.dataset import OmicsDataset
from omicsync.utils.logging import get_logger

logger = get_logger("integration.concat")


def simple_concat(
    dataset: OmicsDataset,
    modalities: Optional[Sequence[str]] = None,
    fill_missing: float = 0.0,
) -> np.ndarray:
    """Concatenate modalities into a flat numpy array.

    Parameters
    ----------
    dataset:
        An :class:`~omicsync.core.dataset.OmicsDataset`.
    modalities:
        Modality names to include.  ``None`` uses all modalities.
    fill_missing:
        Value for missing entries (default 0.0).

    Returns
    -------
    numpy.ndarray
        Shape (n_samples, total_features).
    """
    df = dataset.to_dataframe(modalities=modalities, fill_missing=fill_missing)
    logger.info(
        "simple_concat: output shape %s.", df.shape
    )
    return df.values


def weighted_concat(
    dataset: OmicsDataset,
    weights: Optional[Dict[str, float]] = None,
    fill_missing: float = 0.0,
) -> np.ndarray:
    """Concatenate modalities with per-modality feature scaling.

    Each modality's features are multiplied by the corresponding weight
    before concatenation, allowing you to up-weight or down-weight
    specific data types.

    Parameters
    ----------
    dataset:
        An :class:`~omicsync.core.dataset.OmicsDataset`.
    weights:
        Mapping from modality name to scalar weight.  Modalities not listed
        receive a weight of 1.0.
    fill_missing:
        Value for missing entries.

    Returns
    -------
    numpy.ndarray
        Shape (n_samples, total_features).
    """
    weights = weights or {}
    frames: List[pd.DataFrame] = []

    for name, mod in dataset._modalities.items():
        w = weights.get(name, 1.0)
        prefixed = mod.data.add_prefix(f"{name}__") * w
        frames.append(prefixed)

    if not frames:
        return np.empty((0, 0))

    df = frames[0].join(frames[1:], how="outer").fillna(fill_missing)
    logger.info("weighted_concat: output shape %s.", df.shape)
    return df.values


def pca_concat(
    dataset: OmicsDataset,
    n_components_per_modality: int = 50,
    fill_missing: float = 0.0,
) -> np.ndarray:
    """Reduce each modality by PCA then concatenate the scores.

    Requires ``scikit-learn``.

    Parameters
    ----------
    dataset:
        An :class:`~omicsync.core.dataset.OmicsDataset`.
    n_components_per_modality:
        Number of PCA components to retain per modality (capped at the
        modality's feature count).
    fill_missing:
        Value for missing entries before PCA.

    Returns
    -------
    numpy.ndarray
        Shape (n_samples, n_components_per_modality × n_modalities).

    Raises
    ------
    ImportError
        If ``scikit-learn`` is not installed.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for pca_concat(). "
            "Install it with: pip install scikit-learn"
        ) from exc

    all_samples = dataset.common_samples
    parts: List[np.ndarray] = []

    for name, mod in dataset._modalities.items():
        data = mod.data.reindex(all_samples).fillna(fill_missing).values.astype(float)
        n_comp = min(n_components_per_modality, data.shape[1], data.shape[0])

        imputer = SimpleImputer(strategy="mean")
        data = imputer.fit_transform(data)

        pca = PCA(n_components=n_comp, random_state=0)
        scores = pca.fit_transform(data)
        parts.append(scores)
        logger.info(
            "pca_concat: %r → %d components (%.1f%% variance).",
            name,
            n_comp,
            100.0 * pca.explained_variance_ratio_.sum(),
        )

    result = np.concatenate(parts, axis=1)
    logger.info("pca_concat: final shape %s.", result.shape)
    return result
