"""scikit-learn Pipeline compatibility for OmicsDataset."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from omicsync.core.dataset import OmicsDataset
from omicsync.utils.logging import get_logger

logger = get_logger("integration.sklearn_compat")


class OmicsSyncTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer for :class:`~omicsync.core.dataset.OmicsDataset`.

    Aligns samples, normalises each modality, and returns a concatenated
    numpy array (or DataFrame) suitable for downstream estimators.

    Parameters
    ----------
    align: bool
        Whether to align samples across modalities during fit (default ``True``).
    normalize: bool
        Whether to apply per-modality normalisation during fit (default ``True``).
    modalities: list[str] or None
        Modality names to include.  ``None`` uses all modalities.
    fill_missing: float
        Value used for missing entries in the output (default ``0.0``).

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from omicsync.integration.sklearn_compat import OmicsSyncTransformer
    >>> pipe = Pipeline([
    ...     ('omicsync', OmicsSyncTransformer()),
    ...     ('classifier', SomeClassifier()),
    ... ])
    """

    def __init__(
        self,
        align: bool = True,
        normalize: bool = True,
        modalities: Optional[Sequence[str]] = None,
        fill_missing: float = 0.0,
    ) -> None:
        self.align = align
        self.normalize = normalize
        self.modalities = modalities
        self.fill_missing = fill_missing

        self._feature_names: Optional[List[str]] = None
        self._output_transform: Optional[str] = None

    def fit(self, X: OmicsDataset, y=None) -> "OmicsSyncTransformer":
        """Learn normalisation parameters from *X*.

        Parameters
        ----------
        X:
            An :class:`~omicsync.core.dataset.OmicsDataset`.
        y:
            Ignored (present for sklearn API compatibility).

        Returns
        -------
        OmicsSyncTransformer
            *self*.
        """
        if not isinstance(X, OmicsDataset):
            raise TypeError(
                f"OmicsSyncTransformer expects an OmicsDataset, got {type(X).__name__}."
            )
        if self.align:
            X.align_samples(strategy="intersection")
        if self.normalize:
            X.normalize(per_modality=True)

        df = X.to_dataframe(
            modalities=list(self.modalities) if self.modalities else None,
            fill_missing=self.fill_missing,
        )
        self._feature_names = df.columns.tolist()
        logger.info(
            "OmicsSyncTransformer.fit: %d features learned.", len(self._feature_names)
        )
        return self

    def transform(self, X: OmicsDataset, y=None) -> np.ndarray:
        """Apply learned normalisation and return a numpy array.

        Parameters
        ----------
        X:
            An :class:`~omicsync.core.dataset.OmicsDataset`.
        y:
            Ignored.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Shape (n_samples, n_features).  Returns a DataFrame if
            ``set_output(transform='pandas')`` was called.
        """
        if self._feature_names is None:
            raise RuntimeError("fit() must be called before transform().")
        if not isinstance(X, OmicsDataset):
            raise TypeError(
                f"OmicsSyncTransformer expects an OmicsDataset, got {type(X).__name__}."
            )

        df = X.to_dataframe(
            modalities=list(self.modalities) if self.modalities else None,
            fill_missing=self.fill_missing,
        )
        # Align columns to fitted feature names
        df = df.reindex(columns=self._feature_names, fill_value=self.fill_missing)

        if self._output_transform == "pandas":
            return df
        return df.values

    def fit_transform(self, X: OmicsDataset, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X:
            An :class:`~omicsync.core.dataset.OmicsDataset`.
        y:
            Ignored.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
        """
        return self.fit(X, y).transform(X, y)

    def get_feature_names_out(
        self, input_features=None
    ) -> np.ndarray:
        """Return feature names with modality prefix.

        Returns
        -------
        numpy.ndarray of str
            E.g. ``["rna__EGFR", "mut__TP53", ...]``.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self._feature_names is None:
            raise RuntimeError("fit() must be called before get_feature_names_out().")
        return np.array(self._feature_names, dtype=object)

    def set_output(self, *, transform: Optional[str] = None) -> "OmicsSyncTransformer":
        """Set the output format for :meth:`transform`.

        Parameters
        ----------
        transform:
            ``"pandas"`` to return a DataFrame; ``None`` for numpy array.

        Returns
        -------
        OmicsSyncTransformer
            *self*.
        """
        if transform not in (None, "pandas"):
            raise ValueError(f"Unknown transform format {transform!r}. Valid: None, 'pandas'.")
        self._output_transform = transform
        return self
