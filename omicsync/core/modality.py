"""OmicsModality base class and modality-specific subclasses."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from omicsync.utils.logging import get_logger
from omicsync.utils.validation import (
    validate_dataframe,
    validate_modality_type,
    check_value_range,
    validate_sample_ids,
)

logger = get_logger("core.modality")


class OmicsModality:
    """Container for a single omics modality (samples × features).

    Parameters
    ----------
    data:
        DataFrame indexed by sample IDs, columns are feature IDs.
    modality_type:
        One of ``"rna"``, ``"mutations"``, ``"methylation"``, ``"cnv"``,
        ``"protein"``.
    source:
        Data source identifier, e.g. ``"tcga"``, ``"geo"``, ``"csv"``.
    metadata:
        Arbitrary key/value metadata stored alongside the data.

    Raises
    ------
    ValueError
        If *modality_type* is invalid or the DataFrame is malformed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        modality_type: str,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        validate_modality_type(modality_type)
        validate_dataframe(data, name=f"{modality_type} data")
        validate_sample_ids(data.index.tolist())

        self._data = data.copy()
        self.modality_type = modality_type
        self.source = source
        self.metadata: Dict = metadata or {}

        check_value_range(self._data, self.modality_type)
        logger.info(
            "Loaded %s modality from %s: %d samples × %d features.",
            modality_type,
            source,
            self.n_samples,
            self.n_features,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """The underlying data matrix (samples × features)."""
        return self._data

    @property
    def n_samples(self) -> int:
        """Number of samples (rows)."""
        return self._data.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features (columns)."""
        return self._data.shape[1]

    @property
    def sample_ids(self) -> pd.Index:
        """Sample identifiers (row index)."""
        return self._data.index

    @property
    def feature_ids(self) -> pd.Index:
        """Feature identifiers (column index)."""
        return self._data.columns

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def filter_features(
        self,
        min_variance: float = 0.0,
        min_nonzero_frac: float = 0.0,
    ) -> "OmicsModality":
        """Remove low-information features in-place and return *self*.

        Parameters
        ----------
        min_variance:
            Drop features whose variance is below this threshold.
        min_nonzero_frac:
            Drop features where the fraction of non-zero values is below this.

        Returns
        -------
        OmicsModality
            *self*, for method chaining.
        """
        mask = np.ones(self.n_features, dtype=bool)

        if min_variance > 0.0:
            variances = self._data.var(axis=0, skipna=True)
            mask &= variances.values >= min_variance

        if min_nonzero_frac > 0.0:
            nonzero_frac = (self._data != 0).mean(axis=0)
            mask &= nonzero_frac.values >= min_nonzero_frac

        n_before = self.n_features
        self._data = self._data.loc[:, mask]
        n_after = self.n_features
        logger.info(
            "%s: filtered features %d → %d (kept %.1f%%).",
            self.modality_type,
            n_before,
            n_after,
            100.0 * n_after / max(n_before, 1),
        )
        return self

    def filter_samples(self, sample_ids: Sequence) -> "OmicsModality":
        """Keep only the specified samples in-place and return *self*.

        Parameters
        ----------
        sample_ids:
            Iterable of sample IDs to retain.

        Returns
        -------
        OmicsModality
            *self*, for method chaining.

        Raises
        ------
        ValueError
            If none of the provided IDs are present in this modality.
        """
        requested = pd.Index(sample_ids)
        common = self._data.index.intersection(requested)
        if len(common) == 0:
            raise ValueError(
                f"None of the {len(requested)} requested sample IDs were found "
                f"in {self.modality_type} modality."
            )
        n_before = self.n_samples
        self._data = self._data.loc[common]
        logger.info(
            "%s: filtered samples %d → %d.",
            self.modality_type,
            n_before,
            self.n_samples,
        )
        return self

    def describe(self) -> Dict:
        """Return a summary dictionary of this modality.

        Returns
        -------
        dict
            Keys: ``modality_type``, ``source``, ``n_samples``,
            ``n_features``, ``value_min``, ``value_max``, ``value_mean``,
            ``missing_frac``.
        """
        vals = self._data.values.ravel().astype(float)
        finite = vals[np.isfinite(vals)]
        return {
            "modality_type": self.modality_type,
            "source": self.source,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "value_min": float(finite.min()) if len(finite) else float("nan"),
            "value_max": float(finite.max()) if len(finite) else float("nan"),
            "value_mean": float(finite.mean()) if len(finite) else float("nan"),
            "missing_frac": float(np.isnan(vals).mean()),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"modality_type={self.modality_type!r}, "
            f"shape=({self.n_samples}, {self.n_features}), "
            f"source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Modality-specific subclasses
# ---------------------------------------------------------------------------


class RNAModality(OmicsModality):
    """Modality subclass for RNA expression data.

    Validates that all values are non-negative.

    Parameters
    ----------
    data:
        DataFrame of RNA expression values (samples × genes).
    source:
        Data source identifier.
    metadata:
        Optional metadata dict.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(data, modality_type="rna", source=source, metadata=metadata)
        finite_vals = self._data.values[np.isfinite(self._data.values)]
        if len(finite_vals) > 0 and finite_vals.min() < 0:
            raise ValueError(
                "RNAModality: data contains negative values. "
                "RNA expression values must be non-negative (counts or TPM)."
            )


class MutationModality(OmicsModality):
    """Modality subclass for somatic mutation data.

    Parameters
    ----------
    data:
        Binary or count-based mutation matrix (samples × genes).
    source:
        Data source identifier.
    metadata:
        Optional metadata dict.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            data, modality_type="mutations", source=source, metadata=metadata
        )


class MethylationModality(OmicsModality):
    """Modality subclass for DNA methylation data.

    Validates that beta values lie in [0, 1] if the data appears to be
    beta values (i.e. all finite values are in [-6, 6] is permitted for
    M-values, but pure beta must be in [0, 1]).

    Parameters
    ----------
    data:
        Methylation matrix (samples × CpG sites).
    source:
        Data source identifier.
    metadata:
        Optional metadata dict.
    value_type:
        ``"beta"`` (default) or ``"mvalue"``.  Beta values are validated
        to lie in [0, 1]; M-values have no range constraint.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
        value_type: str = "beta",
    ) -> None:
        if value_type not in ("beta", "mvalue"):
            raise ValueError(
                f"value_type must be 'beta' or 'mvalue', got {value_type!r}."
            )
        self.value_type = value_type
        super().__init__(
            data, modality_type="methylation", source=source, metadata=metadata
        )
        if value_type == "beta":
            finite_vals = self._data.values[np.isfinite(self._data.values)]
            if len(finite_vals) > 0:
                if finite_vals.min() < -0.01 or finite_vals.max() > 1.01:
                    raise ValueError(
                        "MethylationModality (beta): values must be in [0, 1]. "
                        f"Got min={finite_vals.min():.4f}, max={finite_vals.max():.4f}. "
                        "If these are M-values, set value_type='mvalue'."
                    )


class CNVModality(OmicsModality):
    """Modality subclass for copy-number variation data.

    Parameters
    ----------
    data:
        CNV matrix (samples × genes/segments).
    source:
        Data source identifier.
    metadata:
        Optional metadata dict.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(data, modality_type="cnv", source=source, metadata=metadata)


class ProteinModality(OmicsModality):
    """Modality subclass for protein abundance data.

    Parameters
    ----------
    data:
        Protein abundance matrix (samples × proteins).
    source:
        Data source identifier.
    metadata:
        Optional metadata dict.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            data, modality_type="protein", source=source, metadata=metadata
        )


# Convenience mapping from modality_type string to subclass
MODALITY_CLASSES: Dict[str, type] = {
    "rna": RNAModality,
    "mutations": MutationModality,
    "methylation": MethylationModality,
    "cnv": CNVModality,
    "protein": ProteinModality,
}


def make_modality(
    data: pd.DataFrame,
    modality_type: str,
    source: str = "unknown",
    metadata: Optional[Dict] = None,
    **kwargs,
) -> OmicsModality:
    """Instantiate the appropriate :class:`OmicsModality` subclass.

    Parameters
    ----------
    data:
        Feature matrix (samples × features).
    modality_type:
        One of the recognised modality types.
    source:
        Data source identifier.
    metadata:
        Optional metadata dict.
    **kwargs:
        Passed through to the subclass constructor.

    Returns
    -------
    OmicsModality
        The appropriate subclass instance.
    """
    validate_modality_type(modality_type)
    cls = MODALITY_CLASSES[modality_type]
    return cls(data, source=source, metadata=metadata, **kwargs)
