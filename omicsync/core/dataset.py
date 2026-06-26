"""OmicsDataset: the main user-facing multi-omics container."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from omicsync.core.modality import OmicsModality
from omicsync.core.sample_index import SampleIndex
from omicsync.utils.logging import get_logger

logger = get_logger("core.dataset")


class OmicsDataset:
    """Multi-omics container holding one or more :class:`OmicsModality` objects.

    Parameters
    ----------
    modalities:
        Mapping from modality name (e.g. ``"rna"``) to :class:`OmicsModality`.
    study_id:
        Study identifier, e.g. ``"TCGA-BRCA"``.
    metadata:
        Arbitrary dataset-level metadata.

    Raises
    ------
    TypeError
        If *modalities* values are not :class:`OmicsModality` instances.
    """

    def __init__(
        self,
        modalities: Dict[str, OmicsModality],
        study_id: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> None:
        for name, mod in modalities.items():
            if not isinstance(mod, OmicsModality):
                raise TypeError(
                    f"Expected OmicsModality for modality {name!r}, "
                    f"got {type(mod).__name__}."
                )
        self._modalities: Dict[str, OmicsModality] = dict(modalities)
        self.study_id = study_id
        self.metadata: Dict = metadata or {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def modality_names(self) -> List[str]:
        """Names of loaded modalities."""
        return list(self._modalities.keys())

    @property
    def common_samples(self) -> pd.Index:
        """Sample IDs present in *all* modalities."""
        if not self._modalities:
            return pd.Index([])
        arrays = [mod.sample_ids for mod in self._modalities.values()]
        return SampleIndex.align(arrays, strategy="intersection")

    @property
    def sample_coverage(self) -> pd.DataFrame:
        """Boolean DataFrame: rows = all samples, columns = modalities.

        ``True`` indicates the sample is present in that modality.
        """
        si = SampleIndex()
        return si.summarise(
            {name: mod.sample_ids for name, mod in self._modalities.items()}
        )

    @property
    def n_complete_cases(self) -> int:
        """Number of samples present in every modality."""
        return len(self.common_samples)

    # ------------------------------------------------------------------
    # Mutation methods (all return self for chaining)
    # ------------------------------------------------------------------

    def align_samples(
        self,
        strategy: str = "intersection",
        fill_value: float = np.nan,
    ) -> "OmicsDataset":
        """Retain only samples that are present according to *strategy*.

        Parameters
        ----------
        strategy:
            ``"intersection"`` (default) — keep only samples present in all
            modalities.  ``"union"`` — keep all samples; modalities that do
            not have a sample will have ``fill_value`` for that row.
        fill_value:
            Value used to fill missing samples when ``strategy="union"``.

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.

        Raises
        ------
        ValueError
            If *strategy* is not recognised.
        """
        if strategy not in ("intersection", "union"):
            raise ValueError(
                f"Unknown alignment strategy {strategy!r}. "
                "Valid: 'intersection', 'union'."
            )
        arrays = [mod.sample_ids for mod in self._modalities.values()]
        aligned = SampleIndex.align(arrays, strategy=strategy)

        for name, mod in self._modalities.items():
            if strategy == "intersection":
                mod.filter_samples(aligned)
            else:
                missing = aligned.difference(mod.sample_ids)
                if len(missing) > 0:
                    filler = pd.DataFrame(
                        fill_value,
                        index=missing,
                        columns=mod.feature_ids,
                    )
                    new_data = pd.concat([mod.data, filler]).loc[aligned]
                    mod._data = new_data

        logger.info(
            "align_samples (%s): %d samples across %d modalities.",
            strategy,
            len(aligned),
            len(self._modalities),
        )
        return self

    def normalize(self, per_modality: bool = True) -> "OmicsDataset":
        """Apply default normalisation to each modality in-place.

        Normalisation applied per modality type:

        * **rna**: :func:`~omicsync.normalisation.rna.detect_and_normalise`
        * **methylation**: :func:`~omicsync.normalisation.methylation.detect_and_normalise`
        * **cnv**: log2-ratio relative to diploid, clipped to [-2, 2]
        * **mutations**: binarise at 0
        * **protein**: z-score per feature

        Parameters
        ----------
        per_modality:
            If ``False``, skip normalisation (no-op, for API compatibility).

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.
        """
        if not per_modality:
            return self

        from omicsync.normalisation import rna as rna_norm
        from omicsync.normalisation import methylation as meth_norm
        from omicsync.normalisation import cnv as cnv_norm
        from omicsync.normalisation import mutations as mut_norm
        from omicsync.normalisation import protein as prot_norm

        _dispatch = {
            "rna": rna_norm.detect_and_normalise,
            "methylation": meth_norm.detect_and_normalise,
            "cnv": lambda df: cnv_norm.log2_ratio(cnv_norm.centre_diploid(df)).clip(-2, 2),
            "mutations": lambda df: mut_norm.binarise(df, threshold=0),
            "protein": prot_norm.z_score,
        }

        for name, mod in self._modalities.items():
            fn = _dispatch.get(mod.modality_type)
            if fn is not None:
                logger.info("Normalising modality %r (%s).", name, mod.modality_type)
                mod._data = fn(mod.data)

        return self

    def filter_features(
        self,
        min_variance: float = 0.0,
        min_nonzero_frac: float = 0.0,
    ) -> "OmicsDataset":
        """Apply feature filtering to all modalities.

        Parameters
        ----------
        min_variance:
            Minimum variance for a feature to be kept.
        min_nonzero_frac:
            Minimum fraction of non-zero values for a feature to be kept.

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.
        """
        for mod in self._modalities.values():
            mod.filter_features(
                min_variance=min_variance,
                min_nonzero_frac=min_nonzero_frac,
            )
        return self

    def drop_modality(self, name: str) -> "OmicsDataset":
        """Remove a modality by name.

        Parameters
        ----------
        name:
            Modality name to remove.

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.

        Raises
        ------
        KeyError
            If *name* is not in the dataset.
        """
        if name not in self._modalities:
            raise KeyError(
                f"Modality {name!r} not found. "
                f"Available: {self.modality_names}."
            )
        del self._modalities[name]
        logger.info("Dropped modality %r.", name)
        return self

    def add_modality(self, name: str, modality: OmicsModality) -> "OmicsDataset":
        """Add a new modality.

        Parameters
        ----------
        name:
            Name for the new modality.
        modality:
            :class:`OmicsModality` instance to add.

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.

        Raises
        ------
        TypeError
            If *modality* is not an :class:`OmicsModality`.
        ValueError
            If *name* is already present.
        """
        if not isinstance(modality, OmicsModality):
            raise TypeError(
                f"Expected OmicsModality, got {type(modality).__name__}."
            )
        if name in self._modalities:
            raise ValueError(
                f"Modality {name!r} already exists. "
                "Use drop_modality() first to replace it."
            )
        self._modalities[name] = modality
        logger.info("Added modality %r (%s).", name, modality.modality_type)
        return self

    def subset_samples(self, sample_ids: Sequence) -> "OmicsDataset":
        """Filter all modalities to the specified samples.

        Parameters
        ----------
        sample_ids:
            Sample IDs to retain.

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.
        """
        for mod in self._modalities.values():
            mod.filter_samples(sample_ids)
        return self

    def subset_cancer_types(self, types: Sequence[str]) -> "OmicsDataset":
        """Filter samples by cancer type using the dataset metadata.

        Requires ``metadata["sample_cancer_type"]`` to be a dict mapping
        sample ID to cancer type string.

        Parameters
        ----------
        types:
            Cancer type labels to retain.

        Returns
        -------
        OmicsDataset
            *self*, for method chaining.

        Raises
        ------
        KeyError
            If ``sample_cancer_type`` is not in :attr:`metadata`.
        """
        if "sample_cancer_type" not in self.metadata:
            raise KeyError(
                "metadata['sample_cancer_type'] is not set. "
                "Populate it with a dict mapping sample_id → cancer_type."
            )
        type_map: Dict[str, str] = self.metadata["sample_cancer_type"]
        keep = [sid for sid, ct in type_map.items() if ct in types]
        return self.subset_samples(keep)

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        modalities: Optional[Sequence[str]] = None,
        fill_missing: float = np.nan,
    ) -> pd.DataFrame:
        """Return a concatenated samples × features DataFrame.

        Column names are prefixed with the modality name, e.g.
        ``"rna__EGFR"``, ``"mut__TP53"``.

        Parameters
        ----------
        modalities:
            Subset of modality names to include.  ``None`` means all.
        fill_missing:
            Value used to fill when samples differ across modalities.

        Returns
        -------
        pandas.DataFrame
            Concatenated feature matrix.
        """
        names = modalities if modalities is not None else self.modality_names
        frames = []
        for name in names:
            if name not in self._modalities:
                raise KeyError(f"Modality {name!r} not found.")
            mod = self._modalities[name]
            prefixed = mod.data.add_prefix(f"{name}__")
            frames.append(prefixed)

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for frame in frames[1:]:
            result = result.join(frame, how="outer")

        if not np.isnan(fill_missing):
            result = result.fillna(fill_missing)
        return result

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Return a dict mapping modality name to its DataFrame.

        Returns
        -------
        dict[str, pandas.DataFrame]
        """
        return {name: mod.data.copy() for name, mod in self._modalities.items()}

    def to_mofa2(self) -> Dict[str, Any]:
        """Format data for mofapy2 entry_point input.

        Returns
        -------
        dict
            Keys: ``"data"`` (list-of-lists format), ``"views"`` (view names),
            ``"groups"`` (group names, single group here), ``"samples"`` (list
            of sample ID lists per group/view).

        Notes
        -----
        MOFA2 expects data as a list of views, each a list of groups, each a
        2D numpy array (samples × features), with NaN for missing values.
        """
        views = self.modality_names
        all_samples = self.common_samples.tolist()

        data_list: List[List[np.ndarray]] = []
        for name in views:
            mod = self._modalities[name]
            mat = mod.data.reindex(all_samples).values.astype(float)
            data_list.append([mat])

        return {
            "data": data_list,
            "views": views,
            "groups": ["group1"],
            "samples": [[all_samples]],
        }

    def to_tensor(self, dtype: Any = None):
        """Return a PyTorch tensor of the concatenated feature matrix.

        Requires ``torch`` to be installed.

        Parameters
        ----------
        dtype:
            PyTorch dtype.  Defaults to ``torch.float32``.

        Returns
        -------
        torch.Tensor

        Raises
        ------
        ImportError
            If ``torch`` is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for to_tensor(). "
                "Install it with: pip install torch"
            ) from exc
        if dtype is None:
            dtype = torch.float32
        df = self.to_dataframe()
        return torch.tensor(df.values, dtype=dtype)

    def to_anndata(self):
        """Return an AnnData object with modalities stored in obsm.

        Requires ``anndata`` to be installed.

        Returns
        -------
        anndata.AnnData

        Raises
        ------
        ImportError
            If ``anndata`` is not installed.
        """
        try:
            import anndata as ad
        except ImportError as exc:
            raise ImportError(
                "anndata is required for to_anndata(). "
                "Install it with: pip install anndata"
            ) from exc

        common = self.common_samples
        X = self.to_dataframe().reindex(common).values

        obsm = {}
        for name, mod in self._modalities.items():
            obsm[f"X_{name}"] = mod.data.reindex(common).values

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=common),
            obsm=obsm,
        )
        adata.uns["study_id"] = self.study_id
        adata.uns["modalities"] = self.modality_names
        return adata

    def describe(self) -> pd.DataFrame:
        """Print and return a summary table of all modalities.

        Returns
        -------
        pandas.DataFrame
            One row per modality with shape and value statistics.
        """
        rows = []
        for name, mod in self._modalities.items():
            row = mod.describe()
            row["name"] = name
            rows.append(row)
        df = pd.DataFrame(rows).set_index("name")
        logger.info("Dataset %r: %d modalities.", self.study_id, len(self._modalities))
        return df

    def __repr__(self) -> str:
        modality_str = ", ".join(
            f"{name}({mod.n_samples}×{mod.n_features})"
            for name, mod in self._modalities.items()
        )
        return (
            f"OmicsDataset("
            f"study_id={self.study_id!r}, "
            f"modalities=[{modality_str}], "
            f"n_common_samples={self.n_complete_cases})"
        )
