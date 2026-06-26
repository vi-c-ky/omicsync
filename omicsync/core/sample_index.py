"""Sample ID harmonisation logic for multi-omics datasets."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from omicsync.utils.barcode import truncate_to_participant, truncate_to_sample
from omicsync.utils.logging import get_logger

logger = get_logger("core.sample_index")

_LEVEL_FUNCS = {
    "participant": truncate_to_participant,
    "sample": truncate_to_sample,
    "aliquot": lambda x: x,
}


class SampleIndex:
    """Manages sample ID sets and harmonisation across modalities.

    Parameters
    ----------
    sample_ids:
        Initial set of sample identifiers (optional).
    """

    def __init__(self, sample_ids: Optional[Sequence] = None) -> None:
        self._ids: pd.Index = (
            pd.Index(sample_ids) if sample_ids is not None else pd.Index([])
        )

    @classmethod
    def from_barcodes(
        cls,
        barcodes: Sequence[str],
        level: str = "participant",
    ) -> "SampleIndex":
        """Create a :class:`SampleIndex` by truncating TCGA barcodes.

        Parameters
        ----------
        barcodes:
            Full TCGA aliquot barcodes.
        level:
            Truncation level: ``"participant"`` (default), ``"sample"``,
            or ``"aliquot"`` (no truncation).

        Returns
        -------
        SampleIndex

        Raises
        ------
        ValueError
            If *level* is not recognised.
        """
        if level not in _LEVEL_FUNCS:
            raise ValueError(
                f"Unknown barcode level {level!r}. "
                f"Valid levels: {list(_LEVEL_FUNCS)}."
            )
        func = _LEVEL_FUNCS[level]
        truncated = []
        for bc in barcodes:
            try:
                truncated.append(func(bc))
            except ValueError:
                logger.warning("Could not parse barcode %r at level %r; keeping as-is.", bc, level)
                truncated.append(bc)
        idx = cls(truncated)
        logger.info(
            "SampleIndex: %d barcodes → %d unique IDs at level %r.",
            len(barcodes),
            len(set(truncated)),
            level,
        )
        return idx

    @staticmethod
    def align(
        list_of_sample_id_arrays: Sequence[Union[Sequence, pd.Index]],
        strategy: str = "intersection",
    ) -> pd.Index:
        """Find common samples across multiple modalities.

        Parameters
        ----------
        list_of_sample_id_arrays:
            One array/index per modality.
        strategy:
            ``"intersection"`` (default) — samples present in every modality.
            ``"union"`` — all samples seen across any modality.

        Returns
        -------
        pandas.Index
            Aligned sample IDs.

        Raises
        ------
        ValueError
            If *strategy* is unrecognised or the input list is empty.
        """
        if not list_of_sample_id_arrays:
            raise ValueError("list_of_sample_id_arrays must not be empty.")
        if strategy not in ("intersection", "union"):
            raise ValueError(
                f"Unknown strategy {strategy!r}. Valid: 'intersection', 'union'."
            )

        indices = [pd.Index(arr) for arr in list_of_sample_id_arrays]
        if strategy == "intersection":
            result = indices[0]
            for idx in indices[1:]:
                result = result.intersection(idx)
        else:
            result = indices[0]
            for idx in indices[1:]:
                result = result.union(idx)

        logger.info(
            "SampleIndex.align (%s): %d common samples from %d modalities.",
            strategy,
            len(result),
            len(indices),
        )
        return result

    @staticmethod
    def match_fuzzy(
        ids_a: Sequence[str],
        ids_b: Sequence[str],
    ) -> Dict[str, Optional[str]]:
        """Match IDs across two sets, tolerating minor formatting differences.

        Normalises by converting to uppercase and replacing dots with dashes
        before matching.

        Parameters
        ----------
        ids_a:
            Reference ID set.
        ids_b:
            Query ID set.

        Returns
        -------
        dict
            Mapping from each ID in *ids_a* to the best matching ID in
            *ids_b*, or ``None`` if no match found.
        """

        def _normalise(s: str) -> str:
            return s.strip().upper().replace(".", "-")

        norm_b: Dict[str, str] = {_normalise(b): b for b in ids_b}
        result: Dict[str, Optional[str]] = {}
        for a in ids_a:
            key = _normalise(a)
            result[a] = norm_b.get(key)
        n_matched = sum(v is not None for v in result.values())
        logger.info(
            "Fuzzy match: %d/%d IDs matched.", n_matched, len(ids_a)
        )
        return result

    def summarise(
        self,
        modality_sample_ids: Dict[str, Sequence],
    ) -> pd.DataFrame:
        """Report how many samples are present in each modality combination.

        Parameters
        ----------
        modality_sample_ids:
            Mapping from modality name to its sample IDs.

        Returns
        -------
        pandas.DataFrame
            Rows are samples; columns are modality names; values are boolean
            (``True`` = present).  An additional ``n_modalities`` column gives
            the count of modalities present for each sample.
        """
        all_ids = set()
        for ids in modality_sample_ids.values():
            all_ids.update(ids)
        df = pd.DataFrame(
            {
                name: pd.Index(all_ids).isin(ids)
                for name, ids in modality_sample_ids.items()
            },
            index=pd.Index(sorted(all_ids), name="sample_id"),
        )
        df["n_modalities"] = df.sum(axis=1)
        return df
