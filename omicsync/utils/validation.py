"""Input validation helpers for omicsync."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from omicsync.utils.logging import get_logger

logger = get_logger("utils.validation")

VALID_MODALITY_TYPES = frozenset(
    {"rna", "mutations", "methylation", "cnv", "protein"}
)


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    min_samples: int = 1,
    min_features: int = 1,
) -> None:
    """Validate that *df* is a non-empty DataFrame with the expected shape.

    Parameters
    ----------
    df:
        DataFrame to validate.
    name:
        Human-readable name used in error messages.
    min_samples:
        Minimum number of rows required.
    min_features:
        Minimum number of columns required.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If the DataFrame does not meet size requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}.")
    if df.shape[0] < min_samples:
        raise ValueError(
            f"{name} must have at least {min_samples} sample(s); got {df.shape[0]}."
        )
    if df.shape[1] < min_features:
        raise ValueError(
            f"{name} must have at least {min_features} feature(s); got {df.shape[1]}."
        )


def validate_modality_type(modality_type: str) -> None:
    """Raise :exc:`ValueError` if *modality_type* is not a recognised type.

    Parameters
    ----------
    modality_type:
        One of ``"rna"``, ``"mutations"``, ``"methylation"``, ``"cnv"``,
        ``"protein"``.

    Raises
    ------
    ValueError
        If *modality_type* is unrecognised.
    """
    if modality_type not in VALID_MODALITY_TYPES:
        raise ValueError(
            f"Unknown modality_type {modality_type!r}. "
            f"Valid types: {sorted(VALID_MODALITY_TYPES)}."
        )


def check_value_range(df: pd.DataFrame, modality_type: str) -> None:
    """Warn if values in *df* look unusual for the given modality type.

    Parameters
    ----------
    df:
        Feature matrix (samples × features).
    modality_type:
        One of the recognised omicsync modality types.
    """
    values = df.values.ravel()
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        logger.warning("DataFrame contains no finite values.")
        return

    vmin, vmax, vmean = finite.min(), finite.max(), finite.mean()

    if modality_type == "rna":
        if vmin < 0:
            logger.warning(
                "RNA modality: found negative values (min=%.4f). "
                "Expected non-negative counts or expression values.",
                vmin,
            )
    elif modality_type == "methylation":
        if vmax > 1.05 or vmin < -1.05:
            logger.warning(
                "Methylation modality: values outside [-1, 1] detected "
                "(min=%.4f, max=%.4f). Beta values should be in [0, 1]; "
                "M-values typically in [-5, 5].",
                vmin,
                vmax,
            )
    elif modality_type == "mutations":
        unique = np.unique(finite)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            logger.warning(
                "Mutation modality: non-binary values detected. "
                "Consider calling binarise() first."
            )
    elif modality_type == "protein":
        if abs(vmean) > 10:
            logger.warning(
                "Protein modality: mean value %.4f seems high. "
                "Consider z-score normalisation.",
                vmean,
            )


def validate_sample_ids(ids: Sequence) -> None:
    """Check sample IDs for duplicates, NaN, and empty strings.

    Parameters
    ----------
    ids:
        Sequence of sample identifiers.

    Raises
    ------
    ValueError
        If any ID is NaN, empty, or duplicated.
    """
    seen: set = set()
    duplicates: list = []
    for idx, sid in enumerate(ids):
        if sid is None or (isinstance(sid, float) and np.isnan(sid)):
            raise ValueError(f"Sample ID at position {idx} is NaN/None.")
        if isinstance(sid, str) and sid.strip() == "":
            raise ValueError(f"Sample ID at position {idx} is an empty string.")
        if sid in seen:
            duplicates.append(sid)
        seen.add(sid)
    if duplicates:
        raise ValueError(f"Duplicate sample IDs found: {duplicates[:10]}.")
