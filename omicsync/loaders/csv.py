"""Generic CSV/TSV loader for omicsync."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from omicsync.core.dataset import OmicsDataset
from omicsync.core.modality import make_modality, OmicsModality
from omicsync.utils.logging import get_logger
from omicsync.utils.validation import validate_modality_type

logger = get_logger("loaders.csv")


def _detect_separator(path: Union[str, Path]) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".txt"):
        return "\t"
    if suffix == ".csv":
        return ","
    # Peek at first line to detect
    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()
    if first_line.count("\t") > first_line.count(","):
        return "\t"
    return ","


def load_csv(
    path: Union[str, Path],
    modality_type: str,
    sample_col: Optional[str] = "sample_id",
    feature_orientation: str = "samples_as_rows",
    source: str = "csv",
    **kwargs,
) -> OmicsModality:
    """Load a single CSV/TSV file into an :class:`~omicsync.core.modality.OmicsModality`.

    Parameters
    ----------
    path:
        Path to the CSV or TSV file.
    modality_type:
        One of ``"rna"``, ``"mutations"``, ``"methylation"``, ``"cnv"``,
        ``"protein"``.
    sample_col:
        Name of the column that contains sample IDs when
        ``feature_orientation="samples_as_rows"``.  Set to ``None`` to use the
        existing index.  Ignored when ``feature_orientation="samples_as_columns"``.
    feature_orientation:
        ``"samples_as_rows"`` (default) — rows are samples, columns are
        features.  ``"samples_as_columns"`` — transpose after reading.
    source:
        Source label stored in the modality metadata.
    **kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    OmicsModality
        The appropriate modality subclass.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *modality_type* or *feature_orientation* is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    validate_modality_type(modality_type)
    if feature_orientation not in ("samples_as_rows", "samples_as_columns"):
        raise ValueError(
            f"Unknown feature_orientation {feature_orientation!r}. "
            "Valid: 'samples_as_rows', 'samples_as_columns'."
        )

    sep = kwargs.pop("sep", _detect_separator(path))
    df = pd.read_csv(path, sep=sep, **kwargs)

    if feature_orientation == "samples_as_rows":
        if sample_col is not None:
            if sample_col not in df.columns:
                raise ValueError(
                    f"sample_col={sample_col!r} not found in columns: {df.columns.tolist()[:10]}..."
                )
            df = df.set_index(sample_col)
    else:
        if sample_col is not None and sample_col in df.columns:
            df = df.set_index(sample_col)
        df = df.T

    df = df.apply(pd.to_numeric, errors="coerce")

    logger.info(
        "load_csv: loaded %s modality from %s — shape %s.",
        modality_type,
        path.name,
        df.shape,
    )
    return make_modality(df, modality_type=modality_type, source=source)


def load_multimodal_csv(
    paths_dict: Dict[str, Union[str, Path]],
    modality_types: Optional[Dict[str, str]] = None,
    study_id: str = "custom",
    **kwargs,
) -> OmicsDataset:
    """Load multiple CSV/TSV files into an :class:`~omicsync.core.dataset.OmicsDataset`.

    Parameters
    ----------
    paths_dict:
        Mapping from modality name to file path.
    modality_types:
        Mapping from modality name to modality_type string.  If ``None``,
        the modality name itself is used as the type.
    study_id:
        Study identifier for the resulting dataset.
    **kwargs:
        Forwarded to :func:`load_csv` for every modality.

    Returns
    -------
    OmicsDataset

    Raises
    ------
    ValueError
        If a modality name cannot be resolved to a valid modality type.
    """
    modalities: Dict[str, OmicsModality] = {}
    for name, path in paths_dict.items():
        mtype = (modality_types or {}).get(name, name)
        logger.info("load_multimodal_csv: loading %r from %s.", name, path)
        modalities[name] = load_csv(path, modality_type=mtype, **kwargs)

    return OmicsDataset(modalities, study_id=study_id)
