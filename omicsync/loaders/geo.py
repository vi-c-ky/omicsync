"""GEO loader using GEOparse."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from omicsync.core.modality import make_modality, OmicsModality
from omicsync.utils.logging import get_logger

logger = get_logger("loaders.geo")


def load_geo(
    accession: str,
    modality_type: str,
    destdir: str = ".",
    silent: bool = True,
    **kwargs,
) -> OmicsModality:
    """Download and parse a GEO series into an :class:`~omicsync.core.modality.OmicsModality`.

    Requires ``GEOparse`` to be installed (``pip install GEOparse``).

    Parameters
    ----------
    accession:
        GEO series accession, e.g. ``"GSE12345"``.
    modality_type:
        One of ``"rna"``, ``"mutations"``, ``"methylation"``, ``"cnv"``,
        ``"protein"``.
    destdir:
        Directory to download GEO files into.
    silent:
        Suppress GEOparse download progress output (default ``True``).
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`GEOparse.get_GEO`.

    Returns
    -------
    OmicsModality

    Raises
    ------
    ImportError
        If ``GEOparse`` is not installed.
    ValueError
        If the series has no usable expression matrix.
    """
    try:
        import GEOparse
    except ImportError as exc:
        raise ImportError(
            "GEOparse is required for load_geo(). "
            "Install it with: pip install GEOparse"
        ) from exc

    logger.info("load_geo: fetching %s from NCBI GEO.", accession)
    gse = GEOparse.get_GEO(accession, destdir=destdir, silent=silent, **kwargs)

    platforms = gse.gpls
    if len(platforms) > 1:
        logger.warning(
            "load_geo: %s has %d platforms (%s). "
            "Using first platform; consider filtering manually.",
            accession,
            len(platforms),
            list(platforms.keys()),
        )

    # Build expression matrix from GSMs
    gsms = gse.gsms
    if not gsms:
        raise ValueError(f"GEO series {accession} contains no samples (GSMs).")

    frames = {}
    for sample_name, gsm in gsms.items():
        table = gsm.table
        if table.empty:
            logger.warning("load_geo: sample %s has an empty table; skipping.", sample_name)
            continue
        # Detect value column: prefer "VALUE", else first numeric column
        value_col = "VALUE" if "VALUE" in table.columns else None
        if value_col is None:
            for col in table.columns:
                if col != "ID_REF" and pd.api.types.is_numeric_dtype(table[col]):
                    value_col = col
                    break
        if value_col is None:
            logger.warning("load_geo: cannot find value column in sample %s.", sample_name)
            continue
        id_col = "ID_REF" if "ID_REF" in table.columns else table.columns[0]
        frames[sample_name] = table.set_index(id_col)[value_col]

    if not frames:
        raise ValueError(f"No usable data found in GEO series {accession}.")

    df = pd.DataFrame(frames).T
    df.index.name = "sample_id"
    df = df.apply(pd.to_numeric, errors="coerce")

    logger.info(
        "load_geo: loaded %s — %d samples × %d features.",
        accession,
        df.shape[0],
        df.shape[1],
    )
    return make_modality(df, modality_type=modality_type, source=f"geo:{accession}")
