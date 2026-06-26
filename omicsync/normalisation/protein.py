"""Protein abundance normalisation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omicsync.utils.logging import get_logger

logger = get_logger("normalisation.protein")


def z_score(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalise protein abundance per feature (column).

    Constant columns (zero standard deviation) are set to 0.

    Parameters
    ----------
    df:
        Protein abundance matrix (samples × proteins).

    Returns
    -------
    pandas.DataFrame
        Z-scored matrix.
    """
    data = df.values.astype(float)
    mu = np.nanmean(data, axis=0, keepdims=True)
    sd = np.nanstd(data, axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    result = (data - mu) / sd
    logger.info("z_score (protein): applied to %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def median_centring(df: pd.DataFrame) -> pd.DataFrame:
    """Centre each protein on its median across samples.

    Parameters
    ----------
    df:
        Protein abundance matrix (samples × proteins).

    Returns
    -------
    pandas.DataFrame
        Median-centred matrix.
    """
    data = df.values.astype(float)
    medians = np.nanmedian(data, axis=0, keepdims=True)
    result = data - medians
    logger.info("median_centring: applied to %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)
