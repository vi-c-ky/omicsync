"""Copy number variation normalisation utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from omicsync.utils.logging import get_logger

logger = get_logger("normalisation.cnv")


def centre_diploid(df: pd.DataFrame, diploid: float = 2.0) -> pd.DataFrame:
    """Subtract the diploid baseline from all values.

    Parameters
    ----------
    df:
        CNV matrix (samples × genes). Values are typically absolute copy
        number estimates (centred around 2 for diploid).
    diploid:
        The baseline copy number to subtract (default 2.0).

    Returns
    -------
    pandas.DataFrame
        Copy number deviation from diploid.
    """
    result = df.values.astype(float) - diploid
    logger.info("centre_diploid: subtracted diploid=%.1f from %s.", diploid, df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def log2_ratio(df: pd.DataFrame, pseudo: float = 1.0) -> pd.DataFrame:
    """Compute log2 copy-number ratio relative to diploid.

    Assumes input is already centred (deviation from diploid = 0).
    Adds *pseudo* before log2 to handle zero deviations.

    Parameters
    ----------
    df:
        CNV deviation matrix (output of :func:`centre_diploid`).
    pseudo:
        Pseudocount added before log2 transform (default 1.0).

    Returns
    -------
    pandas.DataFrame
        log2 ratio matrix.
    """
    data = df.values.astype(float)
    shifted = data + pseudo
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(shifted > 0, np.log2(shifted), np.nan)
    logger.info("log2_ratio: applied log2 to %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def discretise(
    df: pd.DataFrame,
    thresholds: Tuple[float, float, float, float] = (-1.0, -0.3, 0.3, 1.0),
) -> pd.DataFrame:
    """Discretise copy-number values into -2/-1/0/1/2 states.

    Parameters
    ----------
    df:
        CNV matrix (log2 ratios or deviations from diploid).
    thresholds:
        Four boundary values ``(deep_del, del, amp, high_amp)`` that define
        the five copy-number states:

        * < deep_del  → -2 (deep deletion)
        * < del       → -1 (deletion)
        * <= amp      → 0  (diploid)
        * <= high_amp → 1  (gain)
        * > high_amp  → 2  (amplification)

    Returns
    -------
    pandas.DataFrame
        Integer copy-number state matrix.
    """
    if len(thresholds) != 4:
        raise ValueError("thresholds must have exactly 4 values.")
    t1, t2, t3, t4 = thresholds
    data = df.values.astype(float)
    result = np.zeros_like(data, dtype=float)
    result[data < t1] = -2.0
    result[(data >= t1) & (data < t2)] = -1.0
    result[(data > t3) & (data <= t4)] = 1.0
    result[data > t4] = 2.0
    logger.info("discretise: discretised CNV matrix %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)
