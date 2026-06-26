"""Methylation normalisation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omicsync.utils.logging import get_logger

logger = get_logger("normalisation.methylation")


def beta_to_m(df: pd.DataFrame) -> pd.DataFrame:
    """Convert beta values to M-values: log2(beta / (1 - beta)).

    Parameters
    ----------
    df:
        Beta value matrix (samples × CpG sites). Values must be in (0, 1).

    Returns
    -------
    pandas.DataFrame
        M-value matrix.

    Raises
    ------
    ValueError
        If any value is outside [0, 1].
    """
    data = df.values.astype(float)
    finite = data[np.isfinite(data)]
    if len(finite) > 0 and (finite.min() < 0 or finite.max() > 1):
        raise ValueError(
            f"beta_to_m: beta values must be in [0, 1]. "
            f"Got min={finite.min():.4f}, max={finite.max():.4f}. "
            "Clip first with clip_beta()."
        )
    eps = 1e-6
    clipped = np.clip(data, eps, 1 - eps)
    result = np.log2(clipped / (1 - clipped))
    logger.info("beta_to_m: converted beta → M-values for %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def m_to_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Convert M-values back to beta values: 2^M / (2^M + 1).

    Parameters
    ----------
    df:
        M-value matrix (samples × CpG sites).

    Returns
    -------
    pandas.DataFrame
        Beta value matrix in (0, 1).
    """
    data = df.values.astype(float)
    exp = np.power(2.0, data)
    result = exp / (exp + 1.0)
    logger.info("m_to_beta: converted M-values → beta for %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def clip_beta(
    df: pd.DataFrame,
    low: float = 0.001,
    high: float = 0.999,
) -> pd.DataFrame:
    """Clip beta values to avoid extreme values near 0 and 1.

    Parameters
    ----------
    df:
        Beta value matrix.
    low:
        Lower clip bound (default 0.001).
    high:
        Upper clip bound (default 0.999).

    Returns
    -------
    pandas.DataFrame
        Clipped beta matrix.
    """
    result = df.values.astype(float).clip(low, high)
    logger.info("clip_beta: clipped to [%.4f, %.4f] for %s.", low, high, df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def detect_and_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect M-values vs beta values and normalise to clipped beta.

    Heuristic: if any finite value is outside [0, 1], treat as M-values
    and convert to beta.  Otherwise clip beta to [0.001, 0.999].

    Parameters
    ----------
    df:
        Methylation matrix (samples × CpG sites).

    Returns
    -------
    pandas.DataFrame
        Beta values clipped to [0.001, 0.999].
    """
    data = df.values.astype(float)
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        logger.warning("detect_and_normalise: no finite values; skipping.")
        return df

    vmin, vmax = finite.min(), finite.max()

    if vmin < -0.01 or vmax > 1.01:
        logger.info(
            "detect_and_normalise (methylation): detected M-values "
            "(min=%.4f, max=%.4f); converting to beta.",
            vmin, vmax,
        )
        result = m_to_beta(df)
    else:
        logger.info(
            "detect_and_normalise (methylation): detected beta values "
            "(min=%.4f, max=%.4f); clipping.",
            vmin, vmax,
        )
        result = df

    return clip_beta(result)
