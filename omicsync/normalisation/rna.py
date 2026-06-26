"""RNA-seq normalisation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from omicsync.utils.logging import get_logger

logger = get_logger("normalisation.rna")


def log1p_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to all values.

    Parameters
    ----------
    df:
        Expression matrix (samples × features). Values must be non-negative.

    Returns
    -------
    pandas.DataFrame
        log1p-transformed matrix with same index and columns.
    """
    result = np.log1p(df.values.astype(float))
    logger.info("log1p_normalise: applied to %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def tpm_to_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to a TPM expression matrix.

    Parameters
    ----------
    df:
        TPM expression matrix (samples × genes).

    Returns
    -------
    pandas.DataFrame
        log1p(TPM) matrix.
    """
    logger.info("tpm_to_log1p: applying log1p to TPM matrix %s.", df.shape)
    return log1p_normalise(df)


def counts_to_tpm(df: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
    """Convert raw counts to TPM using gene lengths.

    Parameters
    ----------
    df:
        Raw count matrix (samples × genes).
    gene_lengths:
        Gene lengths in base pairs, indexed by gene ID matching *df* columns.

    Returns
    -------
    pandas.DataFrame
        TPM matrix.

    Raises
    ------
    ValueError
        If gene lengths are missing for any column in *df*.
    """
    missing = df.columns.difference(gene_lengths.index)
    if len(missing) > 0:
        raise ValueError(
            f"Gene lengths missing for {len(missing)} genes: {missing[:5].tolist()}..."
        )
    lengths = gene_lengths.reindex(df.columns).values.astype(float)
    rpk = df.values.astype(float) / (lengths / 1e3)
    scaling = rpk.sum(axis=1, keepdims=True) / 1e6
    tpm = rpk / np.where(scaling == 0, 1.0, scaling)
    logger.info("counts_to_tpm: converted %s to TPM.", df.shape)
    return pd.DataFrame(tpm, index=df.index, columns=df.columns)


def quantile_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile normalise a matrix so each sample has the same distribution.

    Parameters
    ----------
    df:
        Expression matrix (samples × features).

    Returns
    -------
    pandas.DataFrame
        Quantile-normalised matrix.
    """
    data = df.values.astype(float).copy()
    n_samples, n_features = data.shape

    sort_indices = np.argsort(data, axis=1)
    sorted_data = np.sort(data, axis=1)
    row_means = sorted_data.mean(axis=0)

    result = np.empty_like(data)
    for i in range(n_samples):
        result[i, sort_indices[i]] = row_means

    logger.info("quantile_normalise: applied to %s.", df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def z_score(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """Z-score normalise the expression matrix.

    Parameters
    ----------
    df:
        Expression matrix.
    axis:
        ``0`` to z-score per feature (column), ``1`` to z-score per sample (row).

    Returns
    -------
    pandas.DataFrame
        Z-scored matrix. Constant features/samples are set to 0.
    """
    data = df.values.astype(float)
    mu = np.nanmean(data, axis=axis, keepdims=True)
    sd = np.nanstd(data, axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    result = (data - mu) / sd
    logger.info("z_score: applied along axis=%d to %s.", axis, df.shape)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def detect_and_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect RNA value type and apply appropriate normalisation.

    Heuristic:

    * If max value > 50 and median > 5 → assume raw counts, apply log1p.
    * If max value in [0.1, 50] and median < 5 → assume TPM, apply log1p.
    * Otherwise → assume already normalised, return as-is.

    Parameters
    ----------
    df:
        RNA expression matrix (samples × features).

    Returns
    -------
    pandas.DataFrame
        Normalised matrix.
    """
    vals = df.values.ravel().astype(float)
    finite = vals[np.isfinite(vals) & (vals >= 0)]
    if len(finite) == 0:
        logger.warning("detect_and_normalise: no finite non-negative values; skipping.")
        return df

    vmax = finite.max()
    vmedian = np.median(finite)

    if vmax > 50 and vmedian > 5:
        logger.info(
            "detect_and_normalise: detected raw counts (max=%.1f, median=%.2f); "
            "applying log1p.",
            vmax, vmedian,
        )
        return log1p_normalise(df)
    elif vmax > 0.1:
        logger.info(
            "detect_and_normalise: detected TPM-like values (max=%.1f, median=%.2f); "
            "applying log1p.",
            vmax, vmedian,
        )
        return tpm_to_log1p(df)
    else:
        logger.info(
            "detect_and_normalise: values appear already normalised "
            "(max=%.4f, median=%.4f); returning as-is.",
            vmax, vmedian,
        )
        return df
