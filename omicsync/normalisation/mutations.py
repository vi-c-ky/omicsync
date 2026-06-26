"""Mutation matrix processing utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from omicsync.utils.logging import get_logger

logger = get_logger("normalisation.mutations")

# Standard Sequence Ontology consequence terms
CONSEQUENCE_TERMS = frozenset({
    "missense_variant",
    "stop_gained",
    "stop_lost",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "inframe_insertion",
    "inframe_deletion",
    "synonymous_variant",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "non_coding_transcript_variant",
})


def binarise(df: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
    """Binarise a mutation matrix: any value above *threshold* becomes 1.

    Parameters
    ----------
    df:
        Mutation matrix (samples × genes). May contain counts or continuous
        variant scores.
    threshold:
        Values strictly above this threshold are set to 1; others to 0.

    Returns
    -------
    pandas.DataFrame
        Binary mutation matrix with dtype float64.
    """
    result = (df.values.astype(float) > threshold).astype(float)
    logger.info(
        "binarise: threshold=%.2f applied to %s; "
        "%.1f%% mutated entries.",
        threshold,
        df.shape,
        100.0 * result.mean(),
    )
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def filter_by_consequence(
    df: pd.DataFrame,
    consequences: Sequence[str],
    consequence_map: dict | None = None,
) -> pd.DataFrame:
    """Keep only genes that have at least one sample with a specified consequence.

    This function operates on a pre-binarised mutation matrix.  If a
    ``consequence_map`` is provided (mapping gene → consequence), genes whose
    mapped consequence is not in *consequences* are zeroed out.

    Parameters
    ----------
    df:
        Mutation matrix (samples × genes).
    consequences:
        Consequence types to retain, e.g. ``["missense_variant", "stop_gained"]``.
    consequence_map:
        Optional dict mapping gene ID to its predominant consequence.  If
        ``None``, this function simply returns *df* unchanged with a warning.

    Returns
    -------
    pandas.DataFrame
        Filtered mutation matrix.
    """
    if consequence_map is None:
        logger.warning(
            "filter_by_consequence: no consequence_map provided; returning input unchanged."
        )
        return df

    keep = [gene for gene in df.columns if consequence_map.get(gene) in consequences]
    n_before = df.shape[1]
    result = df[keep].copy()
    logger.info(
        "filter_by_consequence: kept %d/%d genes matching %s.",
        len(keep),
        n_before,
        list(consequences),
    )
    return result


def compute_tmb(df: pd.DataFrame) -> pd.Series:
    """Compute tumour mutation burden (total mutations per sample).

    Parameters
    ----------
    df:
        Binary mutation matrix (samples × genes).

    Returns
    -------
    pandas.Series
        Mutation count per sample, indexed by sample ID.
    """
    tmb = df.sum(axis=1).rename("tmb")
    logger.info(
        "compute_tmb: TMB computed for %d samples; mean=%.2f.", len(tmb), tmb.mean()
    )
    return tmb
