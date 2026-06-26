"""Shared pytest fixtures with synthetic multi-omics data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicsync.core.dataset import OmicsDataset
from omicsync.core.modality import (
    CNVModality,
    MethylationModality,
    MutationModality,
    ProteinModality,
    RNAModality,
)

# All 50 possible sample IDs
ALL_SAMPLES = [f"SAMPLE_{i:03d}" for i in range(50)]
RNG = np.random.default_rng(42)


def _sample_subset(exclude_count: int, offset: int) -> list[str]:
    """Return ALL_SAMPLES minus *exclude_count* samples starting at *offset*."""
    exclude = set(ALL_SAMPLES[offset: offset + exclude_count])
    return [s for s in ALL_SAMPLES if s not in exclude]


@pytest.fixture
def synthetic_rna_df() -> pd.DataFrame:
    """50 samples × 200 genes, log-normal values (non-negative)."""
    samples = _sample_subset(10, 0)
    data = RNG.lognormal(mean=3.0, sigma=1.0, size=(len(samples), 200))
    return pd.DataFrame(
        data,
        index=pd.Index(samples, name="sample_id"),
        columns=[f"GENE{i:04d}" for i in range(200)],
    )


@pytest.fixture
def synthetic_methylation_df() -> pd.DataFrame:
    """40 samples × 100 CpG sites, uniform [0, 1] beta values."""
    samples = _sample_subset(10, 5)
    data = RNG.uniform(0.05, 0.95, size=(len(samples), 100))
    return pd.DataFrame(
        data,
        index=pd.Index(samples, name="sample_id"),
        columns=[f"CpG{i:04d}" for i in range(100)],
    )


@pytest.fixture
def synthetic_mutations_df() -> pd.DataFrame:
    """40 samples × 50 genes, binary 0/1."""
    samples = _sample_subset(10, 10)
    data = RNG.integers(0, 2, size=(len(samples), 50)).astype(float)
    return pd.DataFrame(
        data,
        index=pd.Index(samples, name="sample_id"),
        columns=[f"GENE{i:04d}" for i in range(50)],
    )


@pytest.fixture
def synthetic_cnv_df() -> pd.DataFrame:
    """40 samples × 100 genes, values centred around 0."""
    samples = _sample_subset(10, 15)
    data = RNG.normal(loc=0.0, scale=0.5, size=(len(samples), 100))
    return pd.DataFrame(
        data,
        index=pd.Index(samples, name="sample_id"),
        columns=[f"GENE{i:04d}" for i in range(100)],
    )


@pytest.fixture
def synthetic_protein_df() -> pd.DataFrame:
    """40 samples × 30 proteins, normal values."""
    samples = _sample_subset(10, 20)
    data = RNG.normal(loc=0.0, scale=1.0, size=(len(samples), 30))
    return pd.DataFrame(
        data,
        index=pd.Index(samples, name="sample_id"),
        columns=[f"PROT{i:03d}" for i in range(30)],
    )


@pytest.fixture
def synthetic_dataset(
    synthetic_rna_df,
    synthetic_methylation_df,
    synthetic_mutations_df,
    synthetic_cnv_df,
    synthetic_protein_df,
) -> OmicsDataset:
    """OmicsDataset with all 5 modalities; ~20 samples are common to all."""
    modalities = {
        "rna": RNAModality(synthetic_rna_df, source="synthetic"),
        "methylation": MethylationModality(synthetic_methylation_df, source="synthetic"),
        "mutations": MutationModality(synthetic_mutations_df, source="synthetic"),
        "cnv": CNVModality(synthetic_cnv_df, source="synthetic"),
        "protein": ProteinModality(synthetic_protein_df, source="synthetic"),
    }
    return OmicsDataset(modalities, study_id="SYNTHETIC")
