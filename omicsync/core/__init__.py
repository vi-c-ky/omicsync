"""Core data structures for omicsync."""

from omicsync.core.dataset import OmicsDataset
from omicsync.core.modality import (
    OmicsModality,
    RNAModality,
    MutationModality,
    MethylationModality,
    CNVModality,
    ProteinModality,
    make_modality,
)
from omicsync.core.sample_index import SampleIndex

__all__ = [
    "OmicsDataset",
    "OmicsModality",
    "RNAModality",
    "MutationModality",
    "MethylationModality",
    "CNVModality",
    "ProteinModality",
    "make_modality",
    "SampleIndex",
]
