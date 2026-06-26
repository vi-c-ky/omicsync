"""omicsync — Multi-omics data harmonisation for Python."""

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
from omicsync.utils.logging import set_verbose, get_logger

__version__ = "0.1.0"
__author__ = "Paterson V."
__license__ = "MIT"

__all__ = [
    "__version__",
    "OmicsDataset",
    "OmicsModality",
    "RNAModality",
    "MutationModality",
    "MethylationModality",
    "CNVModality",
    "ProteinModality",
    "make_modality",
    "SampleIndex",
    "set_verbose",
    "get_logger",
]
