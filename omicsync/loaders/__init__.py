"""Data loaders for omicsync."""

from omicsync.loaders.csv import load_csv, load_multimodal_csv
from omicsync.loaders.tcga import load_tcga_files, download_tcga_manifest
from omicsync.loaders.geo import load_geo
from omicsync.loaders.open_targets import (
    load_open_targets_targets,
    add_open_targets_annotations,
)

__all__ = [
    "load_csv",
    "load_multimodal_csv",
    "load_tcga_files",
    "download_tcga_manifest",
    "load_geo",
    "load_open_targets_targets",
    "add_open_targets_annotations",
]
