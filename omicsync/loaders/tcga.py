"""TCGA data loaders for omicsync."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from omicsync.core.dataset import OmicsDataset
from omicsync.core.modality import make_modality, OmicsModality
from omicsync.core.sample_index import SampleIndex
from omicsync.utils.barcode import truncate_to_participant
from omicsync.utils.logging import get_logger

logger = get_logger("loaders.tcga")

# Expected filename patterns for each modality within a TCGA data directory.
# Pattern: {cancer_type}_{modality}.{ext}
_MODALITY_FILE_PATTERNS = {
    "rna": ["{cancer_type}_rna.tsv", "{cancer_type}_rna.csv",
            "{cancer_type}_rna_fpkm.tsv", "{cancer_type}_htseq_counts.tsv"],
    "mutations": ["{cancer_type}_mutations.tsv", "{cancer_type}_mutations.maf",
                  "{cancer_type}_somatic.maf"],
    "methylation": ["{cancer_type}_methylation.tsv", "{cancer_type}_methylation.csv"],
    "cnv": ["{cancer_type}_cnv.tsv", "{cancer_type}_cnv.csv",
            "{cancer_type}_gistic2.tsv"],
    "protein": ["{cancer_type}_protein.tsv", "{cancer_type}_protein.csv",
                "{cancer_type}_rppa.tsv"],
}

# GDC data portal file type identifiers used in the manifest helper
_GDC_DATA_TYPES = {
    "rna": "Gene Expression Quantification",
    "mutations": "Masked Somatic Mutation",
    "methylation": "Methylation Beta Value",
    "cnv": "Copy Number Segment",
    "protein": "Protein Expression Quantification",
}


def _find_modality_file(
    data_dir: Path,
    cancer_type: str,
    modality: str,
) -> Optional[Path]:
    """Search *data_dir* for a file matching known naming conventions."""
    patterns = _MODALITY_FILE_PATTERNS.get(modality, [])
    for pattern in patterns:
        candidate = data_dir / pattern.format(cancer_type=cancer_type.upper())
        if candidate.exists():
            return candidate
        candidate = data_dir / pattern.format(cancer_type=cancer_type.lower())
        if candidate.exists():
            return candidate
    # Fallback: look for any file containing the modality name
    for f in data_dir.iterdir():
        if modality in f.name.lower() and f.suffix in (".tsv", ".csv", ".maf"):
            return f
    return None


def _load_maf_mutations(path: Path) -> pd.DataFrame:
    """Parse a MAF file into a binary genes × samples mutation matrix."""
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)

    required = {"Hugo_Symbol", "Tumor_Sample_Barcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"MAF file missing required columns: {missing}. "
            f"Found: {df.columns.tolist()[:15]}."
        )

    df["sample_id"] = df["Tumor_Sample_Barcode"].apply(
        lambda x: truncate_to_participant(x) if isinstance(x, str) and x.startswith("TCGA") else x
    )
    mat = (
        df.groupby(["sample_id", "Hugo_Symbol"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )
    return mat.astype(float)


def _load_generic_matrix(path: Path) -> pd.DataFrame:
    """Load a TSV/CSV as a sample-indexed matrix.

    Tries to detect whether samples are rows or columns.
    """
    sep = "\t" if path.suffix in (".tsv", ".maf") else ","
    df = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)

    # Heuristic: if more than half of values in first column look numeric,
    # samples are rows; otherwise transpose.
    numeric_frac = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean()
    if numeric_frac < 0.5:
        df = df.T

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_tcga_files(
    data_dir: Union[str, Path],
    cancer_type: str,
    modalities: Sequence[str],
) -> OmicsDataset:
    """Load TCGA data from local files into an :class:`~omicsync.core.dataset.OmicsDataset`.

    Parameters
    ----------
    data_dir:
        Path to directory containing TCGA data files.  Expected naming:
        ``{cancer_type}_{modality}.tsv`` or ``{cancer_type}_{modality}.csv``.
        For mutations, ``.maf`` files are also supported.
    cancer_type:
        TCGA cancer type abbreviation, e.g. ``"BRCA"``.
    modalities:
        List of modality names to load, e.g.
        ``["rna", "mutations", "methylation"]``.

    Returns
    -------
    OmicsDataset

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    ValueError
        If no file is found for a requested modality.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loaded: Dict[str, OmicsModality] = {}

    for modality in modalities:
        path = _find_modality_file(data_dir, cancer_type, modality)
        if path is None:
            raise ValueError(
                f"No file found for modality {modality!r} in {data_dir}. "
                f"Expected patterns like: "
                + str(_MODALITY_FILE_PATTERNS.get(modality, []))
            )

        logger.info("load_tcga_files: loading %r from %s.", modality, path.name)

        if modality == "mutations" and path.suffix == ".maf":
            df = _load_maf_mutations(path)
        else:
            df = _load_generic_matrix(path)

        # Harmonise TCGA barcodes if detected
        if all(str(idx).startswith("TCGA") for idx in df.index[:5]):
            df.index = pd.Index(
                [truncate_to_participant(str(i)) for i in df.index],
                name=df.index.name,
            )
            df = df[~df.index.duplicated(keep="first")]
            logger.info(
                "load_tcga_files: truncated %r sample IDs to participant level.",
                modality,
            )

        loaded[modality] = make_modality(df, modality_type=modality, source="tcga")

    dataset = OmicsDataset(loaded, study_id=f"TCGA-{cancer_type.upper()}")

    # Print coverage report
    coverage = dataset.sample_coverage
    logger.info(
        "TCGA %s coverage: %d total samples, %d complete cases.",
        cancer_type.upper(),
        len(coverage),
        dataset.n_complete_cases,
    )
    return dataset


def download_tcga_manifest(
    cancer_type: str,
    modalities: Sequence[str],
    output_dir: Union[str, Path],
) -> None:
    """Print GDC data portal instructions for downloading TCGA data.

    This function does NOT download anything itself — TCGA data via the GDC
    API requires authentication tokens.  It prints the required ``curl``
    commands and GDC portal URLs so you can retrieve the data manually.

    Parameters
    ----------
    cancer_type:
        TCGA cancer type abbreviation, e.g. ``"BRCA"``.
    modalities:
        Modalities to retrieve, e.g. ``["rna", "mutations"]``.
    output_dir:
        Directory where you plan to save the files (used in printed commands).
    """
    ct = cancer_type.upper()
    output_dir = Path(output_dir)

    instructions = [
        f"",
        f"TCGA {ct} data download instructions",
        f"{'=' * 50}",
        f"",
        f"To download TCGA data you need a GDC account and an API token.",
        f"",
        f"1. Create/log into your account at: https://portal.gdc.cancer.gov/",
        f"2. Download your API token from:",
        f"   Profile → Download Token  (valid 30 days)",
        f"3. Save the token to a file, e.g. ~/gdc-token.txt",
        f"",
        f"Data types needed for {ct}:",
        f"",
    ]

    for modality in modalities:
        data_type = _GDC_DATA_TYPES.get(modality, modality)
        filename = f"{ct}_{modality}.tsv"
        instructions += [
            f"  [{modality.upper()}]",
            f"  GDC data type : {data_type}",
            f"  Portal filter : https://portal.gdc.cancer.gov/repository"
            f"?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-{ct}%22%5D%7D%7D%5D%7D"
            f"&facetTab=files&searchTableTab=files",
            f"  Save to       : {output_dir / filename}",
            f"",
        ]

    instructions += [
        f"Alternative: use the GDC client tool",
        f"  pip install gdc-client",
        f"  gdc-client download -t ~/gdc-token.txt -d {output_dir} <manifest-file>",
        f"",
        f"Alternative: use TCGAbiolinks (R):",
        f"  library(TCGAbiolinks)",
        f"  query <- GDCquery(project = 'TCGA-{ct}', ...)",
        f"  GDCdownload(query)",
        f"",
    ]

    for line in instructions:
        logger.info(line)
        print(line)
