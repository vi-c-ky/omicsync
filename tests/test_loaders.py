"""Tests for data loaders."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicsync.loaders.csv import load_csv, load_multimodal_csv
from omicsync.utils.barcode import (
    truncate_to_participant,
    truncate_to_sample,
    is_tumour,
    is_normal,
    batch_parse,
    parse_barcode,
)


class TestLoadCsv:
    def _write_rna_csv(self, path: Path, orientation: str = "samples_as_rows") -> None:
        rng = np.random.default_rng(0)
        data = rng.lognormal(3, 1, size=(20, 50))
        if orientation == "samples_as_rows":
            df = pd.DataFrame(
                data,
                index=[f"S{i}" for i in range(20)],
                columns=[f"GENE{i}" for i in range(50)],
            )
            df.index.name = "sample_id"
            df.to_csv(path, sep="\t")
        else:
            df = pd.DataFrame(
                data.T,
                index=[f"GENE{i}" for i in range(50)],
                columns=[f"S{i}" for i in range(20)],
            )
            df.index.name = "gene_id"
            df.to_csv(path, sep="\t")

    def test_load_csv_samples_as_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "rna.tsv"
            self._write_rna_csv(p, orientation="samples_as_rows")
            mod = load_csv(p, modality_type="rna", sample_col="sample_id")
            assert mod.n_samples == 20
            assert mod.n_features == 50

    def test_load_csv_samples_as_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "rna.tsv"
            self._write_rna_csv(p, orientation="samples_as_columns")
            mod = load_csv(
                p,
                modality_type="rna",
                sample_col=None,
                feature_orientation="samples_as_columns",
                index_col=0,
            )
            assert mod.n_samples == 20
            assert mod.n_features == 50

    def test_load_csv_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv", modality_type="rna")

    def test_load_csv_invalid_modality_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            pd.DataFrame({"A": [1.0]}).to_csv(p)
            with pytest.raises(ValueError, match="Unknown modality_type"):
                load_csv(p, modality_type="invalid_type")

    def test_load_csv_invalid_orientation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.csv"
            pd.DataFrame({"sample_id": ["S1"], "G1": [1.0]}).to_csv(p)
            with pytest.raises(ValueError, match="feature_orientation"):
                load_csv(p, modality_type="rna", feature_orientation="bad_value")


class TestLoadMultimodalCsv:
    def test_load_multimodal_csv(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write RNA file
            rna_path = Path(tmpdir) / "rna.tsv"
            rna_df = pd.DataFrame(
                rng.lognormal(3, 1, (15, 30)),
                index=[f"S{i}" for i in range(15)],
                columns=[f"G{i}" for i in range(30)],
            )
            rna_df.index.name = "sample_id"
            rna_df.to_csv(rna_path, sep="\t")

            # Write protein file
            prot_path = Path(tmpdir) / "protein.tsv"
            prot_df = pd.DataFrame(
                rng.normal(0, 1, (10, 20)),
                index=[f"S{i}" for i in range(10)],
                columns=[f"P{i}" for i in range(20)],
            )
            prot_df.index.name = "sample_id"
            prot_df.to_csv(prot_path, sep="\t")

            ds = load_multimodal_csv(
                {"rna": rna_path, "protein": prot_path},
                modality_types={"rna": "rna", "protein": "protein"},
                study_id="test_study",
            )
            assert "rna" in ds.modality_names
            assert "protein" in ds.modality_names
            assert ds._modalities["rna"].n_samples == 15
            assert ds._modalities["protein"].n_samples == 10


class TestBarcodeUtils:
    BARCODES = [
        "TCGA-02-0001-01A-01R-0177-13",
        "TCGA-02-0002-11A-01D-0178-05",
        "TCGA-AB-1234-01B-02R-0777-01",
    ]

    def test_barcode_truncation_participant(self):
        assert truncate_to_participant(self.BARCODES[0]) == "TCGA-02-0001"
        assert truncate_to_participant(self.BARCODES[1]) == "TCGA-02-0002"

    def test_barcode_truncation_sample(self):
        assert truncate_to_sample(self.BARCODES[0]) == "TCGA-02-0001-01A"

    def test_barcode_tumour(self):
        assert is_tumour("TCGA-02-0001-01A-01R-0177-13") is True
        assert is_tumour("TCGA-02-0002-11A-01D-0178-05") is False

    def test_barcode_normal(self):
        assert is_normal("TCGA-02-0002-11A-01D-0178-05") is True
        assert is_normal("TCGA-02-0001-01A-01R-0177-13") is False

    def test_parse_barcode(self):
        parsed = parse_barcode("TCGA-02-0001-01A-01R-0177-13")
        assert parsed["project"] == "TCGA"
        assert parsed["tss"] == "02"
        assert parsed["participant"] == "0001"
        assert parsed["sample"] == "01"
        assert parsed["vial"] == "A"

    def test_batch_parse(self):
        df = batch_parse(self.BARCODES)
        assert len(df) == 3
        assert "is_tumour" in df.columns
        assert "is_normal" in df.columns
        assert df.loc[0, "is_tumour"] == True
        assert df.loc[1, "is_normal"] == True

    def test_invalid_barcode_raises(self):
        with pytest.raises(ValueError, match="Not a valid TCGA barcode"):
            parse_barcode("NOT_A_TCGA_BARCODE")

    def test_truncate_to_participant_short_barcode_raises(self):
        with pytest.raises(ValueError):
            truncate_to_participant("TCGA-02")
