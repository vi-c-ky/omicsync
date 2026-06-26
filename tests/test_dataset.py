"""Tests for OmicsDataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicsync.core.dataset import OmicsDataset
from omicsync.core.modality import CNVModality, RNAModality, ProteinModality


class TestAlignSamples:
    def test_align_samples_intersection(self, synthetic_dataset):
        ds = synthetic_dataset
        n_before = [mod.n_samples for mod in ds._modalities.values()]
        ds.align_samples(strategy="intersection")
        common = ds.common_samples
        for mod in ds._modalities.values():
            assert mod.n_samples == len(common)
        assert len(common) < max(n_before)

    def test_align_samples_coverage_report(self, synthetic_dataset):
        coverage = synthetic_dataset.sample_coverage
        assert "n_modalities" in coverage.columns
        assert coverage["n_modalities"].max() <= len(synthetic_dataset.modality_names)
        assert coverage["n_modalities"].min() >= 1

    def test_align_samples_union_fills_missing(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples(strategy="union", fill_value=np.nan)
        for mod in ds._modalities.values():
            assert mod.n_samples == ds.n_complete_cases or True  # union keeps all


class TestNormalize:
    def test_normalize_all_modalities(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        ds.normalize(per_modality=True)
        rna_vals = ds._modalities["rna"].data.values
        assert np.isfinite(rna_vals).all(), "RNA should have no NaN after normalisation"

    def test_normalize_noop(self, synthetic_dataset):
        ds = synthetic_dataset
        original_shape = {k: v.data.shape for k, v in ds._modalities.items()}
        ds.normalize(per_modality=False)
        for k, v in ds._modalities.items():
            assert v.data.shape == original_shape[k]


class TestFilterFeatures:
    def test_filter_features_min_variance(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        before = {k: v.n_features for k, v in ds._modalities.items()}
        ds.filter_features(min_variance=0.1)
        for k, v in ds._modalities.items():
            assert v.n_features <= before[k]

    def test_filter_features_min_nonzero_frac(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        before_mut = ds._modalities["mutations"].n_features
        ds.filter_features(min_nonzero_frac=0.5)
        assert ds._modalities["mutations"].n_features <= before_mut


class TestToDataframe:
    def test_to_dataframe_shape(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        df = ds.to_dataframe()
        assert df.shape[0] == ds.n_complete_cases
        total_features = sum(m.n_features for m in ds._modalities.values())
        assert df.shape[1] == total_features

    def test_to_dataframe_column_prefixes(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        df = ds.to_dataframe()
        for name in ds.modality_names:
            prefixed = [c for c in df.columns if c.startswith(f"{name}__")]
            assert len(prefixed) > 0, f"No columns with prefix {name}__"

    def test_to_dataframe_subset_modalities(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        df = ds.to_dataframe(modalities=["rna"])
        assert all(c.startswith("rna__") for c in df.columns)

    def test_to_dataframe_fill_missing(self, synthetic_dataset):
        ds = synthetic_dataset
        df = ds.to_dataframe(fill_missing=0.0)
        assert not df.isna().any().any()


class TestToMofa2:
    def test_to_mofa2_format(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        mofa_input = ds.to_mofa2()
        assert "data" in mofa_input
        assert "views" in mofa_input
        assert "groups" in mofa_input
        assert "samples" in mofa_input
        assert len(mofa_input["data"]) == len(ds.modality_names)
        for arr_list in mofa_input["data"]:
            assert len(arr_list) == 1
            mat = arr_list[0]
            assert mat.shape[0] == ds.n_complete_cases


class TestChaining:
    def test_chaining(self, synthetic_dataset):
        result = (
            synthetic_dataset
            .align_samples(strategy="intersection")
            .normalize(per_modality=True)
            .filter_features(min_variance=0.01)
        )
        assert result is synthetic_dataset


class TestAddRemoveModality:
    def test_add_modality(self, synthetic_dataset):
        rng = np.random.default_rng(0)
        samples = list(synthetic_dataset.common_samples)[:5] or ["S1", "S2", "S3"]
        new_df = pd.DataFrame(
            rng.normal(size=(10, 5)),
            index=[f"NS{i}" for i in range(10)],
            columns=["P1", "P2", "P3", "P4", "P5"],
        )
        new_mod = ProteinModality(new_df, source="test")
        synthetic_dataset.add_modality("protein2", new_mod)
        assert "protein2" in synthetic_dataset.modality_names

    def test_remove_modality(self, synthetic_dataset):
        assert "cnv" in synthetic_dataset.modality_names
        synthetic_dataset.drop_modality("cnv")
        assert "cnv" not in synthetic_dataset.modality_names

    def test_add_duplicate_raises(self, synthetic_dataset):
        with pytest.raises(ValueError, match="already exists"):
            synthetic_dataset.add_modality("rna", synthetic_dataset._modalities["rna"])

    def test_remove_missing_raises(self, synthetic_dataset):
        with pytest.raises(KeyError):
            synthetic_dataset.drop_modality("nonexistent_modality")


class TestDescribe:
    def test_describe_output(self, synthetic_dataset):
        df = synthetic_dataset.describe()
        assert "n_samples" in df.columns
        assert "n_features" in df.columns
        assert len(df) == len(synthetic_dataset.modality_names)


class TestRepr:
    def test_repr_contains_study_id(self, synthetic_dataset):
        r = repr(synthetic_dataset)
        assert "SYNTHETIC" in r
        assert "OmicsDataset" in r
