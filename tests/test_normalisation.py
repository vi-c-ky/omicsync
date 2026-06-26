"""Tests for normalisation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicsync.normalisation import cnv, methylation, mutations, protein, rna


def _make_df(data, n_samples=20, n_features=30, seed=0) -> pd.DataFrame:
    if data is None:
        rng = np.random.default_rng(seed)
        data = rng.lognormal(3, 1, (n_samples, n_features))
    return pd.DataFrame(
        data,
        index=[f"S{i}" for i in range(len(data))],
        columns=[f"F{i}" for i in range(len(data[0]))],
    )


class TestRNANorm:
    def test_log1p_normalise_nonnegative(self):
        df = _make_df(None)
        result = rna.log1p_normalise(df)
        assert result.values.min() >= 0.0
        assert result.shape == df.shape

    def test_log1p_normalise_correct_values(self):
        data = np.array([[0.0, 1.0, np.e - 1]])
        df = _make_df(data)
        result = rna.log1p_normalise(df)
        assert np.isclose(result.iloc[0, 0], 0.0)
        assert np.isclose(result.iloc[0, 1], np.log(2))
        assert np.isclose(result.iloc[0, 2], 1.0, atol=1e-5)

    def test_detect_rna_type_counts(self):
        rng = np.random.default_rng(0)
        counts_df = _make_df(rng.integers(100, 10000, size=(20, 30)).astype(float))
        result = rna.detect_and_normalise(counts_df)
        assert result.values.max() < 15, "log1p(counts) should be small"

    def test_detect_rna_type_tpm(self):
        rng = np.random.default_rng(0)
        tpm_df = _make_df(rng.uniform(0.1, 50, size=(20, 30)))
        result = rna.detect_and_normalise(tpm_df)
        assert result.values.max() < 10, "log1p(TPM) should be small"

    def test_z_score_feature_mean_zero(self):
        rng = np.random.default_rng(0)
        df = _make_df(rng.normal(5, 2, (50, 20)))
        result = rna.z_score(df, axis=0)
        col_means = result.values.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=1e-10)

    def test_quantile_normalise_equal_distributions(self):
        rng = np.random.default_rng(0)
        df = _make_df(rng.lognormal(3, 1, (10, 50)))
        result = rna.quantile_normalise(df)
        # Each sample should have the same sorted values
        sorted_rows = np.sort(result.values, axis=1)
        assert np.allclose(sorted_rows[0], sorted_rows[1], atol=1e-10)

    def test_counts_to_tpm_requires_gene_lengths(self):
        rng = np.random.default_rng(0)
        df = _make_df(rng.integers(0, 1000, size=(5, 10)).astype(float))
        lengths = pd.Series(
            rng.integers(500, 5000, size=10),
            index=df.columns,
        )
        result = rna.counts_to_tpm(df, lengths)
        # TPM should sum to ~1e6 per sample
        row_sums = result.values.sum(axis=1)
        assert np.allclose(row_sums, 1e6, rtol=0.01)


class TestMethylationNorm:
    def test_beta_to_m_and_back(self):
        rng = np.random.default_rng(0)
        beta = rng.uniform(0.05, 0.95, (20, 50))
        df = pd.DataFrame(beta, index=[f"S{i}" for i in range(20)],
                          columns=[f"CpG{i}" for i in range(50)])
        m_df = methylation.beta_to_m(df)
        restored = methylation.m_to_beta(m_df)
        assert np.allclose(df.values, restored.values, atol=1e-5)

    def test_clip_beta_range(self):
        df = pd.DataFrame([[0.0, 0.5, 1.0]], columns=["A", "B", "C"],
                          index=["S0"])
        result = methylation.clip_beta(df, low=0.001, high=0.999)
        assert result.values.min() >= 0.001
        assert result.values.max() <= 0.999

    def test_detect_mvalue_converts_to_beta(self):
        rng = np.random.default_rng(0)
        m_vals = rng.normal(0, 2, (20, 50))
        df = pd.DataFrame(m_vals, index=[f"S{i}" for i in range(20)],
                          columns=[f"C{i}" for i in range(50)])
        result = methylation.detect_and_normalise(df)
        assert result.values.min() >= 0.001
        assert result.values.max() <= 0.999

    def test_detect_beta_clips(self):
        rng = np.random.default_rng(0)
        beta = rng.uniform(0.001, 0.999, (10, 20))
        df = pd.DataFrame(beta, index=[f"S{i}" for i in range(10)],
                          columns=[f"C{i}" for i in range(20)])
        result = methylation.detect_and_normalise(df)
        assert result.values.min() >= 0.001
        assert result.values.max() <= 0.999

    def test_beta_to_m_raises_on_out_of_range(self):
        df = pd.DataFrame([[1.5, 0.5]], columns=["A", "B"], index=["S0"])
        with pytest.raises(ValueError, match="beta values must be in"):
            methylation.beta_to_m(df)


class TestCNVNorm:
    def test_cnv_centre_diploid(self):
        df = pd.DataFrame([[2.0, 4.0, 1.0]], columns=["A", "B", "C"], index=["S0"])
        result = cnv.centre_diploid(df, diploid=2.0)
        assert np.allclose(result.values, [[0.0, 2.0, -1.0]])

    def test_log2_ratio(self):
        df = pd.DataFrame([[0.0, 1.0, 2.0]], columns=["A", "B", "C"], index=["S0"])
        result = cnv.log2_ratio(df, pseudo=1.0)
        # log2(0+1)=0, log2(1+1)=1, log2(2+1)=1.585
        assert np.isclose(result.iloc[0, 0], 0.0)
        assert np.isclose(result.iloc[0, 1], 1.0)

    def test_discretise_states(self):
        df = pd.DataFrame(
            [[-1.5, -0.5, 0.0, 0.5, 1.5]],
            columns=list("ABCDE"),
            index=["S0"],
        )
        result = cnv.discretise(df, thresholds=(-1.0, -0.3, 0.3, 1.0))
        assert result.iloc[0, 0] == -2.0
        assert result.iloc[0, 1] == -1.0
        assert result.iloc[0, 2] == 0.0
        assert result.iloc[0, 3] == 1.0
        assert result.iloc[0, 4] == 2.0


class TestMutationsNorm:
    def test_binarise_mutations(self):
        df = pd.DataFrame(
            [[0, 1, 2, 0.5]],
            columns=list("ABCD"),
            index=["S0"],
        )
        result = mutations.binarise(df, threshold=0)
        assert result.iloc[0, 0] == 0.0
        assert result.iloc[0, 1] == 1.0
        assert result.iloc[0, 2] == 1.0
        assert result.iloc[0, 3] == 1.0

    def test_compute_tmb(self):
        df = pd.DataFrame(
            [[1, 0, 1], [0, 0, 0], [1, 1, 1]],
            columns=["A", "B", "C"],
            index=["S0", "S1", "S2"],
        )
        tmb = mutations.compute_tmb(df)
        assert tmb["S0"] == 2
        assert tmb["S1"] == 0
        assert tmb["S2"] == 3


class TestProteinNorm:
    def test_z_score_protein(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(5, 3, (50, 20)),
            index=[f"S{i}" for i in range(50)],
            columns=[f"P{i}" for i in range(20)],
        )
        result = protein.z_score(df)
        col_means = result.values.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=1e-10)

    def test_median_centring(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(10, 2, (30, 15)),
            index=[f"S{i}" for i in range(30)],
            columns=[f"P{i}" for i in range(15)],
        )
        result = protein.median_centring(df)
        col_medians = np.median(result.values, axis=0)
        assert np.allclose(col_medians, 0.0, atol=1e-10)
