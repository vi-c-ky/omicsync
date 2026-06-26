"""Tests for integration modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicsync.integration.concat import pca_concat, simple_concat, weighted_concat
from omicsync.integration.sklearn_compat import OmicsSyncTransformer


class TestSklearnTransformer:
    def test_sklearn_transformer_fit_transform(self, synthetic_dataset):
        ds = synthetic_dataset
        transformer = OmicsSyncTransformer(align=True, normalize=True)
        result = transformer.fit_transform(ds)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == ds.n_complete_cases

    def test_sklearn_transformer_feature_names(self, synthetic_dataset):
        ds = synthetic_dataset
        transformer = OmicsSyncTransformer(align=True, normalize=False)
        transformer.fit(ds)
        names = transformer.get_feature_names_out()
        assert len(names) > 0
        # Each name should have a modality prefix
        for name in names:
            assert "__" in name, f"Feature name {name!r} missing modality prefix."

    def test_pipeline_compatible(self, synthetic_dataset):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        ds = synthetic_dataset
        pipe = Pipeline([
            ("omicsync", OmicsSyncTransformer(align=True, normalize=True)),
            ("scaler", StandardScaler()),
        ])
        result = pipe.fit_transform(ds)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_transform_before_fit_raises(self, synthetic_dataset):
        transformer = OmicsSyncTransformer()
        with pytest.raises(RuntimeError, match="fit()"):
            transformer.transform(synthetic_dataset)

    def test_feature_names_before_fit_raises(self):
        transformer = OmicsSyncTransformer()
        with pytest.raises(RuntimeError, match="fit()"):
            transformer.get_feature_names_out()

    def test_set_output_pandas(self, synthetic_dataset):
        ds = synthetic_dataset
        transformer = OmicsSyncTransformer(align=True, normalize=False)
        transformer.fit(ds)
        transformer.set_output(transform="pandas")
        result = transformer.transform(ds)
        assert isinstance(result, pd.DataFrame)

    def test_wrong_input_type_raises(self, synthetic_dataset):
        transformer = OmicsSyncTransformer()
        transformer.fit(synthetic_dataset)
        with pytest.raises(TypeError, match="OmicsDataset"):
            transformer.transform(np.zeros((5, 10)))


class TestSimpleConcat:
    def test_simple_concat_shape(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        result = simple_concat(ds)
        total_features = sum(m.n_features for m in ds._modalities.values())
        assert result.shape == (ds.n_complete_cases, total_features)

    def test_simple_concat_fill_missing(self, synthetic_dataset):
        result = simple_concat(synthetic_dataset, fill_missing=0.0)
        assert not np.isnan(result).any()


class TestWeightedConcat:
    def test_weighted_concat_output(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        weights = {name: 1.0 for name in ds.modality_names}
        result = weighted_concat(ds, weights=weights)
        assert result.ndim == 2
        assert result.shape[0] > 0


class TestPcaConcat:
    def test_pca_concat_dimensions(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        n_comp = 5
        result = pca_concat(ds, n_components_per_modality=n_comp)
        n_modalities = len(ds.modality_names)
        assert result.shape[0] == ds.n_complete_cases
        assert result.shape[1] <= n_comp * n_modalities

    def test_pca_concat_no_nan(self, synthetic_dataset):
        ds = synthetic_dataset
        ds.align_samples()
        result = pca_concat(ds, n_components_per_modality=3)
        assert not np.isnan(result).any()
