# Tutorial: TCGA BRCA Multi-omics Analysis

This tutorial shows a complete workflow using TCGA breast cancer (BRCA) data.

## Download the data

```python
from omicsync.loaders.tcga import download_tcga_manifest

download_tcga_manifest(
    cancer_type="BRCA",
    modalities=["rna", "mutations", "methylation", "cnv", "protein"],
    output_dir="./BRCA_data",
)
```

Follow the printed instructions to download files from the GDC portal.

## Load and harmonise

```python
from omicsync.loaders.tcga import load_tcga_files

dataset = load_tcga_files(
    data_dir="./BRCA_data",
    cancer_type="BRCA",
    modalities=["rna", "mutations", "methylation", "cnv", "protein"],
)

print(dataset.sample_coverage)
print(f"Common samples: {dataset.n_complete_cases}")
```

## Normalise and filter

```python
(dataset
    .align_samples(strategy="intersection")
    .normalize()
    .filter_features(min_variance=0.01, min_nonzero_frac=0.05))
```

## Run MOFA2

```python
from omicsync.integration.mofa import MOFA2Wrapper

wrapper = MOFA2Wrapper(dataset, n_factors=20, seed=42)
wrapper.prepare().train(output_path="brca_mofa.hdf5")

factors = wrapper.get_factors()
wrapper.plot_variance_explained()
top_rna = wrapper.top_features(factor=1, modality="rna", n=20)
print(top_rna)
```

## Export for ML

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from omicsync.integration.sklearn_compat import OmicsSyncTransformer

# Assume `labels` is a pd.Series of clinical labels
pipe = Pipeline([
    ("omicsync", OmicsSyncTransformer()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(dataset, labels.reindex(dataset.common_samples))
```
