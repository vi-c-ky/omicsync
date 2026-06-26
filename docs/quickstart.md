# Quickstart Guide

## 1. Installation

```bash
pip install omicsync
# with all optional extras:
pip install "omicsync[all]"
```

---

## 2. Loading TCGA BRCA data from local files

Download TCGA BRCA files using the GDC portal (see `download_tcga_manifest()` for instructions), then:

```python
from omicsync.loaders.tcga import load_tcga_files

dataset = load_tcga_files(
    data_dir="/path/to/BRCA/data",
    cancer_type="BRCA",
    modalities=["rna", "mutations", "methylation", "cnv", "protein"],
)
print(dataset)
# OmicsDataset(study_id='TCGA-BRCA', modalities=[rna(800×20531), ...], n_common_samples=...)
```

---

## 3. Aligning samples

```python
# Keep only samples present in ALL modalities
dataset.align_samples(strategy="intersection")

# Or keep all samples, filling missing entries with NaN
dataset.align_samples(strategy="union", fill_value=float("nan"))
```

---

## 4. Normalising each modality

```python
# Auto-detect and apply appropriate normalisation per modality type
dataset.normalize()

# Or chain with alignment and filtering
(dataset
    .align_samples()
    .normalize()
    .filter_features(min_variance=0.01, min_nonzero_frac=0.05))
```

---

## 5. Exporting to MOFA2

```python
from omicsync.integration.mofa import MOFA2Wrapper

wrapper = MOFA2Wrapper(dataset, n_factors=15, convergence_mode="fast", seed=42)
wrapper.prepare().train(output_path="brca_mofa.hdf5")

factors = wrapper.get_factors()         # samples × factors DataFrame
weights = wrapper.get_weights()         # dict of modality → weights DataFrame
r2 = wrapper.get_variance_explained()   # factors × modalities
wrapper.plot_variance_explained()
```

---

## 6. Using in a scikit-learn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from omicsync.integration.sklearn_compat import OmicsSyncTransformer

pipe = Pipeline([
    ("omicsync", OmicsSyncTransformer(align=True, normalize=True)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# X = dataset, y = labels
pipe.fit(dataset, labels)
predictions = pipe.predict(test_dataset)
```

---

## 7. Loading custom CSV data

```python
from omicsync.loaders.csv import load_csv, load_multimodal_csv

# Single file
rna_mod = load_csv("my_rna.tsv", modality_type="rna", sample_col="sample_id")

# Multiple files at once
dataset = load_multimodal_csv({
    "rna":     "my_rna.tsv",
    "protein": "my_protein.csv",
    "cnv":     "my_cnv.tsv",
}, modality_types={"rna": "rna", "protein": "protein", "cnv": "cnv"})

# Export to flat DataFrame
df = dataset.to_dataframe()
# Columns: rna__GENE001, protein__PROT001, cnv__GENE001, ...
```
