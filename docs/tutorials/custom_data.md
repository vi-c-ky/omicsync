# Tutorial: Using Custom CSV Data

## Single modality

```python
from omicsync.loaders.csv import load_csv

# Samples as rows (default)
rna_mod = load_csv(
    "my_rna.tsv",
    modality_type="rna",
    sample_col="sample_id",   # column containing sample IDs
)

# Samples as columns (genes as rows)
rna_mod = load_csv(
    "my_rna_genes_as_rows.tsv",
    modality_type="rna",
    feature_orientation="samples_as_columns",
    index_col=0,
)
```

## Multiple modalities

```python
from omicsync.loaders.csv import load_multimodal_csv

dataset = load_multimodal_csv(
    paths_dict={
        "rna":     "data/rna.tsv",
        "protein": "data/protein.csv",
        "cnv":     "data/cnv.tsv",
    },
    modality_types={
        "rna":     "rna",
        "protein": "protein",
        "cnv":     "cnv",
    },
    study_id="my_study",
)

dataset.align_samples().normalize()
df = dataset.to_dataframe()
print(df.shape)  # (n_samples, n_features_total)
```

## Creating modalities programmatically

```python
import numpy as np
import pandas as pd
from omicsync import OmicsDataset, RNAModality, ProteinModality

rna_df = pd.DataFrame(
    np.random.lognormal(3, 1, (50, 100)),
    index=[f"SAMPLE_{i}" for i in range(50)],
    columns=[f"GENE{i}" for i in range(100)],
)
prot_df = pd.DataFrame(
    np.random.normal(0, 1, (50, 30)),
    index=[f"SAMPLE_{i}" for i in range(50)],
    columns=[f"PROT{i}" for i in range(30)],
)

dataset = OmicsDataset(
    modalities={
        "rna":     RNAModality(rna_df, source="custom"),
        "protein": ProteinModality(prot_df, source="custom"),
    },
    study_id="my_study",
)

print(dataset)
```
