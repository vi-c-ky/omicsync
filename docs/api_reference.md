# API Reference

## omicsync.core.dataset

### OmicsDataset

```
OmicsDataset(modalities, study_id="unknown", metadata=None)
```

**Properties**
- `modality_names` → list[str]
- `common_samples` → pd.Index
- `sample_coverage` → pd.DataFrame
- `n_complete_cases` → int

**Methods** (all return `self` for chaining)
- `align_samples(strategy="intersection", fill_value=nan)`
- `normalize(per_modality=True)`
- `filter_features(min_variance=0.0, min_nonzero_frac=0.0)`
- `drop_modality(name)`
- `add_modality(name, modality)`
- `subset_samples(sample_ids)`
- `subset_cancer_types(types)`

**Export**
- `to_dataframe(modalities=None, fill_missing=nan)` → pd.DataFrame
- `to_dict()` → dict[str, pd.DataFrame]
- `to_mofa2()` → dict
- `to_tensor(dtype=None)` → torch.Tensor *(requires torch)*
- `to_anndata()` → anndata.AnnData *(requires anndata)*
- `describe()` → pd.DataFrame

---

## omicsync.core.modality

### OmicsModality

Base class. Subclasses: `RNAModality`, `MutationModality`, `MethylationModality`, `CNVModality`, `ProteinModality`.

```
OmicsModality(data, modality_type, source="unknown", metadata=None)
```

**Properties**: `data`, `n_samples`, `n_features`, `sample_ids`, `feature_ids`

**Methods**
- `filter_features(min_variance=0.0, min_nonzero_frac=0.0)` → self
- `filter_samples(sample_ids)` → self
- `describe()` → dict

---

## omicsync.loaders

| Function | Description |
|----------|-------------|
| `load_csv(path, modality_type, ...)` | Load a single CSV/TSV |
| `load_multimodal_csv(paths_dict, ...)` | Load multiple files into OmicsDataset |
| `load_tcga_files(data_dir, cancer_type, modalities)` | Load TCGA local files |
| `download_tcga_manifest(cancer_type, modalities, output_dir)` | Print GDC download instructions |
| `load_geo(accession, modality_type, ...)` | Load GEO series |
| `load_open_targets_targets(disease_ids, ...)` | Query Open Targets GraphQL |

---

## omicsync.normalisation

| Module | Functions |
|--------|-----------|
| `rna` | `log1p_normalise`, `tpm_to_log1p`, `counts_to_tpm`, `quantile_normalise`, `z_score`, `detect_and_normalise` |
| `methylation` | `beta_to_m`, `m_to_beta`, `clip_beta`, `detect_and_normalise` |
| `cnv` | `centre_diploid`, `log2_ratio`, `discretise` |
| `mutations` | `binarise`, `filter_by_consequence`, `compute_tmb` |
| `protein` | `z_score`, `median_centring` |

---

## omicsync.integration

| Class/Function | Description |
|----------------|-------------|
| `MOFA2Wrapper` | Train MOFA2 factor model on an OmicsDataset |
| `OmicsSyncTransformer` | sklearn-compatible transformer |
| `simple_concat(dataset, ...)` | Flat numpy concatenation |
| `weighted_concat(dataset, weights, ...)` | Weighted concatenation |
| `pca_concat(dataset, n_components, ...)` | PCA reduction then concat |

---

## omicsync.utils.barcode

| Function | Description |
|----------|-------------|
| `parse_barcode(barcode)` | Parse TCGA barcode into fields dict |
| `truncate_to_participant(barcode)` | Return TCGA-XX-XXXX |
| `truncate_to_sample(barcode)` | Return TCGA-XX-XXXX-XX |
| `is_tumour(barcode)` | Sample type 01-09 |
| `is_normal(barcode)` | Sample type 10-19 |
| `batch_parse(barcodes)` | Parse list to DataFrame |
