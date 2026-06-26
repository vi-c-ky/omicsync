# Changelog

All notable changes to omicsync will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-06-26

### Added
- `OmicsDataset`: chainable multi-omics container
- `OmicsModality` base class with subclasses for RNA, methylation, mutations, CNV, protein
- `SampleIndex`: TCGA barcode harmonisation and sample alignment
- Loaders: CSV/TSV, TCGA local files, GEO (via GEOparse), Open Targets GraphQL
- Normalisation: RNA (log1p, TPM, quantile, z-score, auto-detect), methylation (beta/M-value), CNV, mutations (binarise, TMB), protein (z-score, median centring)
- Integration: MOFA2 wrapper, sklearn `OmicsSyncTransformer`, simple/weighted/PCA concat
- Utilities: TCGA barcode parser, input validation, consistent logging
- Full test suite with synthetic data fixtures
- MIT licence
