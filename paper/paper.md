---
title: 'omicsync: A Python library for multi-omics data harmonisation'
tags:
  - Python
  - bioinformatics
  - multi-omics
  - genomics
  - TCGA
  - data harmonisation
authors:
  - name: Victoria Paterson
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - index: 1
    name: School of Informatics, University of Edinburgh, United Kingdom
date: 1 July 2026
bibliography: paper.bib
---

# Summary

Multi-omics research integrates several layers of biological measurement, such as gene expression, DNA methylation, copy number variation, somatic mutations, and protein abundance, to build a more complete picture of disease biology than any single layer provides. Studies using datasets such as The Cancer Genome Atlas (TCGA) [@tomczak2015] routinely combine five or more modalities measured across hundreds or thousands of patient samples.

Before any analysis can begin, a substantial data preparation step is required. Sample identifiers must be aligned across modalities, which in TCGA data involves parsing barcodes that encode patient, sample, and aliquot information and vary in truncation level between different data types. Each modality requires different normalisation: RNA-seq count data is log-transformed, methylation beta values must be clipped or converted to M-values, copy number data is centred on the diploid baseline, and mutation data is typically binarised. Missing modalities must be handled consistently, and the resulting data must be packaged in a format compatible with downstream tools such as MOFA2 [@argelaguet2020] or scikit-learn [@pedregosa2011].

omicsync is a Python library that handles this preparation workflow through a chainable API. It loads data from TCGA local files, GEO series, generic CSV/TSV files, and the Open Targets Platform GraphQL API [@ochoa2023]. It aligns sample identifiers, normalises each modality using sensible defaults with options for custom normalisation, and exports to a unified DataFrame, a MOFA2-compatible dictionary, PyTorch tensors, or AnnData objects. An sklearn-compatible transformer enables omicsync to be embedded directly in scikit-learn Pipeline objects.

# Statement of need

Every multi-omics project begins with the same preprocessing workflow, yet there is no standard Python library that handles it end to end. Researchers either write bespoke code, adapt code from previous projects, or spend days on data wrangling before any biological analysis can begin. The TCGA barcode alignment problem alone routinely costs days of work: barcodes across RNA-seq, methylation, and mutation files use different truncation levels, meaning a naive join on sample identifiers silently drops samples or produces incorrect merges.

omicsync addresses this directly. The primary target audience is computational biologists and bioinformaticians who work with bulk multi-omics data, particularly TCGA data, and need to prepare it for factor analysis, machine learning, or survival analysis. The library was developed from the preprocessing workflow used in a pan-cancer immune exclusion study integrating five omics modalities across 11,060 samples from 31 cancer types [@paterson2026], and the API design reflects the practical requirements encountered in that work.

# State of the field

Several Python packages address aspects of multi-omics data management. OmicVerse [@zeng2024] is a broad analysis platform covering bulk, single-cell, and spatial transcriptomics, and is the most comprehensive tool currently available. However, it is a full analysis framework rather than a focused harmonisation library: it bundles dozens of analysis methods and its preprocessing functionality is not designed to be used independently as a modular component in other pipelines. OpenOmics [@tran2021] addresses multi-omics integration and database retrieval but has seen limited maintenance since 2021 and does not support TCGA barcode harmonisation or sklearn compatibility. Omilayers [@vasileiadis2025] focuses on database storage for multi-omics data using SQLite and DuckDB, solving a different problem from sample alignment and normalisation.

omicsync occupies a narrower, more focused position: it solves the data preparation problem and produces clean, aligned output that can be passed to any downstream tool, including OmicVerse, MOFA2, or scikit-learn. The two are complementary rather than competing.

# Software design

The central design decision was to separate the harmonisation problem from the analysis problem. Many existing tools bundle preprocessing with analysis methods, which makes the preprocessing logic difficult to reuse or test independently. omicsync exposes a single user-facing object, `OmicsDataset`, that holds aligned modality data and provides chainable methods for alignment, normalisation, filtering, and export. Each modality is represented by a typed subclass of `OmicsModality` that performs modality-specific validation at construction time, so type errors surface early rather than propagating through an analysis.

TCGA barcode harmonisation is handled by a dedicated `SampleIndex` class that parses barcodes to participant, sample, or aliquot level and implements fuzzy matching for common formatting differences such as dash-versus-dot separators. Coverage reporting is built in: after alignment, `OmicsDataset.sample_coverage` returns a DataFrame showing which modalities are available for each sample, giving researchers a clear view of missingness before they proceed.

Normalisation methods are implemented per modality with auto-detection where feasible. The RNA normalisation module uses value distribution heuristics to distinguish raw counts from TPM-normalised data and applies the appropriate transform, logging which transform was applied. Methylation normalisation detects whether values are beta values or M-values based on their range and converts accordingly. These defaults can be overridden at any point.

The sklearn compatibility layer implements `BaseEstimator` and `TransformerMixin`, allowing `OmicsDataset` to be used as the first step in a scikit-learn `Pipeline`. This was a deliberate design choice to lower the barrier to combining multi-omics preprocessing with standard machine learning workflows without requiring users to learn a new API for the downstream modelling step.

Optional dependencies (mofapy2, GEOparse, torch, anndata) are imported lazily and raise `ImportError` with installation instructions if they are not available. This keeps the core library lightweight while providing a clear extension path.

# Research impact statement

omicsync was developed from the preprocessing pipeline used in a pan-cancer immune exclusion study analysing 11,060 patient samples across 31 cancer types [@paterson2026], currently under review. The library encodes the practical solutions to TCGA barcode alignment, missing modality handling, and per-modality normalisation developed in that work, making them available to other researchers without requiring them to rediscover the same solutions independently.

The library is published on the Python Package Index at [https://pypi.org/project/omicsync/](https://pypi.org/project/omicsync/) and the source code is available at [https://github.com/vi-c-ky/omicsync](https://github.com/vi-c-ky/omicsync) under the MIT licence. An automated test suite covering core functionality is included and runs on Python 3.9, 3.10, and 3.11 via GitHub Actions continuous integration.

# AI usage disclosure

Claude (Anthropic, claude-sonnet-4-6) was used during development of omicsync and preparation of this paper. In the software: Claude assisted with code generation for several modules including the TCGA barcode parser, normalisation auto-detection logic, and the sklearn compatibility layer. In this paper: Claude assisted with drafting and copy-editing of the text. In all cases, the author reviewed, validated, and where necessary rewrote AI-assisted outputs. All design decisions, the overall architecture of the library, and the scientific framing of the paper reflect the author's own judgement. The author takes full responsibility for the accuracy and completeness of the submitted materials.

# Acknowledgements

The author thanks Ian Simpson (University of Edinburgh) for arXiv endorsement and feedback on related multi-omics work. No financial support was received for the development of omicsync.

# References
