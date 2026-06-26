# omicsync

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/omicsync.svg)](https://pypi.org/project/omicsync/)

**A Python library for multi-omics data harmonisation.**

omicsync handles the tedious work of aligning sample IDs, normalising each modality consistently, and exporting to downstream tools so you can focus on biology, not data wrangling.

---

## Installation

```bash
pip install omicsync
```

With optional extras:

```bash
pip install "omicsync[mofa]"       # MOFA2 factor analysis
pip install "omicsync[geo]"        # GEO data loading
pip install "omicsync[anndata]"    # AnnData export
pip install "omicsync[torch]"      # PyTorch tensor export
pip install "omicsync[all]"        # Everything
```

---

## Quick Start

```python
import omicsync as oms
from omicsync.loaders.csv import load_multimodal_csv

# Load multiple modalities from CSV files
dataset = load_multimodal_csv({
    "rna":     "brca_rna.tsv",
    "protein": "brca_rppa.tsv",
    "cnv":     "brca_cnv.tsv",
}, study_id="TCGA-BRCA")

# Align, normalise, filter — all chainable
dataset.align_samples().normalize().filter_features(min_variance=0.01)

# Export to DataFrame or MOFA2
df = dataset.to_dataframe()          # samples × features, prefixed columns
mofa_input = dataset.to_mofa2()      # dict ready for mofapy2 entry_point
```

---

## Features

- **Sample harmonisation** — TCGA barcode parsing, fuzzy ID matching, coverage reporting
- **Per-modality normalisation** — auto-detection of count/TPM/M-value formats
- **Chainable API** — `dataset.align().normalize().filter_features()`
- **sklearn compatibility** — use `OmicsSyncTransformer` in a `Pipeline`
- **Multiple export formats** — DataFrame, dict, MOFA2, PyTorch tensor, AnnData
- **Open Targets integration** — query target-disease associations via GraphQL
- **Type hints throughout** — fully typed public API

---

## Supported Data Sources

| Source | Loader | Notes |
|--------|--------|-------|
| TCGA | `load_tcga_files()` | Local files; barcode auto-harmonisation |
| GEO | `load_geo()` | Via GEOparse; requires `omicsync[geo]` |
| CSV/TSV | `load_csv()` | Any tabular file |
| Open Targets | `load_open_targets_targets()` | GraphQL API v4 |

---

## Supported Modalities

| Modality | Class | Default Normalisation |
|----------|-------|-----------------------|
| RNA expression | `RNAModality` | `detect_and_normalise()` (log1p) |
| DNA methylation | `MethylationModality` | M→beta conversion + clip |
| Copy number | `CNVModality` | log2 ratio, clipped [-2, 2] |
| Somatic mutations | `MutationModality` | Binarise at threshold |
| Protein abundance | `ProteinModality` | Z-score per protein |

---

## Documentation

- [Quickstart guide](docs/quickstart.md)
- [API reference](docs/api_reference.md)
- [Tutorial: TCGA BRCA](docs/tutorials/tcga_brca.md)
- [Tutorial: Custom CSV data](docs/tutorials/custom_data.md)

---

## Citation

If you use omicsync in your research, please cite:

> Paterson V. (2026). *omicsync: A Python library for multi-omics data harmonisation*. GitHub: github.com/vi-c-ky/omicsync

---

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Run the test suite (`pytest tests/`)
5. Open a pull request

---

## License

MIT — see [LICENSE](LICENSE) for details.
