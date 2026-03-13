# f2a — File to Analysis

> A Python library that automatically performs descriptive statistical analysis and visualization from data sources

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Installation

```bash
pip install f2a

# With HuggingFace dataset support
pip install f2a[hf]

# All features
pip install f2a[all]
```

## Quick Start

```python
import f2a

# Analyze a local CSV file
report = f2a.analyze("data/sales.csv")
report.show()  # Print summary to console

# Analyze a Hugging Face dataset
report = f2a.analyze("hf://imdb")
report.show()

# Access detailed results
report.stats.summary        # Summary statistics DataFrame
report.stats.correlation     # Correlation matrix
report.viz.plot_distributions()  # Distribution plots
```

## Supported Formats

| Format | Extensions | Extra Install |
|---|---|---|
| CSV / TSV | `.csv`, `.tsv` | — |
| JSON / JSONL | `.json`, `.jsonl` | — |
| Parquet | `.parquet` | `pip install f2a[parquet]` |
| Excel | `.xlsx`, `.xls` | `pip install f2a[excel]` |
| SQLite | `.db`, `.sqlite3` | — |
| Stata | `.dta` | — |
| XML / HTML | `.xml`, `.html` | — |
| HuggingFace | `hf://dataset_name` | `pip install f2a[hf]` |

## Analysis Features

- **Descriptive Statistics**: Mean, median, standard deviation, quantiles, mode, etc.
- **Distribution Analysis**: Skewness, kurtosis, normality tests
- **Correlation Analysis**: Pearson, Spearman, Cramér's V
- **Missing Data Analysis**: Missing ratio, pattern analysis
- **Visualization**: Histograms, boxplots, correlation heatmaps, missing data matrix

## Development

```bash
git clone https://github.com/CocoRoF/f2a.git
cd f2a
pip install -e ".[dev]"
pytest
```

## License

MIT License — See [LICENSE](LICENSE) for details.
