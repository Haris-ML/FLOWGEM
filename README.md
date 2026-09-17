# FLOWGEM

**Generative sampling for missing data under MAR, via approximate Wasserstein gradient flows.**

[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

<!-- Uncomment once GitHub Actions CI is green (Week 8):
[![Tests](https://github.com/Haris-ML/FLOWGEM/actions/workflows/tests.yml/badge.svg)](https://github.com/Haris-ML/FLOWGEM/actions)
-->

## What is FLOWGEM?

FLOWGEM takes a dataset with missing values and generates a complete
dataset drawn from the same distribution. Unlike imputation methods that
fill each missing cell with a single "best guess", FLOWGEM recovers the
full data distribution — preserving variances and correlations — so that
downstream analyses (means, quantiles, correlations) remain unbiased.

It targets data that is **Missing at Random (MAR)**, including the hard
non-monotone case, and is nonparametric: it makes no assumption about the
shape of your data. Under the hood it uses an approximate Wasserstein
gradient flow that iteratively transports a particle ensemble toward the
target distribution.

> **Note:** FLOWGEM currently supports **continuous features only**.
> Support for categorical and mixed-type data is planned as future work.

## Installation

Install the released version from PyPI:

```bash
pip install flowgem
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/Haris-ML/FLOWGEM.git@package
```

## Quick start

```python
import numpy as np
from flowgem import FlowGEM

# Your data, with missing values marked as NaN
X = np.array([
    [1.2,    0.8,    3.1],
    [0.5,    np.nan, 2.7],
    [np.nan, 1.4,    2.9],
    [2.1,    0.3,    np.nan],
])

# Fit and generate a complete sample from the same distribution
model = FlowGEM(T=1000)
X_complete = model.fit(X).generate()
```

`X_complete` is a fully observed array (no NaNs) drawn from the estimated
data distribution.

## Features

- **Generative, not just imputation** — recovers the full data distribution,
  preserving variances and correlations.
- **Handles non-monotone MAR** — works with arbitrary, scattered missingness
  patterns.
- **Nonparametric** — no assumption about the shape of your data.
- **scikit-learn-style API** — familiar `fit` / `generate` interface.
- **Tuning-lean** — sensible defaults; a single heuristic sets the bandwidth.

## Key parameters

| Parameter | What it does | Default |
|-----------|--------------|---------|
| `T` | number of gradient-flow steps (more = better, slower) | `1000` |
| `eta` | step size η (smaller = finer, slower) | `0.01` |
| `sigma` | kernel bandwidth (`None` → chosen by heuristic) | `None` |

## Documentation

Full documentation is available at [Read the Docs](https://flowgem.readthedocs.io).

## Citation

FLOWGEM implements the method introduced in:

> Kremling, G., Näf, J., & Lederer, J. (2026). *Generative Modeling under
> Non-Monotonic MAR Missingness via Approximate Wasserstein Gradient Flows.*

If you use this package in your research, please cite the paper:

```bibtex
@article{kremling2026flowgem,
  title   = {Generative Modeling under Non-Monotonic MAR Missingness
             via Approximate Wasserstein Gradient Flows},
  author  = {Kremling, Gitte and N\"af, Jeffrey and Lederer, Johannes},
  journal = {arXiv preprint arXiv:2604.04567},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.
