# FLOWGEM

**Generative sampling for missing data under MAR, via approximate Wasserstein gradient flows.**

[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/Haris-ML/FLOWGEM)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/Haris-ML/FLOWGEM/blob/main/LICENSE)

[![Tests](https://github.com/Haris-ML/FLOWGEM/actions/workflows/ci.yml/badge.svg)](https://github.com/Haris-ML/FLOWGEM/actions)

## What is FLOWGEM?

FLOWGEM recieves a dataset with missing values and generates a complete dataset sampled from the same distribution. It recovers the complete distribution of the data while preserving variances and correlations which keeps downstream analyses like means, quantiles, and correlations unbiased.

It targets data which is **Missing at Random (MAR)**, including non-monotone MAR. It is nonparametric and makes no assumption about the shape of your data. Under the hood it uses approximate Wasserstein gradient flow which iteratively transports a particle ensemble towards target distribution.

> **Note:** FLOWGEM currently supports **continuous features only**.
> Support for categorical and mixed type data is future work.

## Installation

Install the released version from PyPI:

```bash
pip install flowgem
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/Haris-ML/FLOWGEM.git
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

`X_complete` is a fully observed array (without missing value) which is drawn from estimated data distribution.

## Features

- **Generative**:- generate entire dataset from the full data distribution while preserving variances and correlations.
- **Handles non-monotone MAR**:- works with arbitrary, scattered missingness patterns.
- **Nonparametric** :- No assumption about the shape of your data.
- **scikit-learn-style API**:- familiar `fit` / `generate` interface.
- **Tuning-lean**:- sensible defaults; bandwidth is set from a single heuristic

## Key parameters

| Parameter | What it does | Default |
|-----------|--------------|---------|
| `T` | number of gradient-flow steps (more = better, slower) | `1000` |
| `eta` | step size η (smaller = finer, slower) | `0.01` |
| `sigma` | kernel bandwidth (`None` → chosen by heuristic) | `None` |

## Documentation

Comprehensive and detailed documentation is available at [Read the Docs](https://flowgem.readthedocs.io).

## Citation

The FLOWGEM method is based on:

> Kremling, G., Näf, J., & Lederer, J. (2026). *Generative Modeling under
> Non-Monotonic MAR Missingness via Approximate Wasserstein Gradient Flows.*

While using the package in course of your research, please cite this paper:

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

This project is licensed under the MIT License — see the [LICENSE](https://github.com/Haris-ML/FLOWGEM/blob/main/LICENSE) file.
