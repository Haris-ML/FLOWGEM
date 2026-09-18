# Usage guide

This guide covers how to use FLOWGEM in practice: basic use, working with
missing data, choosing hyperparameters, initialisation, and inspecting the
gradient flow. For a quick overview, see the [README](../README.md).

## Basic usage

FLOWGEM takes an array with missing values (marked as `NaN`) and returns a
complete array drawn from the same distribution.

```python
import numpy as np
from flowgem import FlowGEM

# a small correlated dataset
rng = np.random.default_rng(0)
X = rng.multivariate_normal(
    mean=[0, 0, 0],
    cov=[[1.0, 0.7, 0.0],
         [0.7, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
    size=500,
)

# keep a clean copy so we can compare the recovered structure
X_true = X.copy()

# introduce some missing values (NaN)
X[rng.random(500) < 0.2, 1] = np.nan   # ~20% missing in column 1

# generate a complete sample
model = FlowGEM(T=1000, random_state=0)
X_complete = model.fit(X).generate()

print(X_complete.shape)             # (500, 3)
print(np.isnan(X_complete).any())   # False
print(round(np.corrcoef(X_true[:, 0], X_true[:, 1])[0, 1], 2))       # 0.68
print(round(np.corrcoef(X_complete[:, 0], X_complete[:, 1])[0, 1], 2))  # 0.69
```

The returned `X_complete` is a NumPy array of the same shape as `X`, with
every `NaN` replaced. Crucially, the recovered sample preserves the
**correlation structure** of the data — not just the column means — which is
what makes FLOWGEM generative rather than a simple imputation.

## Working with missing data

FLOWGEM treats any `NaN` in the input as a missing value. You don't build a
mask yourself — the missingness pattern is inferred internally.

```python
import numpy as np
from flowgem import FlowGEM

X = np.array([
    [1.2,    0.8,    3.1],
    [0.5,    np.nan, 2.7],
    [np.nan, 1.4,    2.9],
    [2.1,    0.3,    np.nan],
])

X_complete = FlowGEM(T=1000, random_state=0).fit(X).generate()
```

Each row can have its own missingness pattern — including scattered,
non-monotone patterns, which FLOWGEM is specifically designed to handle.

**Requirements for the input:**

- Values must be **continuous** (floating-point). *FLOWGEM currently supports
  continuous features only; categorical/mixed-type data is future work.*
- Every column must have **at least some observed values** — a fully missing
  column cannot be recovered and raises an error.
- Missing entries are marked with `np.nan`.

## Choosing hyperparameters

FLOWGEM has three main hyperparameters. Sensible defaults work across a wide
range of datasets, so you rarely need to tune them.

### `T` — number of steps (default: 1000)

The number of gradient-flow iterations. More steps let the particles settle
closer to the target distribution.

- **Larger `T`** → better results, but slower.
- There is **no accuracy trade-off**: a larger `T` (with a smaller `eta`) only
  helps — the only cost is computation time.
- Early stopping halts automatically once the particles stop moving, so a
  large `T` is safe.

### `eta` — step size η (default: 0.01)

How far the particles move at each step.

- **Smaller `eta`** → finer, more stable convergence, but needs more steps.
- If `eta` is too large, the particles can overshoot; FLOWGEM detects this and
  halves `eta` automatically.
- Pair a smaller `eta` with a larger `T`.

### `sigma` — kernel bandwidth (default: `None`)

Controls how "local" the velocity estimate is around each point.

- **`None`** (default) → chosen automatically by a median heuristic. Recommended.
- **A float** → use a fixed bandwidth.
- **A list** → cross-validate over the candidates and pick the best.
- Unlike `T` and `eta`, `sigma` needs care: both too-small and too-large
  values degrade results — so prefer the default heuristic.

Other parameters (`random_state`, `grad_tol`, `min_iter`, `dtype`) are
documented in the API reference.

## Initialization

FLOWGEM starts the gradient flow from an initial complete ensemble (`X0`),
built by filling in the missing values. You control this with the `init`
argument.

### `init="mice"` (default)

Fills the missing values using MICE (via `hyperimpute`), giving a strong,
data-adaptive starting point. Recommended for most use.

```python
model = FlowGEM(init="mice").fit(X).generate()
```

### `init="sample"`

Fills each missing entry by resampling from the observed values in the same
column. Lighter and faster than MICE, with no extra dependencies.

```python
model = FlowGEM(init="sample").fit(X).generate()
```

### Supplying your own array

You can also pass a complete array (the same shape as `X`) to start from —
useful if you already have an imputation you trust.

```python
model = FlowGEM(init=my_X0).fit(X).generate()
```

The gradient flow refines whatever starting point you give it, so the final
result is not very sensitive to the initialisation — but a better `X0` can
mean faster convergence.

## Inspecting the flow (advanced)

By default, `generate()` returns only the final sample. Pass
`return_trajectory=True` to get the full sequence of snapshots — one NumPy
array per gradient-flow step — instead:

```python
model = FlowGEM(T=1000, random_state=0).fit(X)

trajectory = model.generate(return_trajectory=True)
# trajectory is a list of arrays, one per step
# trajectory[-1] is the final sample (same as generate())

print(len(trajectory))        # number of steps taken (may be < T if early-stopped)
print(trajectory[0].shape)    # (n, d)
```

This is useful for:

- **Visualising convergence** — watching the particle ensemble move toward the
  target distribution over the iterations (this is how the flow animation is
  produced).
- **Diagnostics** — checking how quickly the flow settles, or whether early
  stopping kicked in before `T` steps.

<!-- Animation to embed here once Sphinx is set up (Day 4-5):
![FLOWGEM flow](_static/flow.gif)
-->
