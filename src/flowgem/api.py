"""Public, class-based API for FLOWGEM.

Wraps the core sampler behind a scikit-learn-style fit/generate interface.
The user supplies a single incomplete array with NaN marking missing entries;
the observation mask and the particle initialisation are handled internally.
"""

import torch

from flowgem.core import sample_flowgem


class FlowGEM:
    """Generative sampler for missing-at-random (MAR) data via an approximate
    Wasserstein gradient flow.

    The user provides a single incomplete array in which missing entries are
    marked with NaN; the observation mask and the initial particle ensemble are
    constructed internally. The interface follows the scikit-learn-style
    fit / generate pattern.

    Parameters
    ----------
    T : int, default=1000
        Number of gradient-flow (time) steps.
    eta : float, default=0.01
        Step size of the forward-Euler update.
    sigma : float, list of float, or None, default=None
        Kernel bandwidth. If None, a median heuristic is used; a float fixes
        the bandwidth; a list triggers cross-validation over the candidates.
    init : {"mice", "sample"} or array-like, default="mice"
        How the initial particles are constructed. "mice" imputes with a
        MICE-style imputer; "sample" resamples observed values column-wise;
        an array is used directly as the initialisation.
    grad_tol : float, default=0.01
        Early-stopping tolerance on the relative gradient norm.
    min_iter : int, default=10
        Minimum number of steps before early stopping may trigger.
    dtype : torch.dtype, default=torch.float64
        Numerical precision used throughout.
    random_state : int or None, default=None
        Seed for the (stochastic) initialisation, for reproducibility.

    Attributes
    ----------
    mask_ : torch.Tensor
        Observation mask inferred in fit(); 1 marks an observed entry.
    X_obs_ : torch.Tensor
        Observed data with missing entries filled by zero.
    X0_ : torch.Tensor
        The initial particle ensemble constructed in fit().

    Examples
    --------
    >>> import torch
    >>> from flowgem import FlowGEM
    >>> X = torch.randn(100, 3)
    >>> X[0, 1] = float("nan")           # mark a missing value
    >>> model = FlowGEM(T=1000, random_state=0)
    >>> X_complete = model.fit(X).generate()
    """

    def __init__(self, T=1000, eta=0.01, sigma=None, init="mice",
                 grad_tol=0.01, min_iter=10, dtype=torch.float64,
                 random_state=None):
        self.T = T
        self.eta = eta
        self.sigma = sigma
        self.init = init
        self.grad_tol = grad_tol
        self.min_iter = min_iter
        self.dtype = dtype
        self.random_state = random_state

    def fit(self, X_incomplete):
        """Prepare the sampler from an incomplete dataset.

        Parameters
        ----------
        X_incomplete : array-like of shape (n_samples, d_features)
            Input data with missing entries marked as NaN.

        Returns
        -------
        self : FlowGEM
            The fitted instance (enables fit(...).generate() chaining).

        Raises
        ------
        ValueError
            If the input is not 2D, or contains no NaN (nothing missing).
        """
        # accept numpy arrays, lists, or tensors; work in the chosen dtype
        X = torch.as_tensor(X_incomplete, dtype=self.dtype)

        if X.ndim != 2:
            raise ValueError(
                f"X_incomplete must be 2D (n_samples, n_features), got {X.ndim}D"
            )
        if not torch.isnan(X).any():
            raise ValueError(
                "X_incomplete has no NaN values; nothing is marked as missing. "
                "Mark missing entries with NaN."
            )

        # observation mask: 1 = observed, 0 = missing  (code's convention)
        self.mask_ = (~torch.isnan(X)).to(self.dtype)

        # observed data with NaN filled by 0 as a placeholder
        # (the sampler uses only the observed columns per pattern, via the mask)
        self.X_obs_ = torch.nan_to_num(X, nan=0.0)

        # initial particles, built from the incomplete data per self.init
        self.X0_ = self._build_X0(X)

        return self

    def _build_X0(self, X):
        """Construct the initial particle ensemble from the incomplete data.

        X is the incomplete tensor (with NaN). Returns a complete X0 tensor.
        """
        init = self.init

        # case 1: user supplied their own initialisation array
        if not isinstance(init, str):
            X0 = torch.as_tensor(init, dtype=self.dtype)
            if X0.shape != X.shape:
                raise ValueError(
                    f"Supplied init array has shape {tuple(X0.shape)}, "
                    f"expected {tuple(X.shape)}"
                )
            return X0

        # case 2: MICE (default), via scikit-learn's IterativeImputer
        if init == "mice":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
            X0_np = imputer.fit_transform(X.cpu().numpy())
            return torch.as_tensor(X0_np, dtype=self.dtype)

        # case 3: column-wise resampling of observed values (paper's simulation init)
        if init == "sample":
            g = None
            if self.random_state is not None:
                g = torch.Generator().manual_seed(self.random_state)

            X0 = X.clone()
            _, d = X.shape
            for j in range(d):
                col = X[:, j]
                missing = torch.isnan(col)
                observed = col[~missing]
                n_missing = int(missing.sum())
                if n_missing > 0:
                    if observed.numel() == 0:
                        raise ValueError(f"Column {j} has no observed values to sample from")
                    idx = torch.randint(0, observed.numel(), (n_missing,), generator=g)
                    X0[missing, j] = observed[idx]
            return X0

        raise ValueError(
            f"Unknown init {init!r}; use 'mice', 'sample', or supply an array"
        )

    def generate(self, return_trajectory=False):
        """Generate a complete sample of the same size as the fitted data.

        Parameters
        ----------
        return_trajectory : bool
            If False (default), return only the final sample as a tensor of
            shape (n, d). If True, return the full list of per-step snapshots.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            The final sample of shape (n, d), or the full trajectory if
            return_trajectory is True.
            
        """
        if not hasattr(self, "X0_"):
            raise RuntimeError("Call fit(...) before generate().")

        # sigma handling: None -> heuristic (sigma_fix=None, sigma_vals=None)
        #                 float -> fixed
        #                 list -> cross-validation over candidates
        sigma_fix = None
        sigma_vals = None
        if isinstance(self.sigma, (int, float)):
            sigma_fix = float(self.sigma)
        elif self.sigma is not None:
            sigma_vals = self.sigma

        Xhats = sample_flowgem(
            self.X0_,
            self.X_obs_,
            self.mask_,
            T=self.T,
            eta=self.eta,
            grad_tol=self.grad_tol,
            min_iter=self.min_iter,
            sigma_fix=sigma_fix,
            sigma_vals=sigma_vals,
            dtype=self.dtype,
        )

        # return the full trajectory or just the final snapshot
        if return_trajectory:
            return [torch.as_tensor(x, dtype=self.dtype) for x in Xhats]
        return torch.as_tensor(Xhats[-1], dtype=self.dtype)