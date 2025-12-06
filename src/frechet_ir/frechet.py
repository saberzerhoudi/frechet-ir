from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

# Try to import sklearn's robust covariance estimators
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN_COV = True
except ImportError:
    HAS_SKLEARN_COV = False


Array2D = NDArray[np.float64]
Array1D = NDArray[np.float64]


# Minimum samples for stable covariance estimation
MIN_SAMPLES_FOR_COV = 3


@dataclass(frozen=True)
class GaussianStats:
    """
    Empirical Gaussian parameters estimated from samples.

    Attributes
    ----------
    mean : (d,) array
        Sample mean vector.
    cov : (d, d) array
        Sample covariance matrix.
    """

    mean: Array1D
    cov: Array2D


def _as_2d_array(x: ArrayLike, *, name: str) -> Array2D:
    """Convert input to a float64 2D array of shape (n_samples, n_dims)."""
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")

    return arr


def _clip_outliers(data: Array2D, *, n_std: float = 5.0) -> Array2D:
    """
    Clip outliers that are more than n_std standard deviations from the mean.
    
    This prevents extreme values from distorting the covariance estimate.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero for constant features
    std = np.where(std > 1e-10, std, 1.0)
    
    # Compute z-scores and clip
    z_scores = (data - mean) / std
    z_clipped = np.clip(z_scores, -n_std, n_std)
    
    # Transform back
    return z_clipped * std + mean


def estimate_gaussian(
    x: ArrayLike,
    *,
    use_shrinkage: bool = True,
    clip_outliers: bool = True,
    outlier_std: float = 5.0,
) -> GaussianStats:
    """
    Estimate the mean and covariance of a multivariate Gaussian
    from a set of samples with robust estimation.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_dims)
        Samples from a distribution.
    use_shrinkage : bool
        If True, use Ledoit-Wolf shrinkage for more stable covariance
        estimation, especially with small sample sizes or high dimensions.
    clip_outliers : bool
        If True, clip outliers before estimation to reduce their influence.
    outlier_std : float
        Number of standard deviations beyond which to clip outliers.

    Returns
    -------
    GaussianStats
        Estimated mean and covariance.

    Notes
    -----
    - Uses Ledoit-Wolf shrinkage by default for robust covariance estimation.
    - Clips outliers at ±5σ by default to prevent extreme values from
      distorting the covariance structure.
    - For very small samples (< 3), returns a regularized identity covariance
      to ensure stability.
    """
    data = _as_2d_array(x, name="x")
    n_samples, n_dims = data.shape

    # Clip outliers if requested
    if clip_outliers and n_samples > MIN_SAMPLES_FOR_COV:
        data = _clip_outliers(data, n_std=outlier_std)

    mean = data.mean(axis=0)

    if n_samples < MIN_SAMPLES_FOR_COV:
        # Too few samples for reliable covariance estimation
        # Use regularized identity matrix scaled by data variance
        var = np.var(data, axis=0).mean() if n_samples > 1 else 1.0
        cov = np.eye(n_dims, dtype=np.float64) * max(var, 1e-6)
    elif use_shrinkage and HAS_SKLEARN_COV:
        # Use Ledoit-Wolf shrinkage for robust estimation
        lw = LedoitWolf()
        lw.fit(data)
        cov = lw.covariance_
    else:
        # Standard unbiased covariance
        cov = np.cov(data, rowvar=False, ddof=1)

    # Ensure covariance is 2D even if n_dims == 1
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    elif cov.ndim == 1:
        cov = cov.reshape(1, 1)

    return GaussianStats(mean=mean, cov=cov.astype(np.float64))


def frechet_gaussians(
    g1: GaussianStats,
    g2: GaussianStats,
    *,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Fréchet distance between two Gaussian distributions.

    This matches the Fréchet Inception Distance (FID) form typically used
    in generative modeling, and corresponds to the closed form of
    the 2-Wasserstein distance between Gaussians.

    Parameters
    ----------
    g1, g2 : GaussianStats
        Gaussian parameters (mean and covariance) for the two distributions.
    eps : float, optional
        Small diagonal regularizer added to covariances to improve
        numerical stability, by default 1e-6.

    Returns
    -------
    float
        Fréchet distance between the two Gaussians (non-negative).

    Raises
    ------
    ValueError
        If the mean or covariance dimensions do not agree.
    """
    mu1, sigma1 = g1.mean, g1.cov
    mu2, sigma2 = g2.mean, g2.cov

    if mu1.shape != mu2.shape:
        raise ValueError(
            f"Mean vectors must have same shape; got {mu1.shape} and {mu2.shape}"
        )
    if sigma1.shape != sigma2.shape:
        raise ValueError(
            f"Covariance matrices must have same shape; got {sigma1.shape} and {sigma2.shape}"
        )

    # Copy covariances and add small diagonal for numerical stability
    eps_eye = np.eye(sigma1.shape[0], dtype=np.float64) * eps
    sigma1_reg = sigma1.copy() + eps_eye
    sigma2_reg = sigma2.copy() + eps_eye

    # Matrix square root of sigma1 * sigma2
    cov_prod = sigma1_reg @ sigma2_reg
    covmean, info = linalg.sqrtm(cov_prod, disp=False)

    if not np.isfinite(covmean).all() or info != 0:
        # If sqrtm fails, try with stronger regularization
        stronger_eps = np.eye(sigma1.shape[0], dtype=np.float64) * (eps * 100)
        cov_prod = (sigma1.copy() + stronger_eps) @ (sigma2.copy() + stronger_eps)
        covmean = linalg.sqrtm(cov_prod)

    # Discard small imaginary components from numerical errors
    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag)) > 1e-3:
            # Significant imaginary part - possible numerical issues
            # Fall back to using only the real part
            pass
        covmean = covmean.real

    diff = mu1 - mu2
    diff_sq = float(diff @ diff)

    # Use regularized traces for consistency
    trace_sigma1 = float(np.trace(sigma1_reg))
    trace_sigma2 = float(np.trace(sigma2_reg))
    trace_sqrt = float(np.trace(covmean))

    fd = diff_sq + trace_sigma1 + trace_sigma2 - 2.0 * trace_sqrt
    # Numerical noise can create tiny negative values
    return float(max(fd, 0.0))


def frechet_from_samples(
    real: ArrayLike,
    simulated: ArrayLike,
    *,
    eps: float = 1e-6,
    use_shrinkage: bool = True,
    clip_outliers: bool = True,
) -> float:
    """
    Convenience wrapper: estimate Gaussians from samples and compute FD.

    This is the distribution-based Fréchet distance used for comparing
    sets of embeddings for real and simulated sessions.

    Parameters
    ----------
    real : array-like, shape (n_real, d)
        Embeddings for the "ideal" / real actions or sessions.
    simulated : array-like, shape (n_sim, d)
        Embeddings for the simulated actions or sessions.
    eps : float, optional
        Diagonal regularizer for numerical stability, by default 1e-6.
    use_shrinkage : bool
        If True, use Ledoit-Wolf shrinkage for covariance estimation.
    clip_outliers : bool
        If True, clip outliers before estimation.

    Returns
    -------
    float
        Fréchet distance between the two empirical distributions.
    """
    real_stats = estimate_gaussian(
        real,
        use_shrinkage=use_shrinkage,
        clip_outliers=clip_outliers,
    )
    sim_stats = estimate_gaussian(
        simulated,
        use_shrinkage=use_shrinkage,
        clip_outliers=clip_outliers,
    )
    return frechet_gaussians(real_stats, sim_stats, eps=eps)
