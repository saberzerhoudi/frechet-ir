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


def _matrix_inv_sqrt(A: Array2D, eps: float = 1e-6) -> Array2D:
    """
    Compute the inverse square root of a symmetric positive definite matrix.
    
    Uses eigendecomposition for numerical stability.
    """
    A_reg = A + np.eye(A.shape[0], dtype=np.float64) * eps
    
    # Eigendecomposition for symmetric matrices
    eigvals, eigvecs = np.linalg.eigh(A_reg)
    eigvals = np.maximum(eigvals, eps)  # Ensure positive
    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
    
    return eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T


def _matrix_sqrt(A: Array2D, eps: float = 1e-6) -> Array2D:
    """
    Compute the square root of a symmetric positive definite matrix.
    
    Falls back to eigendecomposition if scipy.linalg.sqrtm fails.
    """
    A_reg = A + np.eye(A.shape[0], dtype=np.float64) * eps
    sqrtA, info = linalg.sqrtm(A_reg, disp=False)
    
    if not np.isfinite(sqrtA).all() or info != 0:
        # Fallback: eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(A_reg)
        eigvals = np.maximum(eigvals, eps)
        sqrtA = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    if np.iscomplexobj(sqrtA):
        sqrtA = sqrtA.real
    
    return sqrtA


def whitened_frechet_gaussians(
    g_real: GaussianStats,
    g_sim: GaussianStats,
    *,
    eps: float = 1e-6,
) -> float:
    """
    Compute the whitened Fréchet distance between two Gaussian distributions.
    
    This transforms to a coordinate system where the real distribution has
    identity covariance, making the distance scale-invariant and comparable
    across different embedders.
    
    The whitened FD is computed as:
        μ̃ = Σ_real^{-1/2} (μ_sim - μ_real)
        Σ̃ = Σ_real^{-1/2} Σ_sim Σ_real^{-1/2}
        FD_whitened = ||μ̃||² + tr(Σ̃ + I - 2 Σ̃^{1/2})
    
    Parameters
    ----------
    g_real : GaussianStats
        Gaussian parameters for the real/reference distribution.
    g_sim : GaussianStats
        Gaussian parameters for the simulated distribution.
    eps : float, optional
        Small diagonal regularizer for numerical stability.
    
    Returns
    -------
    float
        Whitened Fréchet distance. Interpretation:
        - 0 = identical distributions
        - Values are in "standard deviation units" from the real distribution
        - Scale-invariant: comparable across different embedders
    
    Notes
    -----
    Unlike standard FD, whitened FD is not affected by the absolute scale
    of embedding dimensions. This is useful when comparing results from
    different embedders (e.g., TF-IDF vs BERT).
    """
    mu_real, sigma_real = g_real.mean, g_real.cov
    mu_sim, sigma_sim = g_sim.mean, g_sim.cov
    
    if mu_real.shape != mu_sim.shape:
        raise ValueError(
            f"Mean vectors must have same shape; got {mu_real.shape} and {mu_sim.shape}"
        )
    if sigma_real.shape != sigma_sim.shape:
        raise ValueError(
            f"Covariance matrices must have same shape; got {sigma_real.shape} and {sigma_sim.shape}"
        )
    
    n = sigma_real.shape[0]
    
    # Compute Σ_real^{-1/2}
    A = _matrix_inv_sqrt(sigma_real, eps)
    
    # Whitened mean difference: μ̃ = Σ_real^{-1/2} (μ_sim - μ_real)
    mu_diff = mu_sim - mu_real
    mu_whitened = A @ mu_diff
    mean_term = float(mu_whitened @ mu_whitened)
    
    # Whitened covariance: Σ̃ = Σ_real^{-1/2} Σ_sim Σ_real^{-1/2}
    sigma_whitened = A @ sigma_sim @ A.T
    
    # Trace term: tr(Σ̃ + I - 2 Σ̃^{1/2})
    sqrt_sigma_whitened = _matrix_sqrt(sigma_whitened, eps)
    trace_term = np.trace(sigma_whitened) + n - 2 * np.trace(sqrt_sigma_whitened)
    
    fd_whitened = mean_term + float(trace_term)
    return float(max(fd_whitened, 0.0))


def whitened_frechet_from_samples(
    real: ArrayLike,
    simulated: ArrayLike,
    *,
    eps: float = 1e-6,
    use_shrinkage: bool = True,
    clip_outliers: bool = True,
) -> float:
    """
    Compute whitened Fréchet distance from samples (scale-invariant).
    
    This is a scale-invariant version of FD that is comparable across
    different embedders. It transforms to a coordinate system where
    the real distribution has identity covariance.
    
    Parameters
    ----------
    real : array-like, shape (n_real, d)
        Embeddings for the real actions or sessions.
    simulated : array-like, shape (n_sim, d)
        Embeddings for the simulated actions or sessions.
    eps : float, optional
        Diagonal regularizer for numerical stability.
    use_shrinkage : bool
        If True, use Ledoit-Wolf shrinkage for covariance estimation.
    clip_outliers : bool
        If True, clip outliers before estimation.
    
    Returns
    -------
    float
        Whitened Fréchet distance (scale-invariant).
    
    See Also
    --------
    frechet_from_samples : Standard (non-whitened) Fréchet distance.
    whitened_frechet_gaussians : Whitened FD from pre-computed Gaussian stats.
    
    Examples
    --------
    >>> import numpy as np
    >>> from frechet_ir import whitened_frechet_from_samples
    >>> real = np.random.randn(100, 64)
    >>> sim = np.random.randn(100, 64) + 0.5
    >>> fd = whitened_frechet_from_samples(real, sim)
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
    return whitened_frechet_gaussians(real_stats, sim_stats, eps=eps)

