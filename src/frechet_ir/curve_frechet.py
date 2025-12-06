from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Array2D = NDArray[np.float64]


def _as_curve(x: ArrayLike, name: str) -> Array2D:
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim == 1:
        # Interpret as 1D curve in R^1
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {arr.shape}")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")

    return arr


def discrete_frechet_distance(curve_a: ArrayLike, curve_b: ArrayLike) -> float:
    """
    Compute the discrete Fréchet distance between two polygonal curves.

    This is a dynamic-programming approximation of the continuous
    Fréchet distance as described in classic work by Eiter & Mannila.

    Parameters
    ----------
    curve_a : array-like, shape (n_a, d)
        Sequence of points defining the first curve.
    curve_b : array-like, shape (n_b, d)
        Sequence of points defining the second curve.

    Returns
    -------
    float
        Discrete Fréchet distance between the two curves (non-negative).
    """
    A = _as_curve(curve_a, "curve_a")
    B = _as_curve(curve_b, "curve_b")

    n_a, d_a = A.shape
    n_b, d_b = B.shape

    if d_a != d_b:
        raise ValueError(
            f"Dimension mismatch: curve_a in R^{d_a}, curve_b in R^{d_b}"
        )

    # DP table of distances; we fill it iteratively
    ca = np.full((n_a, n_b), -1.0, dtype=np.float64)

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(A[i] - B[j]))

    # Iterative DP (non-recursive to avoid recursion depth issues)
    for i in range(n_a):
        for j in range(n_b):
            d = dist(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(
                    min(
                        ca[i - 1, j],
                        ca[i - 1, j - 1],
                        ca[i, j - 1],
                    ),
                    d,
                )

    return float(ca[n_a - 1, n_b - 1])

