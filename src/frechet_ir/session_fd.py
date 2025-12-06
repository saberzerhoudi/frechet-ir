from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frechet import frechet_from_samples


Array2D = NDArray[np.float64]


@dataclass(frozen=True)
class SessionEmbeddings:
    """
    Container for embeddings of a single search session.

    Attributes
    ----------
    ideal : (n_actions, d) array
        Embeddings of the "ideal" (ground truth) actions R_{s_i}.
    simulated : (m_actions, d) array
        Embeddings of the simulated actions M(s_i).
        Usually n_actions == m_actions, but the metric does not
        mathematically require it as long as the feature dimension d matches.
    """

    ideal: Array2D
    simulated: Array2D


def _as_2d(x: ArrayLike, name: str) -> Array2D:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def frechet_sessions(
    ideal_embeddings: ArrayLike,
    simulated_embeddings: ArrayLike,
) -> float:
    """
    Compute FD_S^M – Fréchet Distance between distributions over *all* actions.

    This corresponds to Equation (5) in the paper:

        FD_S^M = FD( V(R_S), V(M(S)) )

    where V(R_S) and V(M(S)) denote the sets of embeddings for all ideal
    and simulated actions across the session set S.

    Parameters
    ----------
    ideal_embeddings : array-like, shape (n_ideal, d)
        Embeddings of all ideal actions R_S, concatenated over sessions.
    simulated_embeddings : array-like, shape (n_sim, d)
        Embeddings of all simulated actions M(S), concatenated over sessions.

    Returns
    -------
    float
        Fréchet distance between the two embedding distributions.
    """
    ideal_arr = _as_2d(ideal_embeddings, "ideal_embeddings")
    simulated_arr = _as_2d(simulated_embeddings, "simulated_embeddings")

    if ideal_arr.shape[1] != simulated_arr.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {ideal_arr.shape[1]} (ideal) vs "
            f"{simulated_arr.shape[1]} (simulated)"
        )

    return frechet_from_samples(ideal_arr, simulated_arr)


def frechet_sessions_sequential(
    sessions: Sequence[SessionEmbeddings],
) -> float:
    """
    Compute FD_{S_seq}^M – average per-session Fréchet Distance.

    This corresponds to Equation (6) in the paper:

        FD_{S_seq}^M = (1 / |S|) * sum_{s_i in S} FD( V(R_{s_i}), V(M(s_i)) )

    Parameters
    ----------
    sessions : sequence of SessionEmbeddings
        Each item contains the ideal and simulated embeddings for one session.

    Returns
    -------
    float
        Mean Fréchet distance over all sessions in S.

    Raises
    ------
    ValueError
        If any session has mismatched feature dimensions.
    """
    if not sessions:
        raise ValueError("sessions must not be empty")

    distances: List[float] = []

    for idx, sess in enumerate(sessions):
        ideal = _as_2d(sess.ideal, f"sessions[{idx}].ideal")
        simulated = _as_2d(sess.simulated, f"sessions[{idx}].simulated")

        if ideal.shape[1] != simulated.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch in session {idx}: "
                f"{ideal.shape[1]} (ideal) vs {simulated.shape[1]} (simulated)"
            )

        d = frechet_from_samples(ideal, simulated)
        distances.append(d)

    return float(np.mean(distances))

