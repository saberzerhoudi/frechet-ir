"""
Session-level Fréchet Distance evaluation with optional normalization.

This module provides functions for evaluating simulated search sessions
against real sessions using Fréchet Distance, including multiple normalized
FD (nFD) variants for easier interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .embedding import ActionEmbedder, TfidfActionEmbedder
from .frechet import frechet_from_samples
from .model import Action, Session
from .session_fd import frechet_sessions, frechet_sessions_sequential


Array2D = NDArray[np.float64]


class NFDMethod(str, Enum):
    """Normalization methods for computing nFD."""
    
    RANDOM = "random"
    """nFD = 1 - FD/FD_random, where FD_random is FD against shuffled data."""
    
    SIGMOID = "sigmoid"
    """nFD = exp(-FD/τ), where τ is a temperature parameter."""
    
    SELF_DISTANCE = "self_distance"
    """nFD = 1 - FD/(FD + FD_self), where FD_self is FD of real data split in half."""


@dataclass
class FDResult:
    """
    Results from session FD evaluation.

    Attributes
    ----------
    fd_S_M : float
        Global Fréchet distance (all actions pooled).
    fd_S_seq_M : float
        Mean per-session Fréchet distance.
    n_sessions : int
        Number of sessions evaluated.
    nfd_S_M : float or None
        Normalized FD (global), if computed. Range [0, 1] where 1 is best.
    nfd_S_seq_M : float or None
        Normalized per-session FD, if computed. Range [0, 1] where 1 is best.
    nfd_method : str or None
        The normalization method used ("random", "sigmoid", or "self_distance").
    nfd_reference : float or None
        Reference FD used for normalization (FD_random, τ, or FD_self).
    """

    fd_S_M: float
    fd_S_seq_M: float
    n_sessions: int
    nfd_S_M: Optional[float] = None
    nfd_S_seq_M: Optional[float] = None
    nfd_method: Optional[str] = None
    nfd_reference: Optional[float] = None


def truncate_session_queries(session: Session, k: int) -> Session:
    """
    Truncate a session to include only the first k queries (and their clicks).

    Parameters
    ----------
    session : Session
        The session to truncate.
    k : int
        Maximum number of queries to keep.

    Returns
    -------
    Session
        Truncated session with at most k query actions.
    """
    if k <= 0:
        return Session(
            session_id=session.session_id,
            actions=[],
            topic_id=session.topic_id,
        )

    truncated_actions: List[Action] = []
    query_count = 0

    for action in session.actions:
        if not action.is_click:
            query_count += 1
            if query_count > k:
                break
        truncated_actions.append(action)

    return Session(
        session_id=session.session_id,
        actions=truncated_actions,
        topic_id=session.topic_id,
    )


def _compute_random_baseline_fd(
    real_embeddings: Array2D,
    sim_embeddings: Array2D,
    n_shuffles: int = 10,
    seed: int = 42,
) -> float:
    """
    Compute FD against a random baseline by shuffling embeddings.
    
    This creates a reference point for what "random" simulation would look like.
    """
    rng = np.random.RandomState(seed)
    fd_values = []
    
    combined = np.vstack([real_embeddings, sim_embeddings])
    n_real = len(real_embeddings)
    n_sim = len(sim_embeddings)
    
    for _ in range(n_shuffles):
        indices = rng.permutation(len(combined))
        shuffled_real = combined[indices[:n_real]]
        shuffled_sim = combined[indices[n_real:n_real + n_sim]]
        
        fd = frechet_from_samples(shuffled_real, shuffled_sim)
        fd_values.append(fd)
    
    return float(np.median(fd_values))


def _compute_self_distance_fd(
    real_embeddings: Array2D,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Compute FD_self by splitting real data in half and computing FD.
    
    This represents the "natural variation" within the real data itself.
    """
    if len(real_embeddings) < 4:
        return 1.0  # Fallback for very small datasets
    
    rng = np.random.RandomState(seed)
    fd_values = []
    
    for _ in range(n_splits):
        indices = rng.permutation(len(real_embeddings))
        mid = len(real_embeddings) // 2
        half1 = real_embeddings[indices[:mid]]
        half2 = real_embeddings[indices[mid:2*mid]]
        
        if len(half1) > 1 and len(half2) > 1:
            fd = frechet_from_samples(half1, half2)
            fd_values.append(fd)
    
    return float(np.median(fd_values)) if fd_values else 1.0


def _normalize_fd_random(
    fd: float,
    fd_random: float,
) -> float:
    """
    Normalize FD using random baseline: nFD = 1 - FD/FD_random
    
    Returns value in [0, 1] where 1 = perfect (FD=0), 0 = no better than random.
    """
    if fd_random <= 0:
        return 1.0 if fd == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - fd / fd_random))


def _normalize_fd_sigmoid(
    fd: float,
    temperature: float,
) -> float:
    """
    Normalize FD using sigmoid transform: nFD = exp(-FD/τ)
    
    Returns value in (0, 1] where 1 = perfect (FD=0), approaches 0 as FD → ∞.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return float(np.exp(-fd / temperature))


def _normalize_fd_self_distance(
    fd: float,
    fd_self: float,
) -> float:
    """
    Normalize FD using self-distance: nFD = 1 - FD/(FD + FD_self)
    
    Returns value in [0, 1] where 1 = perfect (FD=0), 0.5 = FD equals natural variation.
    """
    if fd_self <= 0:
        return 1.0 if fd == 0 else 0.0
    return 1.0 - fd / (fd + fd_self)


def evaluate_sessions_fd(
    real_sessions: Sequence[Session],
    simulated_sessions: Sequence[Session],
    embedder: Optional[ActionEmbedder] = None,
    *,
    truncate_queries: Optional[int] = None,
    limit_sessions: Optional[int] = None,
    compute_nfd: bool = False,
    nfd_method: Union[NFDMethod, Literal["random", "sigmoid", "self_distance"]] = "random",
    nfd_n_shuffles: int = 10,
    nfd_temperature: float = 1.0,
) -> FDResult:
    """
    Evaluate simulated sessions against real sessions using Fréchet Distance.

    Parameters
    ----------
    real_sessions : sequence of Session
        Ground-truth user sessions.
    simulated_sessions : sequence of Session
        Sessions produced by the simulation model.
    embedder : ActionEmbedder, optional
        Embedder to convert actions to vectors. If None, uses TfidfActionEmbedder.
    truncate_queries : int, optional
        If provided, truncate each session to first k queries (for FD@k protocol).
    limit_sessions : int, optional
        If provided, evaluate only the first N sessions.
    compute_nfd : bool
        If True, compute normalized FD (nFD).
    nfd_method : {"random", "sigmoid", "self_distance"}
        Normalization method to use:
        
        - "random" (default): nFD = 1 - FD/FD_random
          Where FD_random is FD against randomly shuffled sessions.
          Interpretation: 1 = perfect, 0 = no better than random.
          
        - "sigmoid": nFD = exp(-FD/τ)
          Where τ is the temperature parameter.
          Interpretation: 1 = perfect, approaches 0 as FD increases.
          
        - "self_distance": nFD = 1 - FD/(FD + FD_self)
          Where FD_self is FD of real data split in half.
          Interpretation: 1 = perfect, 0.5 = FD equals natural variation.
          
    nfd_n_shuffles : int
        Number of shuffles for "random" method (default: 10).
    nfd_temperature : float
        Temperature τ for "sigmoid" method (default: 1.0).

    Returns
    -------
    FDResult
        Evaluation results including FD, per-session FD, and optionally nFD.
    """
    if embedder is None:
        embedder = TfidfActionEmbedder()

    # Normalize method enum
    if isinstance(nfd_method, str):
        nfd_method = NFDMethod(nfd_method)

    # Apply session limits
    real_list = list(real_sessions)
    sim_list = list(simulated_sessions)

    if limit_sessions is not None:
        real_list = real_list[:limit_sessions]
        sim_list = sim_list[:limit_sessions]

    # Truncate queries if requested (FD@k protocol)
    if truncate_queries is not None:
        real_list = [truncate_session_queries(s, truncate_queries) for s in real_list]
        sim_list = [truncate_session_queries(s, truncate_queries) for s in sim_list]

    # Remove empty sessions
    real_list = [s for s in real_list if s.actions]
    sim_list = [s for s in sim_list if s.actions]

    n_sessions = min(len(real_list), len(sim_list))
    if n_sessions == 0:
        return FDResult(fd_S_M=float("inf"), fd_S_seq_M=float("inf"), n_sessions=0)

    # Fit embedder on all sessions
    embedder.fit(real_list, sim_list)

    # Get embeddings
    real_actions = [a for s in real_list for a in s.actions]
    sim_actions = [a for s in sim_list for a in s.actions]

    real_emb = embedder.transform_actions(real_actions)
    sim_emb = embedder.transform_actions(sim_actions)

    # Global FD (all actions pooled)
    fd_global = frechet_sessions(real_emb, sim_emb)

    # Per-session FD
    from .session_fd import SessionEmbeddings
    session_embeddings = [
        SessionEmbeddings(
            ideal=embedder.transform_actions(real_list[i].actions),
            simulated=embedder.transform_actions(sim_list[i].actions),
        )
        for i in range(n_sessions)
        if real_list[i].actions and sim_list[i].actions
    ]
    fd_per_session = frechet_sessions_sequential(session_embeddings) if session_embeddings else 0.0

    # Compute normalized FD if requested
    nfd_global = None
    nfd_per_session = None
    nfd_reference = None
    nfd_method_str = None

    if compute_nfd and len(real_emb) > 0 and len(sim_emb) > 0:
        nfd_method_str = nfd_method.value
        
        if nfd_method == NFDMethod.RANDOM:
            # Method 1: Random baseline
            nfd_reference = _compute_random_baseline_fd(
                real_emb, sim_emb, n_shuffles=nfd_n_shuffles
            )
            nfd_global = _normalize_fd_random(fd_global, nfd_reference)
            nfd_per_session = _normalize_fd_random(fd_per_session, nfd_reference)
            
        elif nfd_method == NFDMethod.SIGMOID:
            # Method 2: Sigmoid transform
            nfd_reference = nfd_temperature
            nfd_global = _normalize_fd_sigmoid(fd_global, nfd_temperature)
            nfd_per_session = _normalize_fd_sigmoid(fd_per_session, nfd_temperature)
            
        elif nfd_method == NFDMethod.SELF_DISTANCE:
            # Method 3: Self-distance normalization
            nfd_reference = _compute_self_distance_fd(real_emb, n_splits=nfd_n_shuffles)
            nfd_global = _normalize_fd_self_distance(fd_global, nfd_reference)
            nfd_per_session = _normalize_fd_self_distance(fd_per_session, nfd_reference)

    return FDResult(
        fd_S_M=fd_global,
        fd_S_seq_M=fd_per_session,
        n_sessions=n_sessions,
        nfd_S_M=nfd_global,
        nfd_S_seq_M=nfd_per_session,
        nfd_method=nfd_method_str,
        nfd_reference=nfd_reference,
    )
