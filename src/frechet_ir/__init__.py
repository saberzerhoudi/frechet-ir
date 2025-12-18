"""
frechet-ir: Fréchet Distance for Evaluating Simulated Search Sessions

This package provides tools for computing the Fréchet Distance between
real (ground-truth) and simulated search sessions. 
"""

from .frechet import (
    GaussianStats,
    estimate_gaussian,
    frechet_gaussians,
    frechet_from_samples,
    whitened_frechet_gaussians,
    whitened_frechet_from_samples,
)
from .model import Action, Session
from .session_fd import (
    SessionEmbeddings,
    frechet_sessions,
    frechet_sessions_sequential,
)
from .embedding import (
    ActionEmbedder,
    TfidfActionEmbedder,
    BertActionEmbedder,
    BertModelPreset,
    Word2VecActionEmbedder,
    Session2VecEmbedder,
    MeanPooledSessionEmbedder,
)
from .loader import (
    load_sessions_from_json,
    load_sessions_from_csv,
    load_sessions_from_dict,
)
from .evaluation import (
    FDResult,
    NFDMethod,
    evaluate_sessions_fd,
    truncate_session_queries,
)
from .curve_frechet import discrete_frechet_distance

__version__ = "0.2.1"

__all__ = [
    # Core FD computation
    "GaussianStats",
    "estimate_gaussian",
    "frechet_gaussians",
    "frechet_from_samples",
    "whitened_frechet_gaussians",
    "whitened_frechet_from_samples",
    # Data models
    "Action",
    "Session",
    # Session-level FD
    "SessionEmbeddings",
    "frechet_sessions",
    "frechet_sessions_sequential",
    # Embedders
    "ActionEmbedder",
    "TfidfActionEmbedder",
    "BertActionEmbedder",
    "BertModelPreset",
    "Word2VecActionEmbedder",
    "Session2VecEmbedder",
    "MeanPooledSessionEmbedder",
    # Data loading
    "load_sessions_from_json",
    "load_sessions_from_csv",
    "load_sessions_from_dict",
    # Evaluation
    "FDResult",
    "NFDMethod",
    "evaluate_sessions_fd",
    "truncate_session_queries",
    # Curve-based FD
    "discrete_frechet_distance",
]
