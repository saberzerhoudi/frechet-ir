from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .model import Action, Session


Array2D = NDArray[np.float64]


class ActionEmbedder(Protocol):
    """Protocol for objects that can embed actions into vectors."""

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "ActionEmbedder":
        ...

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        ...


@dataclass
class TfidfActionEmbedder:
    """
    TF-IDF-based action embedder with dimensionality reduction.

    This embedder captures lexical content from queries using TF-IDF,
    applies SVD for dimensionality reduction (Latent Semantic Analysis),
    and includes behavioral signals (click indicator, rank position).

    The dimensionality reduction is crucial for producing meaningful
    Fréchet Distance values - without it, the high-dimensional sparse
    TF-IDF space leads to extremely large FD values.

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF features before dimensionality reduction.
    n_components : int
        Number of SVD components (final text embedding dimension).
        Default 64 provides a good balance of expressiveness and stability.
    max_rank : int
        Maximum rank value for capping. Ranks above this are clipped.
    include_title : bool
        If True and action has a title, include it in the text.
    include_snippet : bool
        If True and action has a snippet, include it in the text.
    standardize : bool
        If True, standardize features to zero mean and unit variance.
        This helps ensure all features contribute meaningfully to FD.

    Notes
    -----
    The embedding pipeline:
    1. TF-IDF with sublinear_tf for text representation
    2. TruncatedSVD for dimensionality reduction (LSA-style)
    3. Optional standardization for balanced feature contributions
    4. Behavioral features (is_click, rank) appended
    """

    max_features: int = 5000
    n_components: int = 64
    max_rank: int = 10
    include_title: bool = False
    include_snippet: bool = False
    standardize: bool = True
    clip_outliers: bool = True
    outlier_std: float = 5.0
    
    _vectorizer: TfidfVectorizer | None = field(default=None, repr=False)
    _svd: TruncatedSVD | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _feature_dim: int | None = field(default=None, repr=False)

    def _action_text(self, a: Action) -> str:
        """Build text representation of an action."""
        parts = [a.query or ""]
        if self.include_title and a.title:
            parts.append(a.title)
        if self.include_snippet and a.snippet:
            parts.append(a.snippet)
        text = " ".join(parts).strip()
        return text if text else "empty"  # Avoid empty strings

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "TfidfActionEmbedder":
        all_actions: List[Action] = []
        for s in real_sessions:
            all_actions.extend(s.actions)
        for s in simulated_sessions:
            all_actions.extend(s.actions)

        if not all_actions:
            raise ValueError("No actions found in sessions for fitting embedder.")

        texts = [self._action_text(a) for a in all_actions]

        # Step 1: TF-IDF vectorization
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
            norm='l2',  # L2 normalize TF-IDF vectors for stability
            sublinear_tf=True,
            use_idf=True,
        )
        X_tfidf = self._vectorizer.fit_transform(texts)

        # Step 2: Dimensionality reduction via SVD
        # Use min of n_components and available features/samples
        actual_components = min(
            self.n_components,
            X_tfidf.shape[1] - 1,  # SVD requires n_components < n_features
            X_tfidf.shape[0] - 1,  # And < n_samples
        )
        actual_components = max(actual_components, 1)  # At least 1 component
        
        self._svd = TruncatedSVD(n_components=actual_components, random_state=42)
        X_reduced = self._svd.fit_transform(X_tfidf)

        # Step 3: Optional standardization
        if self.standardize:
            self._scaler = StandardScaler()
            self._scaler.fit(X_reduced)

        self._feature_dim = actual_components + 2  # + is_click, rank
        return self

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfActionEmbedder must be fitted before use.")

        if not actions:
            return np.empty((0, self._feature_dim or 0), dtype=np.float64)

        texts = [self._action_text(a) for a in actions]
        
        # TF-IDF -> SVD
        X_tfidf = self._vectorizer.transform(texts)
        X_reduced = self._svd.transform(X_tfidf)
        
        # Optional standardization
        if self.standardize and self._scaler is not None:
            X_reduced = self._scaler.transform(X_reduced)

        n = len(actions)
        
        # Behavioral features (normalized to reasonable range)
        is_click = np.array(
            [1.0 if a.is_click else 0.0 for a in actions],
            dtype=np.float64,
        )
        
        # Rank normalized to [0, 1] range for balanced contribution
        rank_feature = np.array(
            [
                min(float(a.rank), float(self.max_rank)) / float(self.max_rank)
                if a.rank is not None else 0.0
                for a in actions
            ],
            dtype=np.float64,
        )

        is_click_col = is_click.reshape(n, 1)
        rank_col = rank_feature.reshape(n, 1)

        features = np.hstack([X_reduced, is_click_col, rank_col])
        return features.astype(np.float64)


# ---------------------------------------------------------------------------
# BERT-based Action Embedding
# ---------------------------------------------------------------------------


@dataclass
class BertActionEmbedder:
    """
    BERT-based action embedder using sentence transformers.

    For each action, builds a rich text representation including query,
    document title, and snippet (when available), then encodes with a
    pre-trained SentenceTransformer model.

    The dense embeddings from sentence transformers naturally produce
    Fréchet Distance values in reasonable ranges.

    Parameters
    ----------
    model_name : str
        Name of the SentenceTransformer model to use.
        Recommended: "all-MiniLM-L6-v2" (fast, 384-dim) or
                     "all-mpnet-base-v2" (better quality, 768-dim).
    max_seq_length : int
        Maximum sequence length for the transformer.
    batch_size : int
        Batch size for encoding.
    show_progress_bar : bool
        Whether to show progress during encoding.
    max_rank : int
        Maximum rank value for capping.
    standardize : bool
        If True, standardize embeddings to zero mean and unit variance.

    Notes
    -----
    - Uses `normalize_embeddings=True` for stable FD values
    - Includes document context (title/snippet) when available
    - Behavioral features normalized to [0, 1] range
    """

    model_name: str = "all-MiniLM-L6-v2"  # Fast, good quality, 384-dim
    max_seq_length: int = 128
    batch_size: int = 64
    show_progress_bar: bool = False
    max_rank: int = 10
    standardize: bool = True

    _model: object | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _feature_dim: int | None = field(default=None, repr=False)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with `pip install frechet-ir[bert]`."
            ) from exc

        model = SentenceTransformer(self.model_name)
        model.max_seq_length = self.max_seq_length
        self._model = model

        base_dim = model.get_sentence_embedding_dimension()
        self._feature_dim = base_dim + 2  # + is_click, rank

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "BertActionEmbedder":
        self._ensure_model_loaded()
        
        if self.standardize:
            # Fit scaler on all embeddings
            all_actions: List[Action] = []
            for s in real_sessions:
                all_actions.extend(s.actions)
            for s in simulated_sessions:
                all_actions.extend(s.actions)
            
            if all_actions:
                texts = [self._action_text(a) for a in all_actions]
                from sentence_transformers import SentenceTransformer
                model: SentenceTransformer = self._model  # type: ignore
                emb = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=self.show_progress_bar,
                )
                self._scaler = StandardScaler()
                self._scaler.fit(emb)
        
        return self

    @staticmethod
    def _action_text(a: Action) -> str:
        """Build rich text representation of an action."""
        prefix = "[CLICK]" if a.is_click else "[QUERY]"
        parts = [prefix, a.query or ""]
        
        if a.title:
            parts.append(f"[TITLE] {a.title}")
        if a.snippet:
            parts.append(f"[SNIPPET] {a.snippet}")
        
        text = " ".join(parts).strip()
        return text if text else "empty"

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        if not actions:
            return np.empty((0, self._feature_dim or 0), dtype=np.float64)

        self._ensure_model_loaded()
        from sentence_transformers import SentenceTransformer

        model: SentenceTransformer = self._model  # type: ignore
        assert self._feature_dim is not None

        texts = [self._action_text(a) for a in actions]

        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for stability
            show_progress_bar=self.show_progress_bar,
        )

        # Optional standardization
        if self.standardize and self._scaler is not None:
            emb = self._scaler.transform(emb)

        n = len(actions)
        
        # Behavioral features (normalized to [0, 1])
        is_click = np.array(
            [1.0 if a.is_click else 0.0 for a in actions],
            dtype=np.float64,
        )
        
        rank_feature = np.array(
            [
                min(float(a.rank), float(self.max_rank)) / float(self.max_rank)
                if a.rank is not None else 0.0
                for a in actions
            ],
            dtype=np.float64,
        )
        
        is_click_col = is_click.reshape(n, 1)
        rank_col = rank_feature.reshape(n, 1)

        features = np.hstack([emb, is_click_col, rank_col])
        return features.astype(np.float64)
