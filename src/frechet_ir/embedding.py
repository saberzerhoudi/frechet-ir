from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Sequence, Union

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


# ---------------------------------------------------------------------------
# Recommended BERT Models for Search Sessions
# ---------------------------------------------------------------------------

class BertModelPreset(str, Enum):
    """Pre-configured BERT models for search session embedding."""
    
    # Fast, lightweight models (recommended for quick experiments)
    MINILM_L6 = "all-MiniLM-L6-v2"          # 384-dim, fast, good quality
    MINILM_L12 = "all-MiniLM-L12-v2"        # 384-dim, better quality
    
    # High quality general models
    MPNET_BASE = "all-mpnet-base-v2"        # 768-dim, best quality
    DISTILBERT = "all-distilroberta-v1"     # 768-dim, good balance
    
    # MS MARCO trained models (optimized for search/retrieval)
    MSMARCO_MINILM = "msmarco-MiniLM-L6-cos-v5"      # 384-dim, search-optimized
    MSMARCO_DISTILBERT = "msmarco-distilbert-cos-v5"  # 768-dim, search-optimized
    
    # Multilingual models
    MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384-dim, 50+ languages
    
    # E5 models (newer, high quality)
    E5_SMALL = "intfloat/e5-small-v2"       # 384-dim, excellent quality
    E5_BASE = "intfloat/e5-base-v2"         # 768-dim, state-of-the-art


# ---------------------------------------------------------------------------
# TF-IDF Action Embedding (LSA-style)
# ---------------------------------------------------------------------------

@dataclass
class TfidfActionEmbedder:
    """
    TF-IDF-based action embedder with dimensionality reduction.

    This embedder captures lexical content from queries using TF-IDF,
    applies SVD for dimensionality reduction (Latent Semantic Analysis),
    and includes behavioral signals (click indicator, rank position).

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF features before dimensionality reduction.
    n_components : int
        Number of SVD components (final text embedding dimension).
    max_rank : int
        Maximum rank value for capping.
    include_title : bool
        If True and action has a title, include it in the text.
    include_snippet : bool
        If True and action has a snippet, include it in the text.
    standardize : bool
        If True, standardize features to zero mean and unit variance.
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
        return text if text else "empty"

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

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
            norm='l2',
            sublinear_tf=True,
            use_idf=True,
        )
        X_tfidf = self._vectorizer.fit_transform(texts)

        actual_components = min(
            self.n_components,
            X_tfidf.shape[1] - 1,
            X_tfidf.shape[0] - 1,
        )
        actual_components = max(actual_components, 1)
        
        self._svd = TruncatedSVD(n_components=actual_components, random_state=42)
        X_reduced = self._svd.fit_transform(X_tfidf)

        if self.standardize:
            self._scaler = StandardScaler()
            self._scaler.fit(X_reduced)

        self._feature_dim = actual_components + 2
        return self

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfActionEmbedder must be fitted before use.")

        if not actions:
            return np.empty((0, self._feature_dim or 0), dtype=np.float64)

        texts = [self._action_text(a) for a in actions]
        X_tfidf = self._vectorizer.transform(texts)
        X_reduced = self._svd.transform(X_tfidf)
        
        if self.standardize and self._scaler is not None:
            X_reduced = self._scaler.transform(X_reduced)

        n = len(actions)
        is_click = np.array([1.0 if a.is_click else 0.0 for a in actions], dtype=np.float64)
        rank_feature = np.array([
            min(float(a.rank), float(self.max_rank)) / float(self.max_rank)
            if a.rank is not None else 0.0
            for a in actions
        ], dtype=np.float64)

        features = np.hstack([X_reduced, is_click.reshape(n, 1), rank_feature.reshape(n, 1)])
        return features.astype(np.float64)


# ---------------------------------------------------------------------------
# BERT-based Action Embedding
# ---------------------------------------------------------------------------

@dataclass
class BertActionEmbedder:
    """
    BERT-based action embedder using sentence transformers.

    For each action, builds a rich text representation and encodes with a
    pre-trained SentenceTransformer model.

    Parameters
    ----------
    model_name : str or BertModelPreset
        Name of the SentenceTransformer model. Use BertModelPreset for
        pre-configured options, or any HuggingFace model name.
        
        Recommended models:
        - "all-MiniLM-L6-v2": Fast, 384-dim (default)
        - "all-mpnet-base-v2": Best quality, 768-dim
        - "msmarco-MiniLM-L6-cos-v5": Search-optimized, 384-dim
        - "intfloat/e5-small-v2": State-of-the-art, 384-dim
        
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
    """

    model_name: Union[str, BertModelPreset] = BertModelPreset.MINILM_L6
    max_seq_length: int = 128
    batch_size: int = 64
    show_progress_bar: bool = False
    max_rank: int = 10
    standardize: bool = True

    _model: object | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _feature_dim: int | None = field(default=None, repr=False)

    def _get_model_name(self) -> str:
        if isinstance(self.model_name, BertModelPreset):
            return self.model_name.value
        return self.model_name

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

        model = SentenceTransformer(self._get_model_name())
        model.max_seq_length = self.max_seq_length
        self._model = model
        self._feature_dim = model.get_sentence_embedding_dimension() + 2

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "BertActionEmbedder":
        self._ensure_model_loaded()
        
        if self.standardize:
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
                    texts, batch_size=self.batch_size,
                    convert_to_numpy=True, normalize_embeddings=True,
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

        texts = [self._action_text(a) for a in actions]
        emb = model.encode(
            texts, batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )

        if self.standardize and self._scaler is not None:
            emb = self._scaler.transform(emb)

        n = len(actions)
        is_click = np.array([1.0 if a.is_click else 0.0 for a in actions], dtype=np.float64)
        rank_feature = np.array([
            min(float(a.rank), float(self.max_rank)) / float(self.max_rank)
            if a.rank is not None else 0.0
            for a in actions
        ], dtype=np.float64)

        features = np.hstack([emb, is_click.reshape(n, 1), rank_feature.reshape(n, 1)])
        return features.astype(np.float64)


# ---------------------------------------------------------------------------
# Word2Vec/GloVe Action Embedding (Mean Pooling)
# ---------------------------------------------------------------------------

@dataclass
class Word2VecActionEmbedder:
    """
    Word embedding-based action embedder using mean pooling.
    
    Computes query embeddings as the average of query word embeddings,
    and document embeddings from title and snippet words. This follows
    the approach described in Section 4.3 of the paper.

    Parameters
    ----------
    model_name : str
        Name of the word embedding model:
        - "word2vec-google-news-300": Google News Word2Vec (300-dim)
        - "glove-wiki-gigaword-300": GloVe trained on Wikipedia (300-dim)
        - "glove-wiki-gigaword-100": GloVe 100-dim (faster)
        - "fasttext-wiki-news-subwords-300": FastText with subwords
    include_title : bool
        Include document title in embedding.
    include_snippet : bool
        Include document snippet in embedding.
    max_rank : int
        Maximum rank value for capping.
    standardize : bool
        If True, standardize embeddings.
        
    Notes
    -----
    Requires gensim: `pip install gensim`
    """

    model_name: str = "glove-wiki-gigaword-100"
    include_title: bool = True
    include_snippet: bool = False
    max_rank: int = 10
    standardize: bool = True

    _model: object | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _feature_dim: int | None = field(default=None, repr=False)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import gensim.downloader as api
        except ImportError as exc:
            raise RuntimeError(
                "gensim is not installed. Install with `pip install gensim`."
            ) from exc

        self._model = api.load(self.model_name)
        self._feature_dim = self._model.vector_size + 2  # type: ignore

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to mean-pooled word embedding."""
        words = text.lower().split()
        vectors = []
        for word in words:
            try:
                vectors.append(self._model[word])  # type: ignore
            except KeyError:
                continue
        
        if vectors:
            return np.mean(vectors, axis=0).astype(np.float64)
        else:
            return np.zeros(self._model.vector_size, dtype=np.float64)  # type: ignore

    def _action_text(self, a: Action) -> str:
        """Build text representation of an action."""
        parts = [a.query or ""]
        if self.include_title and a.title:
            parts.append(a.title)
        if self.include_snippet and a.snippet:
            parts.append(a.snippet)
        return " ".join(parts).strip()

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "Word2VecActionEmbedder":
        self._ensure_model_loaded()
        
        if self.standardize:
            all_actions: List[Action] = []
            for s in real_sessions:
                all_actions.extend(s.actions)
            for s in simulated_sessions:
                all_actions.extend(s.actions)
            
            if all_actions:
                embeddings = [self._text_to_embedding(self._action_text(a)) for a in all_actions]
                emb_matrix = np.vstack(embeddings)
                self._scaler = StandardScaler()
                self._scaler.fit(emb_matrix)
        
        return self

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        if not actions:
            return np.empty((0, self._feature_dim or 0), dtype=np.float64)

        self._ensure_model_loaded()

        embeddings = [self._text_to_embedding(self._action_text(a)) for a in actions]
        emb_matrix = np.vstack(embeddings)

        if self.standardize and self._scaler is not None:
            emb_matrix = self._scaler.transform(emb_matrix)

        n = len(actions)
        is_click = np.array([1.0 if a.is_click else 0.0 for a in actions], dtype=np.float64)
        rank_feature = np.array([
            min(float(a.rank), float(self.max_rank)) / float(self.max_rank)
            if a.rank is not None else 0.0
            for a in actions
        ], dtype=np.float64)

        features = np.hstack([emb_matrix, is_click.reshape(n, 1), rank_feature.reshape(n, 1)])
        return features.astype(np.float64)


# ---------------------------------------------------------------------------
# Session2Vec Embedding (Doc2Vec for Sessions)
# ---------------------------------------------------------------------------

@dataclass
class Session2VecEmbedder:
    """
    Session-level embedder using Doc2Vec (Session2Vec).
    
    Learns fixed-length vector representations of entire search sessions,
    adapting the Doc2Vec approach for session modeling. This treats each
    session as a "document" composed of query and click actions.

    Parameters
    ----------
    vector_size : int
        Dimensionality of the session vectors.
    window : int
        Context window size for Doc2Vec.
    min_count : int
        Minimum word count to be included in vocabulary.
    epochs : int
        Number of training epochs.
    dm : int
        Training algorithm: 1 for PV-DM, 0 for PV-DBOW.
    include_clicks : bool
        If True, include click indicators in session text.
    standardize : bool
        If True, standardize embeddings.
        
    Notes
    -----
    Requires gensim: `pip install gensim`
    
    This embedder produces SESSION-LEVEL embeddings, not action-level.
    Use `transform_sessions` instead of `transform_actions`.
    """

    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    epochs: int = 40
    dm: int = 1  # PV-DM (distributed memory)
    include_clicks: bool = True
    standardize: bool = True

    _model: object | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)

    def _session_to_words(self, session: Session) -> List[str]:
        """Convert session to list of words/tokens."""
        words = []
        for action in session.actions:
            if action.query:
                words.extend(action.query.lower().split())
            if self.include_clicks and action.is_click:
                words.append(f"__CLICK_RANK_{action.rank or 0}__")
            if action.title:
                words.extend(action.title.lower().split())
        return words

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "Session2VecEmbedder":
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except ImportError as exc:
            raise RuntimeError(
                "gensim is not installed. Install with `pip install gensim`."
            ) from exc

        # Use unique tags to avoid collision between real and simulated sessions
        documents = []
        for s in real_sessions:
            documents.append(TaggedDocument(
                words=self._session_to_words(s),
                tags=[f"real_{s.session_id}"]
            ))
        for s in simulated_sessions:
            documents.append(TaggedDocument(
                words=self._session_to_words(s),
                tags=[f"sim_{s.session_id}"]
            ))

        self._model = Doc2Vec(
            documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            workers=4,
        )

        if self.standardize:
            # Use infer_vector for all sessions to get consistent scaling
            embeddings = [
                self._model.infer_vector(self._session_to_words(s))
                for s in list(real_sessions) + list(simulated_sessions)
            ]
            emb_matrix = np.vstack(embeddings)
            self._scaler = StandardScaler()
            self._scaler.fit(emb_matrix)

        return self

    def transform_sessions(self, sessions: Sequence[Session]) -> Array2D:
        """Transform sessions to embeddings (session-level, not action-level).
        
        Always uses infer_vector() to compute embeddings from session content,
        ensuring fair comparison between real and simulated sessions.
        """
        if not sessions:
            return np.empty((0, self.vector_size), dtype=np.float64)

        if self._model is None:
            raise RuntimeError("Session2VecEmbedder must be fitted before use.")

        # Always infer from content for fair comparison (no lookup by ID)
        embeddings = [
            self._model.infer_vector(self._session_to_words(session))
            for session in sessions
        ]

        emb_matrix = np.vstack(embeddings).astype(np.float64)

        if self.standardize and self._scaler is not None:
            emb_matrix = self._scaler.transform(emb_matrix)

        return emb_matrix

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        """Not supported - use transform_sessions for Session2Vec."""
        raise NotImplementedError(
            "Session2VecEmbedder produces session-level embeddings. "
            "Use transform_sessions() instead, or use evaluate_sessions_fd "
            "with a session-level metric."
        )


# ---------------------------------------------------------------------------
# Mean Pooled Session Embedding (Aggregates Action Embeddings)
# ---------------------------------------------------------------------------

@dataclass
class MeanPooledSessionEmbedder:
    """
    Session-level embedder that aggregates action embeddings via mean pooling.
    
    Wraps any ActionEmbedder and produces session-level embeddings by
    averaging all action embeddings within a session.

    Parameters
    ----------
    action_embedder : ActionEmbedder
        The underlying action embedder to use.
    """

    action_embedder: ActionEmbedder = field(default_factory=TfidfActionEmbedder)

    def fit(
        self,
        real_sessions: Sequence[Session],
        simulated_sessions: Sequence[Session],
    ) -> "MeanPooledSessionEmbedder":
        self.action_embedder.fit(real_sessions, simulated_sessions)
        return self

    def transform_sessions(self, sessions: Sequence[Session]) -> Array2D:
        """Transform sessions to mean-pooled action embeddings."""
        session_embeddings = []
        for session in sessions:
            if session.actions:
                action_embs = self.action_embedder.transform_actions(session.actions)
                session_emb = np.mean(action_embs, axis=0)
            else:
                # Get dimension from a dummy call
                dummy = self.action_embedder.transform_actions([])
                session_emb = np.zeros(dummy.shape[1] if dummy.size else 64)
            session_embeddings.append(session_emb)
        
        return np.vstack(session_embeddings).astype(np.float64)

    def transform_actions(self, actions: Sequence[Action]) -> Array2D:
        """Direct action embedding (delegates to underlying embedder)."""
        return self.action_embedder.transform_actions(actions)
