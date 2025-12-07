# frechet-ir

**Fréchet Distance for Evaluating Simulated Search Sessions**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for computing the Fréchet Distance (FD) between real (ground-truth) and simulated search sessions. This metric provides a principled way to evaluate how well user simulation models replicate actual user behavior in information retrieval systems.

## Installation

```bash
# Basic installation
pip install frechet-ir

# With BERT embeddings support
pip install frechet-ir[bert]
```

Or using Poetry:

```bash
poetry add frechet-ir
poetry add frechet-ir -E bert  # with BERT support
```

## Quick Start

```python
import numpy as np
from frechet_ir import frechet_from_samples

# Compare two sets of embeddings
real_embeddings = np.random.randn(100, 64)
simulated_embeddings = np.random.randn(100, 64)

fd = frechet_from_samples(real_embeddings, simulated_embeddings)
print(f"Fréchet Distance: {fd:.4f}")
```

### Working with Search Sessions

```python
from frechet_ir import (
    Action, Session,
    evaluate_sessions_fd,
    TfidfActionEmbedder,
    load_sessions_from_json,
)

# Load your session data
real_sessions = load_sessions_from_json("real_sessions.json")
simulated_sessions = load_sessions_from_json("simulated_sessions.json")

# Evaluate similarity using FD
result = evaluate_sessions_fd(
    real_sessions,
    simulated_sessions,
    embedder=TfidfActionEmbedder(),  # Uses sensible defaults
)

print(f"Global FD: {result.fd_S_M:.4f}")
print(f"Mean per-session FD: {result.fd_S_seq_M:.4f}")
print(f"Sessions evaluated: {result.n_sessions}")
```

### FD@K Evaluation Protocol

For standardized evaluation:

```python
from frechet_ir import evaluate_sessions_fd, TfidfActionEmbedder

# FD@1: Consider only the first query per session
result_fd1 = evaluate_sessions_fd(
    real_sessions,
    simulated_sessions,
    truncate_queries=1,
    limit_sessions=200,
)

# FD@10: Consider first 10 queries per session
result_fd10 = evaluate_sessions_fd(
    real_sessions,
    simulated_sessions,
    truncate_queries=10,
    limit_sessions=200,
)

print(f"FD@1:  {result_fd1.fd_S_M:.4f}")
print(f"FD@10: {result_fd10.fd_S_M:.4f}")
```

---

## Data Format

### Session Structure

Sessions are sequences of user actions (queries and clicks). Create them programmatically or load from files:

```python
from frechet_ir import Action, Session

# Create actions manually
actions = [
    Action(
        session_id="s1",
        query="python tutorial",
        is_click=False,  # Query submission
    ),
    Action(
        session_id="s1",
        query="python tutorial",
        is_click=True,   # Click on a result
        doc_id="doc_001",
        rank=1,
        title="Python Tutorial",
        snippet="Learn Python programming...",
    ),
]

session = Session(session_id="s1", actions=actions)
```

### Loading from Files

**JSON format:**

```json
[
  {
    "session_id": "s1",
    "topic_id": "topic_1",
    "actions": [
      {"query": "python tutorial", "is_click": false},
      {"query": "python tutorial", "is_click": true, "doc_id": "doc_001", "rank": 1}
    ]
  }
]
```

```python
from frechet_ir import load_sessions_from_json
sessions = load_sessions_from_json("sessions.json")
```

**CSV format:**

```csv
session_id,query,is_click,doc_id,rank,title
s1,python tutorial,false,,,
s1,python tutorial,true,doc_001,1,Python Tutorial
```

```python
from frechet_ir import load_sessions_from_csv
sessions = load_sessions_from_csv("sessions.csv")
```

---

## Dataset-Specific Recommendations

Different datasets provide varying levels of detail. Use the appropriate embedder and configuration based on your data:

### Query Logs Only (minimal data)

If your dataset only contains queries without document information:

```python
from frechet_ir import TfidfActionEmbedder, evaluate_sessions_fd

# Use TF-IDF embedder with defaults (64-dim SVD projection)
embedder = TfidfActionEmbedder(
    max_features=5000,
    n_components=64,  # SVD dimensions
)

# Create actions with just query and is_click
actions = [
    Action(session_id="s1", query="python tutorial", is_click=False),
    Action(session_id="s1", query="python basics", is_click=False),
]
```

**Available fields:** `session_id`, `query`, `is_click`, `timestamp`

### Query + Click Logs (typical web logs)

If you have queries with click information and ranks:

```python
embedder = TfidfActionEmbedder(
    max_features=5000,
    n_components=64,
    max_rank=10,  # Cap rank at 10
)

actions = [
    Action(session_id="s1", query="python", is_click=False),
    Action(session_id="s1", query="python", is_click=True, rank=1, doc_id="d1"),
    Action(session_id="s1", query="python", is_click=True, rank=3, doc_id="d2"),
]
```

**Available fields:** + `rank`, `doc_id`, `dwell_time`, `url`

### Rich Session Data (SERP with snippets/titles)

If your dataset includes document titles and snippets shown in search results:

```python
from frechet_ir import BertActionEmbedder

# Use BERT for semantic understanding of content
embedder = BertActionEmbedder(
    model_name="all-MiniLM-L6-v2",  # Fast, 384-dim
)

actions = [
    Action(
        session_id="s1",
        query="machine learning",
        is_click=True,
        rank=2,
        title="Introduction to Machine Learning",
        snippet="A comprehensive guide to ML algorithms...",
    ),
]
```

**Available fields:** + `title`, `snippet`, `relevance_label`

### Comparison Table

| Dataset Type | Recommended Embedder | Key Parameters | Example Datasets |
|--------------|---------------------|----------------|------------------|
| Query logs only | `TfidfActionEmbedder` | `n_components=64` | AOL, Yandex Toloka |
| Query + clicks | `TfidfActionEmbedder` | `n_components=64`, `max_rank=10` | TREC Session, Sogou-QCL |
| Rich SERP data | `BertActionEmbedder` | `model_name="all-MiniLM-L6-v2"` | TREC Session 2014 |
| Custom embeddings | Direct `frechet_from_samples` | - | Any pre-computed |

---

## Interpreting Results

### What FD Values Mean

The Fréchet Distance measures how different two distributions are:

| FD Value | Interpretation |
|----------|----------------|
| **FD = 0** | Identical distributions (perfect simulation) |
| **FD < 1** | Very similar distributions (excellent simulation) |
| **FD 1-5** | Moderately similar (good simulation) |
| **FD 5-10** | Notable differences (acceptable simulation) |
| **FD > 10** | Significantly different distributions |

> **Note:** Compare FD values only when using the same embedder configuration.

### Normalized FD (nFD)

For easier interpretation, use the normalized FD metric with one of three methods:

```python
from frechet_ir import evaluate_sessions_fd

# Method 1: Random Baseline (default) - recommended
result = evaluate_sessions_fd(
    real_sessions, simulated_sessions,
    compute_nfd=True,
    nfd_method="random",  # default
)

# Method 2: Sigmoid Transform
result = evaluate_sessions_fd(
    real_sessions, simulated_sessions,
    compute_nfd=True,
    nfd_method="sigmoid",
    nfd_temperature=1.0,  # τ parameter
)

# Method 3: Self-Distance
result = evaluate_sessions_fd(
    real_sessions, simulated_sessions,
    compute_nfd=True,
    nfd_method="self_distance",
)

print(f"nFD: {result.nfd_S_M:.4f}")  # Range [0, 1]
```

| Method | Formula | Interpretation |
|--------|---------|----------------|
| **Random Baseline** | `nFD = 1 - FD/FD_random` | 1 = perfect, 0 = no better than random |
| **Sigmoid** | `nFD = exp(-FD/τ)` | 1 = perfect, smooth decay to 0 |
| **Self-Distance** | `nFD = 1 - FD/(FD + FD_self)` | 1 = perfect, 0.5 = matches natural variation |

### Relative Comparisons

FD is most useful for **comparing simulation models**:

```python
# Compare multiple simulators
results = {}
for sim_name, sim_sessions in simulators.items():
    result = evaluate_sessions_fd(real_sessions, sim_sessions, embedder=embedder)
    results[sim_name] = result.fd_S_M

# Lower FD = better simulation
best_simulator = min(results, key=results.get)
print(f"Best simulator: {best_simulator} (FD={results[best_simulator]:.4f})")
```

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `frechet_from_samples(real, sim)` | Compute FD from embedding arrays |
| `frechet_gaussians(g1, g2)` | Compute FD between Gaussian stats |
| `estimate_gaussian(x)` | Estimate mean/covariance from samples |

> **Robustness options** for `frechet_from_samples` and `estimate_gaussian`:
> - `use_shrinkage=True`: Ledoit-Wolf shrinkage for stable covariance
> - `clip_outliers=True`: Clip values beyond 5σ before estimation

### Session Evaluation

| Function | Description |
|----------|-------------|
| `evaluate_sessions_fd(real, sim)` | Full session evaluation pipeline |
| `frechet_sessions(real_emb, sim_emb)` | Global FD from embeddings |
| `frechet_sessions_sequential(sessions)` | Mean per-session FD |
| `truncate_session_queries(session, k)` | Truncate to first k queries |

### Data Loading

| Function | Description |
|----------|-------------|
| `load_sessions_from_json(path)` | Load sessions from JSON file |
| `load_sessions_from_csv(path)` | Load sessions from CSV file |
| `load_sessions_from_dict(data)` | Load sessions from Python dict |

### Embedders

**Action-level embedders** (embed individual actions):

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `TfidfActionEmbedder` | `n_components=64` | TF-IDF + SVD (LSA-style) |
| `BertActionEmbedder` | `model_name` | Sentence-transformer embeddings |
| `Word2VecActionEmbedder` | `model_name` | Mean-pooled word embeddings |

**Session-level embedders** (embed entire sessions):

| Class | Description |
|-------|-------------|
| `Session2VecEmbedder` | Doc2Vec adapted for sessions |
| `MeanPooledSessionEmbedder` | Aggregates action embeddings via mean pooling |

**When to use each level:**

| Level | Best for | Granularity | FD Magnitude |
|-------|----------|-------------|--------------|
| **Action-level** | Precise action-by-action behavior comparison | High (individual clicks/queries) | Higher |
| **Session-level** | Overall session trajectory comparison | Low (aggregated) | Lower |

> **Tip:** Session-level embeddings (mean pooling) smooth out per-action noise, resulting in lower FD values. This doesn't mean "better simulation" - it captures less granular information. Use action-level for detailed behavioral analysis.

### Embedder Selection Guide

| Embedder | Use Case | Speed | External Deps | Semantic Depth |
|----------|----------|-------|---------------|----------------|
| **TfidfActionEmbedder** | Quick lexical comparison, baseline | Fast | None | Basic |
| **BertActionEmbedder** | Deep semantic understanding of queries | Slow | sentence-transformers | Deep |
| **Word2VecActionEmbedder** | Classic word semantics, lightweight | Fast | gensim | Medium |
| **Session2VecEmbedder** | Session-level trajectory patterns | Medium | gensim | Medium |
| **MeanPooledSessionEmbedder** | Smoothed session comparison | Fast | Varies | Varies |

**Detailed recommendations:**

- **TF-IDF + SVD**: Start here. Fast, no GPU needed, captures lexical overlap. Best for initial experiments.

- **BERT (MiniLM/MSMARCO)**: Use when query semantics matter (e.g., "laptop" vs "notebook"). MSMARCO models are optimized for search tasks.

- **Word2Vec/GloVe**: Good middle ground between speed and semantics. Captures word-level meaning without transformer overhead.

- **Session2Vec**: Best for session-level trajectory analysis. Learns session patterns directly from your data.

- **MeanPooled**: Use when you want session-level FD but with action-level embedders. Aggregates action embeddings via averaging.

**nFD Normalization Methods:**

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Random Baseline** | `1 - FD/FD_random` | Default. Compare models against random chance. |
| **Sigmoid** | `exp(-FD/τ)` | Consistent [0,1] scale across different datasets. |
| **Self-Distance** | `1 - FD/(FD+FD_self)` | Compare to natural variation in real data. |

**BERT Model Presets** (`BertModelPreset`):

```python
from frechet_ir import BertActionEmbedder, BertModelPreset

# Use preset enum
embedder = BertActionEmbedder(model_name=BertModelPreset.MSMARCO_MINILM)

# Or use string directly
embedder = BertActionEmbedder(model_name="all-mpnet-base-v2")
```

| Preset | Model Name | Dim | Notes |
|--------|------------|-----|-------|
| `MINILM_L6` | all-MiniLM-L6-v2 | 384 | Fast, good quality (default) |
| `MINILM_L12` | all-MiniLM-L12-v2 | 384 | Better quality |
| `MPNET_BASE` | all-mpnet-base-v2 | 768 | Best quality |
| `MSMARCO_MINILM` | msmarco-MiniLM-L6-cos-v5 | 384 | Search-optimized |
| `MSMARCO_DISTILBERT` | msmarco-distilbert-cos-v5 | 768 | Search-optimized |
| `E5_SMALL` | intfloat/e5-small-v2 | 384 | State-of-the-art |
| `E5_BASE` | intfloat/e5-base-v2 | 768 | State-of-the-art |
| `MULTILINGUAL` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | 50+ languages |

---

## Citation

If you use this library in your research, please cite:

```bibtex
@inproceedings{Zerhoudi:2024:IIR,
  title={Beyond Conventional Metrics: Assessing User Simulators in Information Retrieval},
  author={Zerhoudi, Saber and Granitzer, Michael},
  booktitle={Proceedings of the 14th Italian Information Retrieval Workshop (IIR 2024)},
  pages={3--12},
  year={2024}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
