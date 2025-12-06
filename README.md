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

| Class | Key Parameters | Description |
|-------|----------------|-------------|
| `TfidfActionEmbedder` | `n_components=64`, `standardize=True`, `clip_outliers=True` | TF-IDF + SVD dimensionality reduction |
| `BertActionEmbedder` | `model_name="all-MiniLM-L6-v2"`, `standardize=True` | Sentence-transformer embeddings |

---

## License

MIT License - see [LICENSE](LICENSE) for details.
