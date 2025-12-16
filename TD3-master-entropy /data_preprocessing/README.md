# Data Preprocessing for RL Recommendation System

This folder contains scripts and processed data for preparing embeddings and user data for the TD3-based recommendation system.

## Overview

The preprocessing pipeline converts raw recommendation system data into formats suitable for reinforcement learning training. It processes user beliefs, item embeddings, and cluster information to create structured data for the RL environment.

## Raw Data Files (in `embeddings/` folder)

| File | Description | Format |
|------|-------------|---------|
| `item_embedding.pkl` | Item embeddings from recommendation system | Dict: {token_str: embedding_array} |
| `user_embedding.pkl` | User embeddings from recommendation system | Dict: {user_id: embedding_array} |
| `recommendations.pkl` | Available items per user | Dict: {user_id: [item_list]} |
| `cluster_matrix_manifest_K5_topk5.json` | Cluster information (5 clusters) | JSON with cluster_to_items mapping |
| `item_token_map.json` | Maps item IDs to tokens | Dict: {"N12345": token_number} |
| `user_beliefs_K5_topk5.tsv` | User beliefs toward clusters | TSV with belief vectors |

## Processing Scripts

### 1. `calculate_user_average_beliefs.py` ⭐ **LATEST**
**Purpose**: Calculate average user beliefs from raw TSV data with dual output formats
- **Input**: `user_beliefs_K5_topk5.tsv` (156,963 entries from 50,000 users)
- **Output**: 
  - `user_average_beliefs.pkl` (3.8 MB) - Fast binary format
  - `user_average_beliefs.json` (5.0 MB) - Human-readable format
- **Processing**: Parses belief vectors, groups by user ID, calculates averages
- **Performance**: Loads 50k users in 0.07 seconds (700k+ users/second)

### 2. `process_user_beliefs.py` (Legacy)
**Purpose**: Calculate average user beliefs across multiple entries
- **Input**: `user_beliefs_K5_topk5.tsv` (156,963 entries)
- **Output**: `user_avg_beliefs.pkl` and `user_avg_beliefs.json`
- **Processing**: Groups by user ID and averages belief vectors

### 3. `convert_beliefs_to_vectors.py`
**Purpose**: Convert belief data to numpy arrays for efficient math operations
- **Input**: `user_avg_beliefs.pkl`
- **Output**: `user_belief_vectors.pkl`
- **Processing**: Converts to float32 numpy arrays, validates normalization

### 4. `calculate_cluster_embeddings_fixed.py`
**Purpose**: Calculate cluster embeddings by averaging item embeddings within each cluster
- **Input**: `item_embedding.pkl`, `cluster_matrix_manifest_K5_topk5.json`, `item_token_map.json`
- **Output**: `cluster_embeddings.pkl`, `cluster_embeddings.npy`
- **Processing**: Maps item IDs → tokens → embeddings, then averages per cluster

### 5. `test_load_user_beliefs.py`
**Purpose**: Test loading speed and validate processed user belief data
- **Input**: `user_average_beliefs.pkl`
- **Output**: Performance metrics and data validation
- **Processing**: Benchmarks loading speed, validates data structure

## Processed Data Files (in `processed_data/` folder)

### User Data ⭐ **UPDATED**
| File | Description | Dimensions | Count | Size |
|------|-------------|------------|-------|------|
| `user_average_beliefs.pkl` | **MAIN** - User beliefs (fast binary) | (5,) per user | 50,000 users | 3.8 MB |
| `user_average_beliefs.json` | **MAIN** - User beliefs (human-readable) | (5,) per user | 50,000 users | 5.0 MB |
| `user_belief_vectors.pkl` | Legacy - User beliefs as numpy arrays | (5,) per user | 50,000 users | - |
| `user_avg_beliefs.json` | Legacy - Human-readable user beliefs | (5,) per user | 50,000 users | - |
| `sample_belief_vectors.json` | Sample of 5 users for inspection | (5,) per user | 5 users | - |

### Cluster Data
| File | Description | Dimensions | Count |
|------|-------------|------------|-------|
| `cluster_embeddings.pkl` | Cluster embeddings dictionary | (64,) per cluster | 5 clusters |
| `cluster_embeddings.npy` | Cluster embeddings as array | (5, 64) | 5 clusters |
| `cluster_embeddings_info.json` | Metadata and sample values | - | 5 clusters |

## Data Dimensions Summary

### User Belief Vectors
- **Shape**: `(5,)` per user
- **Type**: `float32`
- **Range**: [0.0005, 0.998] per cluster
- **Normalization**: Sum ≈ 1.0 per user
- **Interpretation**: Probability distribution over 5 item clusters

### Cluster Embeddings
- **Shape**: `(64,)` per cluster
- **Type**: `float32`
- **Coverage**: 100% of items in each cluster
- **Cluster Sizes**:
  - Cluster 0: 13,383 items (largest)
  - Cluster 1: 1,355 items
  - Cluster 2: 1,413 items
  - Cluster 3: 1,381 items
  - Cluster 4: 1,598 items

### Item Embeddings (Original)
- **Shape**: `(64,)` per item
- **Type**: `float32`
- **Count**: 19,131 items
- **Keys**: String tokens (e.g., "5334", "10600")

### User Embeddings (Original)
- **Shape**: To be determined
- **Type**: To be determined
- **Count**: To be determined

## Usage in RL Environment

### Loading Data ⭐ **UPDATED**
```python
import pickle
import numpy as np
import json

# Load user beliefs (FASTEST - recommended for RL training)
with open('processed_data/user_average_beliefs.pkl', 'rb') as f:
    user_beliefs = pickle.load(f)  # Dict: {user_id: numpy_array}

# Load user beliefs (human-readable - for debugging)
with open('processed_data/user_average_beliefs.json', 'r') as f:
    user_beliefs_json = json.load(f)  # Dict: {user_id: list}

# Load cluster embeddings
with open('processed_data/cluster_embeddings.pkl', 'rb') as f:
    cluster_embeddings = pickle.load(f)

# Or load as numpy array
cluster_array = np.load('processed_data/cluster_embeddings.npy')
```

### Mathematical Operations
```python
# User belief for user U100
user_belief = user_beliefs['U100']  # Shape: (5,)

# Cluster embeddings
cluster_0_emb = cluster_embeddings['0']  # Shape: (64,)

# Compute user preference for cluster 0
preference_score = user_belief[0] * np.linalg.norm(cluster_0_emb)

# Compute weighted cluster representation for user
user_cluster_rep = np.sum([
    user_belief[i] * cluster_embeddings[str(i)] 
    for i in range(5)
], axis=0)  # Shape: (64,)
```

## Data Statistics ⭐ **UPDATED**

### User Beliefs (Latest Processing)
- **Total Users**: 50,000
- **Total Records Processed**: 156,963 (multiple entries per user)
- **Belief Vector Dimensions**: 5 (corresponding to K=5 clusters)
- **Data Type**: numpy.ndarray (float64)
- **Loading Performance**: 700k+ users/second from pickle

**Statistical Summary per Dimension:**
- **Mean**: [2.16e-05, 2.13e-04, 1.72e-04, 1.94e-04, 1.83e-04]
- **Std**: [3.87e-05, 3.41e-04, 3.26e-04, 3.37e-04, 2.89e-04]
- **Range**: [0.0, 0.00135] to [0.0, 0.00797] across dimensions

**Sample User Beliefs:**
- User U100: [0.0, 0.0, 0.0, 0.0, 0.000626]
- User U1000: [0.0, 0.000246, 0.000472, 0.0, 0.0]
- User U10001: [0.0, 0.000369, 0.000354, 0.0, 0.000313]

### Cluster Similarities
Cosine similarities between cluster embeddings:
- Cluster 0-1: -0.44 (most dissimilar)
- Cluster 0-3: -0.54 (most dissimilar)
- Cluster 1-3: +0.17 (most similar)
- Others: Range [-0.40, +0.07]

## File Dependencies

```
Raw Data (embeddings/)
├── user_beliefs_K5_topk5.tsv        # 156,963 records, 50k users
├── item_embedding.pkl               # 19,131 items
├── cluster_matrix_manifest_K5_topk5.json  # 5 clusters
└── item_token_map.json              # Item ID → token mapping

Processing Scripts ⭐ **UPDATED**
├── calculate_user_average_beliefs.py    # ⭐ MAIN - Latest user processing
├── test_load_user_beliefs.py           # Performance testing
├── process_user_beliefs.py             # Legacy user processing
├── convert_beliefs_to_vectors.py       # Vector conversion
├── calculate_cluster_embeddings_fixed.py  # Cluster processing
└── debug_data_formats.py               # Data inspection

Processed Data (processed_data/) ⭐ **UPDATED**
├── user_average_beliefs.pkl         # ⭐ MAIN - Fast user beliefs (3.8MB)
├── user_average_beliefs.json        # ⭐ MAIN - Readable user beliefs (5.0MB)
├── cluster_embeddings.pkl           # Main cluster data
├── cluster_embeddings.npy           # Array format
├── user_belief_vectors.pkl          # Legacy user data
└── *_info.json files                # Metadata
```

## Next Steps

The processed data is ready for RL environment integration. Key components needed:
1. **State representation**: Combine user beliefs + current context
2. **Action space**: Item selection from recommendation candidates
3. **Reward function**: Based on user interactions and cluster preferences
4. **Environment dynamics**: How recommendations affect user beliefs over time