# Recommendation RL System - Complete Design Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [State Representation](#state-representation)
4. [Action Space](#action-space)
5. [Reward System](#reward-system)
6. [User Belief System](#user-belief-system)
7. [User Acceptance Model](#user-acceptance-model)
8. [Training Process](#training-process)
9. [Data Flow](#data-flow)
10. [Key Components](#key-components)

---

## 1. System Overview

### Purpose
A reinforcement learning system using Twin Delayed Deep Deterministic Policy Gradient (TD3) to generate personalized recommendations that guide users toward diverse content consumption while respecting their preferences.

### Goal
Reduce "filter bubbles" by encouraging users to explore content across multiple clusters while maintaining high acceptance rates.

### Key Metrics
- **User Beliefs**: 5D probability distribution representing user preferences across 5 content clusters
- **Natural Belief Target**: `[0.700, 0.071, 0.074, 0.072, 0.084]` - optimal distribution derived from data
- **Acceptance Rate**: ~60% (using sigmoid-based intelligent acceptance)
- **Convergence**: Users reach natural belief targets within 50 rounds per episode

---

## 2. Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Training System                          │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   TD3 Agent  │◄────►│ Replay Buffer│                    │
│  │  (Actor +    │      │  (1M trans.) │                    │
│  │   Critics)   │      └──────────────┘                    │
│  └──────┬───────┘                                           │
│         │                                                    │
│         │ Actions (64D virtual items)                       │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Recommendation Environment                     │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │User Beliefs│  │Item Embeddings│  │Cluster Data  │ │  │
│  │  │   (5D)     │  │    (64D)      │  │   (K=5)      │ │  │
│  │  └────────────┘  └──────────────┘  └──────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         │ Rewards (step-based)                              │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │  Statistics  │                                           │
│  │  & Logging   │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### TD3 Agent Architecture

**Actor Network** (Policy):
- Input: 69D state (5D beliefs + 64D user embedding)
- Architecture: 69 → 128 → 128 → 128 → 64
- Activation: ReLU (hidden), Tanh (output)
- Output: 64D virtual item embedding (normalized to [-1, 1])

**Critic Networks** (Q-functions):
- Twin critics (Q1, Q2) for value estimation
- Input: 69D state + 64D action = 133D
- Architecture: 133 → 256 → 256 → 1
- Activation: ReLU
- Output: Q-value (scalar)

---

## 3. State Representation

### State Composition (69D)


```python
state = [
    # User Beliefs (5D) - probability distribution over clusters
    belief_cluster_0,  # e.g., 0.700 (70% preference for cluster 0)
    belief_cluster_1,  # e.g., 0.071 (7.1% preference for cluster 1)
    belief_cluster_2,  # e.g., 0.074
    belief_cluster_3,  # e.g., 0.072
    belief_cluster_4,  # e.g., 0.084
    
    # User Embedding (64D) - learned representation from pre-training
    user_emb_0, user_emb_1, ..., user_emb_63
]
```

### User Beliefs Details

**Initialization**:
- Loaded from `user_average_beliefs.json`
- Initial values close to 0 (e.g., `[0.0, 0.0, 0.0, 0.0, 0.0006]`)
- Represents historical user preferences

**Properties**:
- Sum = 1.0 (probability distribution)
- Range: [0, 1] per cluster
- Updated after each step based on accepted items

**Semantic Meaning**:
- `belief[i]` = proportion of items from cluster i that user has accepted
- Reflects user's actual consumption distribution
- Dynamically updated during training

### User Embedding Details

**Source**: Pre-trained embeddings from `user_embedding.pkl`
- 64-dimensional dense vectors
- Captures user characteristics and preferences
- Fixed during RL training (not updated)

**Purpose**:
- Provides rich user context beyond beliefs
- Enables personalization
- Helps TD3 learn user-specific policies

---

## 4. Action Space

### Action Representation (64D)

**Virtual Item Embedding**:
- TD3 actor outputs a 64D vector representing a "virtual item"
- Normalized to [-1, 1] range using tanh activation
- Not a real item, but a target in embedding space

### Action-to-Recommendation Pipeline

```python
# 1. TD3 generates virtual item
virtual_item = actor(state)  # [64D]

# 2. Find similar real items using cosine similarity
similarities = cosine_similarity(virtual_item, item_embeddings)
top_k_items = argsort(similarities)[-50:]  # Top 50 RL recommendations

# 3. Combine with pre-trained recommendations
pretrained_items = pretrained_recommendations[user_id][:10]
final_recommendations = top_k_items[:15] + pretrained_items[:10]  # 25 total

# 4. Present to user
accepted, rejected = simulate_user_interaction(final_recommendations)
```

### Enhanced Item Embeddings

Items use **enhanced embeddings** combining:
- Original item embedding (64D)
- Cluster embedding (64D)
- Formula: `enhanced = normalize(normalize(item) + normalize(cluster))`

**Benefits**:
- Captures both item-specific and cluster-level features
- Improves recommendation quality
- Facilitates cluster-aware recommendations

---

## 5. Reward System

### Step-Based Reward (Current Implementation)

Rewards are calculated **immediately after each user interaction** (not at episode end).

#### Formula

```python
# For each user after accepting items:
reward = improvement_reward + termination_bonus

# Part (a): Improvement-based reward
for cluster_i in range(5):
    current_distance = |optimal_belief[i] - new_belief[i]|
    previous_distance = |optimal_belief[i] - previous_belief[i]|
    
    if current_distance < previous_distance:  # Improvement detected
        # Absolute performance component
        abs_performance = (max_distance - current_distance) / max_distance
        
        # Improvement bonus
        improvement_bonus = previous_distance - current_distance
        
        # Combined reward for this cluster
        cluster_reward[i] = abs_performance + improvement_bonus

improvement_reward = sum(cluster_reward)  # Sum across all improved clusters

# Part (b): Termination bonus
avg_distance = mean(|optimal_belief - new_belief|)
if avg_distance < 0.18:  # Termination threshold
    termination_bonus = 5.0
else:
    termination_bonus = 0.0

total_reward = improvement_reward + termination_bonus
```

#### Key Properties

1. **Immediate Feedback**: Reward calculated after each step
2. **Improvement-Only**: Only rewards clusters that improved
3. **Dual Components**:
   - Absolute performance: How close to optimal
   - Improvement bonus: How much improved
4. **Termination Bonus**: Large reward (5.0) when near-optimal
5. **GPU Accelerated**: Batch computation for efficiency

#### Reward Ranges

- **Typical per-step reward**: 0.0 to 2.0
- **With termination bonus**: up to 7.0
- **Episode total**: 50-200 (accumulated over 50 rounds)

---

## 6. User Belief System

### Count-Based Belief Update

**Core Principle**:
```
belief[i] = cluster_accepted_count[i] / total_items_in_dataset
```

Where `total_items_in_dataset = 19,130` (sum of all cluster sizes)

This creates an **absolute coverage metric** that:
- Sum ≤ 1.0 (equals 1.0 only if user accepted all items in dataset)
- Reflects actual user coverage of the dataset
- Grows toward natural_belief_target as user accepts more items
- Enables direct comparison with natural_belief_target for reward calculation

### Implementation

```python
# Load dataset information
total_items_in_dataset = 19130  # Sum of all cluster sizes
cluster_sizes = [13383, 1355, 1413, 1381, 1598]

# Initialization (based on historical data)
initial_beliefs = [0.0, 0.0, 0.0, 0.0, 0.0006]
# Reverse calculate initial counts from beliefs
initial_counts = initial_beliefs * total_items_in_dataset  # [0, 0, 0, 0, 11.5]

user_accepted_counts[user_id] = {
    'cluster_counts': [0, 0, 0, 0, 12],  # Rounded counts
}

# Update after each step
for item in accepted_items:
    cluster = get_cluster(item)
    user_accepted_counts[user_id]['cluster_counts'][cluster] += 1

# Calculate new beliefs
counts = user_accepted_counts[user_id]['cluster_counts']
new_beliefs = counts / total_items_in_dataset  # Absolute coverage!
```

### Episode Reset

```python
# Reset beliefs to initial state at episode start
user_beliefs[user_id] = initial_beliefs.copy()
user_accepted_counts[user_id] = initial_counts.copy()
```

### Advantages

1. **Absolute Coverage**: Represents actual dataset coverage, not just relative distribution
2. **Direct Comparison**: Can directly compare with natural_belief_target (both use same denominator)
3. **Growth Trajectory**: Beliefs naturally grow toward target as user accepts more items
4. **Interpretable**: `belief[i]` = proportion of cluster i items accepted out of entire dataset
5. **Traceable**: Counts saved for debugging and analysis

---

## 7. User Acceptance Model

### Intelligent Sigmoid-Based Acceptance

Users accept/reject recommendations based on **intelligent probability** that considers:
1. User's belief for the item's cluster
2. Item's distance to its cluster center

#### Formula

```python
acceptance_prob = distance_factor × sigmoid(user_belief × scale) + baseline

where:
    distance_factor = max(0, 1 - distance / max_distance)
    sigmoid(x) = 1 / (1 + exp(-x))
    scale = 10000  # Amplifies small beliefs
    baseline = 0.1  # Minimum 10% acceptance
    max_distance = 5.0  # From empirical data
```

#### Example Calculations

**Scenario 1: Early stage - Low belief, close item**
- user_belief = 0.001 (accepted ~19 items from this cluster)
- distance = 0.5
- distance_factor = 1 - 0.5/5 = 0.9
- sigmoid(0.001 × 10000) = sigmoid(10) ≈ 0.9999
- acceptance_prob = 0.9 × 0.9999 + 0.1 ≈ **0.999** (99.9%)

**Scenario 2: Early stage - Low belief, far item**
- user_belief = 0.001
- distance = 4.0
- distance_factor = 1 - 4.0/5 = 0.2
- sigmoid(0.001 × 10000) ≈ 0.9999
- acceptance_prob = 0.2 × 0.9999 + 0.1 ≈ **0.30** (30%)

**Scenario 3: Late stage - High belief, close item**
- user_belief = 0.7 (accepted ~13,400 items from this cluster, near saturation)
- distance = 0.5
- distance_factor = 0.9
- sigmoid(0.7 × 10000) ≈ 1.0
- acceptance_prob = 0.9 × 1.0 + 0.1 ≈ **1.0** (100%)

#### Key Properties

1. **Sigmoid Transformation**: Converts tiny beliefs (0.001) to meaningful probabilities
2. **Distance Awareness**: Closer items more likely to be accepted
3. **Baseline Exploration**: Minimum 10% acceptance ensures exploration
4. **Differentiation**: Creates 10%-100% acceptance range
5. **Realistic**: ~60% average acceptance rate

### Configuration Parameters

```yaml
environment:
  use_intelligent_acceptance: true
  sigmoid_scale: 10000
  baseline_acceptance_rate: 0.1
  max_cluster_distance: 5.0
```

---

## 8. Training Process

### Episode Structure

```
Episode (repeated 3 times)
├── Round 1
│   ├── For each user (50,000 users)
│   │   ├── Get state (beliefs + embedding)
│   │   ├── TD3 generates action (virtual item)
│   │   ├── Generate 25 recommendations
│   │   ├── Simulate acceptance (~15 accepted)
│   │   ├── Update beliefs (count-based)
│   │   ├── Calculate step reward
│   │   └── Store transition in replay buffer
│   └── Train TD3 (if warmup complete)
├── Round 2
│   └── ... (same as Round 1)
├── ...
└── Round 50
    └── ... (same as Round 1)
```

### Training Parameters

```yaml
training:
  episodes: 3
  rounds_per_episode: 50
  users_per_round: 50000
  batch_size: 1000  # GPU batch processing
  warmup_transitions: 100000  # ~1 round
  train_frequency: 1  # Train every round

td3:
  discount: 0.95
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_freq: 2
  batch_size: 128
  learning_rate: 3e-4
```

### Warmup Phase

**Purpose**: Fill replay buffer with diverse experiences before training

**Duration**: 100,000 transitions ≈ 1 round (50,000 users × 2 transitions)

**Behavior**:
- TD3 actor generates random actions (with noise)
- No TD3 training occurs
- Replay buffer accumulates experiences

### Training Phase

**Frequency**: Every round after warmup

**Process**:
1. Sample batch of 128 transitions from replay buffer
2. Update critics (Q-networks) using TD3 loss
3. Update actor (policy) every 2 critic updates (delayed)
4. Soft update target networks (τ = 0.005)
5. Repeat 10 times per training session

### GPU Acceleration

**Batch Processing**:
- Process 1000 users simultaneously
- Batch state computation
- Batch TD3 inference
- Batch reward calculation
- **Speedup**: 3-4x compared to sequential processing

---

## 9. Data Flow

### Forward Pass (Recommendation Generation)

```
1. User State Construction
   ├── Load user beliefs (5D)
   ├── Load user embedding (64D)
   └── Concatenate → state (69D)

2. TD3 Action Generation
   ├── state → Actor Network
   └── Output: virtual_item (64D)

3. Item Retrieval
   ├── Compute cosine similarity: virtual_item vs all items
   ├── Select top-50 similar items (RL recommendations)
   └── Add 10 pre-trained recommendations

4. User Interaction Simulation
   ├── For each recommended item:
   │   ├── Get item's cluster
   │   ├── Calculate acceptance probability
   │   └── Accept/reject based on probability
   └── Output: accepted_items, rejected_items

5. Belief Update
   ├── For each accepted item:
   │   └── Increment cluster_counts[item_cluster]
   ├── Recalculate beliefs: counts / total
   └── Update user state

6. Reward Calculation
   ├── Compare new_beliefs vs previous_beliefs
   ├── Calculate improvement_reward
   ├── Check termination condition
   └── Add termination_bonus if applicable

7. Store Transition
   └── (state, action, next_state, reward, done) → Replay Buffer
```

### Backward Pass (TD3 Training)

```
1. Sample Batch
   └── Sample 128 transitions from replay buffer

2. Critic Update
   ├── Compute target Q-values using target networks
   ├── Add noise to target actions (policy smoothing)
   ├── Take minimum of twin Q-values (clipped double Q-learning)
   ├── Compute critic loss: MSE(Q, target_Q)
   └── Update critic networks

3. Actor Update (every 2 critic updates)
   ├── Compute policy loss: -Q1(s, actor(s))
   └── Update actor network

4. Target Network Update
   ├── Soft update: target ← τ × network + (1-τ) × target
   └── Apply to both actor and critic targets
```

---

## 10. Key Components

### RecommendationEnvironment

**Purpose**: Simulates user interactions and manages environment state

**Key Methods**:
- `get_user_state(user_id)`: Returns 69D state vector
- `get_batch_user_states(user_ids)`: GPU-accelerated batch state retrieval
- `get_recommendations(user_id, virtual_item)`: Generates recommendations
- `simulate_user_interaction(user_id, items)`: Simulates acceptance/rejection
- `update_user_beliefs_step(user_id, accepted, rejected)`: Updates beliefs
- `calculate_step_reward_gpu(user_id, new_beliefs)`: Computes reward
- `batch_step(user_ids, virtual_items)`: Batch processing for efficiency
- `reset_episode(active_users)`: Resets environment for new episode

**Data Structures**:
```python
users = {
    user_id: {
        'beliefs': np.array([...], dtype=float32),  # 5D
        'pp1_distance': float,  # Distance to natural target
        'cluster_distances': np.array([...])  # Fixed embedding distances
    }
}

user_accepted_counts = {
    user_id: {
        'cluster_counts': np.array([...]),  # Counts per cluster
        'total_count': int  # Total accepted items
    }
}
```

### RecommendationTrainer

**Purpose**: Manages training loop and coordinates components

**Key Methods**:
- `run_round(round_num)`: Executes one round of training
- `run_episode(episode_num)`: Executes one complete episode
- `train_td3()`: Trains TD3 agent using replay buffer
- `train()`: Main training loop
- `save_model(suffix)`: Saves TD3 model checkpoint
- `save_training_stats()`: Saves training statistics
- `save_current_user_beliefs(episode_num)`: Saves user beliefs

**Training Statistics**:
```python
episode_stats = {
    'episode': int,
    'total_transitions': int,
    'avg_acceptance_rate': float,
    'belief_improvements': int,
    'total_step_rewards': float,
    'avg_step_reward': float,
    'total_reward': float,
    'avg_round_reward': float,
    'reward_change': float,  # vs previous episode
    'avg_belief_distance_to_optimal': float
}
```

### TD3 Agent

**Purpose**: Learns optimal policy for recommendation generation

**Components**:
- **Actor**: Maps states to actions (virtual items)
- **Critic**: Estimates Q-values for state-action pairs
- **Target Networks**: Stabilize training
- **Replay Buffer**: Stores experiences for off-policy learning

**Key Features**:
- Twin critics (reduces overestimation bias)
- Delayed policy updates (improves stability)
- Target policy smoothing (regularization)

### Data Files

**Input Data**:
- `user_average_beliefs.json`: Initial user beliefs
- `user_embedding.pkl`: 64D user embeddings
- `item_embedding.pkl`: 64D item embeddings
- `cluster_embeddings.pkl`: 64D cluster embeddings
- `recommendations.pkl`: Pre-trained recommendations
- `item_token_map.json`: Item ID to token mapping
- `user_token_map.json`: User ID to token mapping
- `cluster_matrix_manifest_K5_topk5.json`: Cluster assignments

**Output Data**:
- `training_all_episodes.json`: Complete training statistics
- `user_beliefs_episode_N.json`: User beliefs per episode
- `current_user_beliefs.json`: Latest user beliefs
- `recommendations_epN.json`: Recommendations per episode
- `td3_recommendation_TIMESTAMP_*.pt`: Model checkpoints

---

## Performance Characteristics

### Computational Complexity

**Per Round**:
- State retrieval: O(batch_size × state_dim) = O(1000 × 69)
- TD3 inference: O(batch_size × network_params) ≈ O(1000 × 50K)
- Similarity computation: O(batch_size × num_items × emb_dim) = O(1000 × 50K × 64)
- Belief updates: O(batch_size × num_clusters) = O(1000 × 5)
- Reward calculation: O(batch_size × num_clusters) = O(1000 × 5)

**Per Episode** (50 rounds):
- Total users processed: 50,000 × 50 = 2,500,000
- Total transitions: 2,500,000
- TD3 training steps: 50 rounds × 10 updates = 500

### Memory Usage

- **Replay Buffer**: ~1 GB (1M transitions × 1KB each)
- **User Beliefs**: ~10 MB (50K users × 5 floats × 4 bytes)
- **Embeddings**: ~30 MB (50K items × 64 floats × 4 bytes)
- **TD3 Model**: ~5 MB (network parameters)
- **Total**: ~1.05 GB

### Training Time

**Per Round** (with GPU):
- Batch processing: ~2-3 seconds
- TD3 training: ~0.5 seconds
- Total: ~3 seconds per round

**Per Episode** (50 rounds):
- ~150 seconds (2.5 minutes)

**Full Training** (3 episodes):
- ~450 seconds (7.5 minutes)

---

## Design Decisions and Rationale

### 1. Count-Based Belief Update

**Decision**: Use count-based distribution instead of fixed increments

**Rationale**:
- Semantically consistent with natural_belief_target
- Automatic normalization (sum = 1)
- Interpretable and traceable
- No manual tuning of learning rates

### 2. Step-Based Rewards

**Decision**: Calculate rewards after each step, not at episode end

**Rationale**:
- Immediate feedback for TD3
- Faster learning convergence
- Better credit assignment
- Enables real-time belief tracking

### 3. Sigmoid-Based Acceptance

**Decision**: Use sigmoid transformation for acceptance probability

**Rationale**:
- Transforms tiny beliefs (0.001) to meaningful probabilities
- Creates realistic acceptance rates (~60%)
- Maintains differentiation across belief ranges
- Ensures exploration (10% baseline)

### 4. Enhanced Item Embeddings

**Decision**: Combine item and cluster embeddings

**Rationale**:
- Captures both item-specific and cluster-level features
- Improves recommendation quality
- Facilitates cluster-aware recommendations
- Better alignment with belief system

### 5. GPU Batch Processing

**Decision**: Process users in batches of 1000

**Rationale**:
- 3-4x speedup over sequential processing
- Efficient GPU utilization
- Reduced CPU-GPU transfer overhead
- Scalable to large user bases

### 6. Natural Belief Target

**Decision**: Use data-derived target `[0.700, 0.071, 0.074, 0.072, 0.084]`

**Rationale**:
- Reflects actual content distribution
- Achievable and realistic
- Balances diversity and preference
- Grounded in empirical data

---

## Future Improvements

### Potential Enhancements

1. **Dynamic Learning Rates**: Adapt belief update rates based on convergence
2. **Multi-Objective Rewards**: Balance diversity, engagement, and satisfaction
3. **Hierarchical Clustering**: Support more granular content categories
4. **User Segmentation**: Different policies for different user types
5. **Online Learning**: Continuous adaptation to new data
6. **Exploration Strategies**: More sophisticated exploration mechanisms
7. **Contextual Features**: Incorporate time, location, device, etc.
8. **Multi-Armed Bandits**: Hybrid RL-bandit approach

### Scalability Considerations

1. **Distributed Training**: Multi-GPU or multi-node training
2. **Approximate Nearest Neighbors**: Faster similarity search
3. **Model Compression**: Smaller networks for deployment
4. **Incremental Updates**: Update only active users
5. **Caching**: Cache embeddings and similarities

---

## References

### Papers
- TD3: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)
- Filter Bubbles: "The Filter Bubble: What the Internet Is Hiding from You" (Pariser, 2011)

### Code Structure
- `run_recommendation_rl.py`: Main entry point
- `recommendation_trainer.py`: Training orchestration
- `recommendation_environment.py`: Environment simulation
- `TD3.py`: TD3 algorithm implementation
- `utils.py`: Replay buffer and utilities
- `config.yaml`: Configuration parameters

---

**Document Version**: 1.0  
**Last Updated**: 2024-11-09  
**Status**: Production
