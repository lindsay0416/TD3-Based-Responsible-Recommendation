# Recommendation System Reinforcement Learning Model Design

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 Recommendation RL Training Framework             │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer    │  Environment    │  Agent Layer    │  Training   │
│  ──────────    │  ──────────     │  ──────────     │  ──────────  │
│  User Beliefs  │  Rec Environment│  TD3 Agent      │  Controller │
│  User Embed    │  Reward Calc    │  Actor Network  │  Experience │
│  Item Embed    │  State Mgmt     │  Critic Network │  Model Save │
│  Cluster Data  │  Interaction    │  Target Network │  Statistics │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Training Hierarchy & Flow

```
Training Process (Complete Learning)
│
├── Episode 1 (Complete Learning Cycle)
│   ├── reset_episode() - Reset user beliefs to initial state
│   ├── Round 1 (Batch Interaction Session)
│   │   ├── next_round() - Clear interaction records
│   │   ├── Batch 1 (Parallel Processing)
│   │   │   ├── Step 1: User_1 → State → Action → Reward
│   │   │   ├── Step 2: User_2 → State → Action → Reward
│   │   │   └── Step N: User_N → State → Action → Reward
│   │   ├── Batch 2...
│   │   └── Batch M
│   ├── Round 2...
│   ├── Round 20
│   ├── update_user_beliefs() - Calculate belief changes
│   ├── calculate_episode_rewards() - Compute rewards
│   └── train_TD3() - Train agent networks
│
├── Episode 2...
└── Episode 10
```

## 🎯 Core Model Components

### 1. State Space Design
```
User State = [User Beliefs (5D) + User Embedding (64D)] = 69D

┌─────────────────┬─────────────────────────────────────────────┐
│ User Beliefs(5D)│              User Embedding(64D)            │
├─────────────────┼─────────────────────────────────────────────┤
│ [0.2, 0.3,      │ [0.1, -0.2, 0.5, ..., 0.3]                │
│  0.1, 0.4, 0.6] │ (Pre-trained user representation)          │
└─────────────────┴─────────────────────────────────────────────┘
```

### 2. Action Space Design
```
Virtual Item Vector = 64D continuous vector, range [-1, 1]

TD3 Output → [0.2, -0.3, 0.8, ..., -0.1] → Cosine similarity matching to real items
```

### 3. Reward Mechanism
```
Episode-level Reward = Distance-based belief improvement reward

if user beliefs move toward optimal targets:
    reward = positive (based on improvement magnitude)
else:
    reward = negative or 0
```

## 🧠 TD3 Neural Network Architecture

### Actor Network
```
Input Layer (69D) → Hidden Layer 1 (512D) → Hidden Layer 2 (512D) 
                 → Hidden Layer 3 (256D) → Output Layer (64D)

Activation: ReLU for hidden layers, Tanh for output
Normalization: Batch Normalization after each hidden layer
Regularization: Dropout (0.1) after each hidden layer
Output Range: [-1, 1] (scaled by max_action)
```

### Critic Network (Twin Networks)
```
Q1 Network:
State+Action (133D) → Hidden Layer 1 (512D) → Hidden Layer 2 (512D) 
                   → Hidden Layer 3 (256D) → Q-value (1D)

Q2 Network: (Same architecture as Q1)
State+Action (133D) → Hidden Layer 1 (512D) → Hidden Layer 2 (512D) 
                   → Hidden Layer 3 (256D) → Q-value (1D)

Activation: ReLU for all hidden layers
Normalization: Batch Normalization after each hidden layer
Regularization: Dropout (0.1) after each hidden layer
```

## ⚙️ Hyperparameters Configuration

### Training Parameters
```yaml
episodes: 10                    # Total training episodes
rounds_per_episode: 20          # Rounds per episode
users_per_round: 50000         # Users processed per round
batch_size: 4000               # Users per batch (increased for efficiency)
warmup_transitions: 100000     # Transitions before TD3 training starts
train_frequency: 3             # Train TD3 every N rounds
```

### TD3 Algorithm Parameters
```yaml
discount: 0.95                 # Gamma - future reward discount factor
tau: 0.005                     # Target network soft update rate
policy_noise: 0.2              # Noise added to target policy
noise_clip: 0.5                # Range to clip target policy noise
policy_freq: 2                 # Delayed policy update frequency
batch_size: 512                # Training batch size (increased)
learning_rate: 3e-4            # Learning rate for both networks
```

### Environment Parameters
```yaml
state_dim: 69                  # User beliefs (5D) + user embedding (64D)
action_dim: 64                 # Virtual item representation dimension
max_action: 1.0                # Maximum action value
user_acceptance_rate: 0.25     # Fallback acceptance probability
use_intelligent_acceptance: true    # Use sigmoid-based acceptance

# Sigmoid Activation Parameters
sigmoid_scale: 10000           # Scale factor for sigmoid activation
baseline_acceptance_rate: 0.1  # Baseline acceptance rate (minimum 10%)
max_cluster_distance: 5        # Maximum distance in each cluster
```

### Recommendation Parameters
```yaml
rl_list_size: 10              # Items from RL recommendations
total_recommendations: 20      # Total items shown (RL + pre-trained)
similarity_metric: "cosine"    # Similarity metric for item matching
```

### Reward Parameters
```yaml
max_reward: 100.0             # Maximum reward value
improvement_threshold: 0.01    # Minimum improvement for reward
use_progressive_reward: false  # Use hybrid vs strict improvement
maintenance_reward_factor: 0.7 # Reward for maintaining performance
exploration_bonus: 0.2        # Bonus for exploration
use_combined_metric: true     # Use engagement + alignment metrics
```

## 📊 Data Flow Diagram

```
User Data → Recommendation Environment ← Item Data
    ↓                    ↓                  ↑
State (69D) → TD3 Actor Network → Action (64D) → Item Matching
    ↓                    ↓                  ↓
Experience → Replay Buffer ← User Feedback ← Recommendation List
    ↓                    ↓                  ↓
TD3 Training ← Reward Calculation ← User Acceptance/Rejection
```

## 🔧 Model Implementation Details

### Network Initialization
```python
# Actor Network
self.actor = Actor(state_dim=69, action_dim=64, max_action=1.0)
- Layer sizes: [69, 512, 512, 256, 64]
- Batch normalization after each hidden layer
- Dropout rate: 0.1
- Activation: ReLU → ReLU → ReLU → Tanh

# Critic Networks (Twin)
self.critic = Critic(state_dim=69, action_dim=64)
- Input size: 69 + 64 = 133 (state + action)
- Layer sizes: [133, 512, 512, 256, 1]
- Two identical networks for TD3's twin Q-learning
```

### Training Process
```python
# Experience Collection
for episode in range(10):
    reset_episode()  # Reset user beliefs to initial state
    for round in range(20):
        next_round()  # Clear round-level interactions
        for batch in user_batches:
            states = get_batch_user_states(batch_users)
            actions = actor(states)  # TD3 inference
            results = environment.batch_step(batch_users, actions)
            store_experiences(results)
    
    # Training
    update_user_beliefs()  # Calculate belief changes
    rewards = calculate_episode_rewards()  # Distance-based rewards
    train_td3_networks()  # Update actor and critic networks
```

### Reward Calculation
```python
def calculate_distance_based_reward(current_beliefs, previous_beliefs):
    reward = 0
    for i in range(len(current_beliefs)):
        current_distance = abs(optimal_beliefs[i] - current_beliefs[i])
        previous_distance = abs(optimal_beliefs[i] - previous_beliefs[i])
        if current_distance < previous_distance:
            improvement = previous_distance - current_distance
            reward += improvement * reward_scale
    return reward
```

## 🎮 Interaction Flow

### Single Step Process
```
1. Get User State
   ├── User beliefs: [0.2, 0.3, 0.1, 0.4, 0.6]
   └── User embedding: [64D vector]

2. TD3 Agent Decision
   ├── Input: 69D state vector
   ├── Actor network inference
   └── Output: 64D virtual item

3. Environment Execution
   ├── Cosine similarity matching to real items
   ├── Generate recommendation list (RL + pre-trained)
   └── Simulate user acceptance/rejection (see Acceptance Rate Calculation)

4. Learning and Feedback
   ├── Record user interactions
   ├── Update user beliefs immediately after acceptance
   ├── Store experience in replay buffer
   └── Periodic TD3 network training
```

## 🎲 User Acceptance Rate Calculation

### Sigmoid Activation Method

The system uses **Sigmoid activation** to transform tiny user belief values into meaningful acceptance probabilities, while keeping original data unchanged.

#### Formula
```
acceptance_rate = (1 - distance/max_distance) × sigmoid(user_belief × scale) + baseline

where:
  sigmoid(x) = 1 / (1 + e^(-x))
  scale = 10000
  baseline = 0.1
  max_distance = 5
```

#### Components

**1. Distance Factor**
```
distance_factor = 1 - (item_distance / max_cluster_distance)

Range: [0, 1]
- distance = 1 (near center)  → factor = 0.8
- distance = 3 (mid)          → factor = 0.4
- distance = 5 (far from center) → factor = 0.0
```

**2. Sigmoid Activation**
```
belief_activated = sigmoid(user_belief × 10000)

Examples:
- user_belief = 0.0006 → sigmoid(6.0) = 0.998
- user_belief = 0.0002 → sigmoid(2.0) = 0.921
- user_belief = 0.0004 → sigmoid(4.0) = 0.991
- user_belief = 0.0000 → sigmoid(0.0) = 0.500
```

**3. Final Acceptance Rate**
```
acceptance_rate = distance_factor × belief_activated + 0.1

Ensures:
- Minimum 10% acceptance (exploration)
- Maximum ~100% acceptance (strong preference + near item)
- Meaningful differentiation based on preference and quality
```

#### Acceptance Rate Examples

**User with Strong Preference (belief = 0.0006)**
| Item Distance | Distance Factor | Sigmoid Value | Acceptance Rate |
|--------------|----------------|---------------|-----------------|
| 1 (near)     | 0.8            | 0.998         | **89.8%** |
| 3 (mid)      | 0.4            | 0.998         | **49.9%** |
| 4 (far)      | 0.2            | 0.998         | **30.0%** |
| 5 (farthest) | 0.0            | 0.998         | **10.0%** |

**User with Moderate Preference (belief = 0.0002)**
| Item Distance | Distance Factor | Sigmoid Value | Acceptance Rate |
|--------------|----------------|---------------|-----------------|
| 1 (near)     | 0.8            | 0.921         | **83.7%** |
| 3 (mid)      | 0.4            | 0.921         | **46.8%** |
| 4 (far)      | 0.2            | 0.921         | **28.4%** |
| 5 (farthest) | 0.0            | 0.921         | **10.0%** |

**User with No Preference (belief = 0.0)**
| Item Distance | Distance Factor | Sigmoid Value | Acceptance Rate |
|--------------|----------------|---------------|-----------------|
| 1 (near)     | 0.8            | 0.500         | **50.0%** |
| 3 (mid)      | 0.4            | 0.500         | **30.0%** |
| 4 (far)      | 0.2            | 0.500         | **20.0%** |
| 5 (farthest) | 0.0            | 0.500         | **10.0%** |

#### Key Observations

1. **Preserves Data Integrity**: Original user beliefs remain unchanged (no artificial inflation)
2. **Meaningful Differentiation**: Acceptance rates range from 10% to 90%
3. **Quality Matters**: High-quality items (near cluster center) have higher acceptance
4. **Exploration Enabled**: Even with no preference, near items have 50% acceptance
5. **Personalization**: Different users show distinct acceptance patterns

#### Implementation Flow

```python
# For each recommended item
for item_id in recommended_items:
    # 1. Get user's belief for item's cluster
    user_belief = user_beliefs[item_cluster]
    
    # 2. Get item's distance to cluster center
    distance = cluster_distances[item_id]
    
    # 3. Calculate distance factor
    distance_factor = max(0.0, 1.0 - (distance / 5.0))
    
    # 4. Apply sigmoid activation
    belief_activated = 1.0 / (1.0 + np.exp(-user_belief * 10000))
    
    # 5. Calculate final acceptance rate
    acceptance_rate = distance_factor * belief_activated + 0.1
    
    # 6. Simulate acceptance decision
    if random.random() < acceptance_rate:
        accepted_items.append(item_id)
    else:
        rejected_items.append(item_id)
```

#### Advantages

1. **No Data Modification**: Keeps original beliefs for RL training
2. **Logical Consistency**: User beliefs never exceed optimal targets
3. **Exploration-Exploitation Balance**: Baseline 10% ensures exploration
4. **Quality-Aware**: Item quality (distance) significantly impacts acceptance
5. **Mathematically Stable**: Sigmoid naturally bounds output to [0,1]

## 🔄 User Belief Update Mechanism

### Real-time Belief Updates

User beliefs are updated **immediately after each recommendation** in every round, not at the end of episodes.

#### Update Timing

```
Episode 1
├── Round 1
│   ├── User U100: Generate recommendations → Accept items → Update beliefs ✓
│   ├── User U1000: Generate recommendations → Accept items → Update beliefs ✓
│   └── User U10001: Generate recommendations → Accept items → Update beliefs ✓
├── Round 2 (uses updated beliefs from Round 1)
│   ├── User U100: Generate recommendations → Accept items → Update beliefs ✓
│   └── ...
└── Round 20
    └── Final beliefs saved to current_user_beliefs.json
```

#### Update Formula

```python
# For each accepted item
for item_id in accepted_items:
    item_cluster = get_cluster(item_id)
    belief_updates[item_cluster] += 0.01  # Increment per accepted item

# Apply updates
new_beliefs = current_beliefs + belief_updates

# Apply constraints
new_beliefs = max(new_beliefs, 0)  # Non-negative
if new_beliefs.sum() > 1.0:
    new_beliefs = new_beliefs / new_beliefs.sum()  # Normalize
new_beliefs = constrain_to_optimal(new_beliefs)  # Don't exceed optimal

# Update immediately
user.beliefs = new_beliefs
```

#### Update Example

**Round 1 - User U100**
```
Initial beliefs:     [0.000, 0.000, 0.000, 0.000, 0.001]
Recommended 20 items → Accepted 3 items from cluster 4
Belief updates:      [0.000, 0.000, 0.000, 0.000, 0.030]
New beliefs:         [0.000, 0.000, 0.000, 0.000, 0.031] ← Updated immediately!
```

**Round 2 - User U100** (uses updated beliefs)
```
Current beliefs:     [0.000, 0.000, 0.000, 0.000, 0.031] ← From Round 1
Recommended 20 items → Accepted 5 items from cluster 4
Belief updates:      [0.000, 0.000, 0.000, 0.000, 0.050]
New beliefs:         [0.000, 0.000, 0.000, 0.000, 0.081] ← Updated again!
```

**Round 20 - User U100**
```
Final beliefs:       [0.000, 0.000, 0.000, 0.000, 0.450]
Saved to: results/current_user_beliefs.json
```

#### Belief Storage

**In-Memory Storage**
```python
# During training (in recommendation_environment.py)
self.users[user_id]['beliefs'] = new_beliefs  # Updated every round
```

**Persistent Storage**
```python
# After each episode (in recommendation_trainer.py)
save_current_user_beliefs(episode_num)
→ Saves to: results/current_user_beliefs.json

# File structure:
{
    "episode": 1,
    "timestamp": "2025-01-08 10:30:00",
    "total_users": 50000,
    "user_beliefs": {
        "U100": {
            "beliefs": [0.0, 0.0, 0.0, 0.0, 0.45],
            "pp1_distance": 0.123,
            "cluster_distances": [0.699, 0.071, 0.074, 0.072, 0.084]
        },
        ...
    }
}
```

#### Update Constraints

1. **Non-negative**: `beliefs >= 0`
2. **Probability constraint**: `sum(beliefs) <= 1.0`
3. **Optimal constraint**: `beliefs[i] <= optimal_beliefs[i]`
4. **Increment size**: `+0.01 per accepted item`

#### Impact on Training

**Immediate Effects**
```
Round N beliefs → Affect Round N+1 acceptance rates
                → Affect Round N+1 recommendations
                → Affect Round N+1 rewards
```

**Cumulative Learning**
```
Episode 1: beliefs evolve over 20 rounds
Episode 2: reset to initial beliefs, learn again
Episode 3: reset to initial beliefs, learn again
...
TD3 Agent: learns optimal strategies across all episodes
```

#### Key Characteristics

1. **Real-time**: Updates happen immediately after each recommendation
2. **Cumulative**: Beliefs accumulate changes within an episode
3. **Reset**: Beliefs reset to initial values at the start of each episode
4. **Persistent**: Final beliefs saved to `current_user_beliefs.json` after each episode
5. **Constrained**: Always respect non-negative, sum, and optimal constraints

#### Belief Update Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Generate Recommendations (20 items)                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Calculate Acceptance Rate (Sigmoid for each item)        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Simulate User Interaction (Accept/Reject)                │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Update Beliefs Immediately                               │
│    - For each accepted item: belief[cluster] += 0.01        │
│    - Apply constraints (non-negative, sum, optimal)         │
│    - Store in memory: self.users[user_id]['beliefs']        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Calculate Step Reward (based on new beliefs)             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Store Transition (state, action, next_state, reward)     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Next Round Uses Updated Beliefs                          │
└─────────────────────────────────────────────────────────────┘
```

This real-time update mechanism ensures that:
- User preferences evolve naturally based on accepted items
- The RL agent learns to guide users toward optimal beliefs
- Each round builds upon the previous round's belief state
- Training reflects realistic user behavior dynamics

## 📈 Key Performance Metrics

### Episode-level Metrics
```
- Acceptance Rate: 35.1%
- Belief Improvements: 30917/50000 (61.8%)
- Average Reward: 6.81
- Reward Trend: -118.6 (last - first round)
- Total Reward: 7,030,485
```

### Training Efficiency Metrics
```
- Processing Time: 2.3s per round
- TD3 Training Frequency: Every 3 rounds
- Batch Processing: 4000 users per batch
- Total Interactions: 10M steps (10 episodes × 20 rounds × 50K users)
```

## 🔬 Experimental Design Philosophy

### Stable User Group Assumption
```
Design Principles:
├── User preferences stable → Reset beliefs to initial state between episodes
├── Agent learning → TD3 parameters accumulate across episodes
├── Environment consistency → Same conditions for different strategies
└── Reproducible experiments → Control variables, focus on algorithm
```

### Learning Objective
```
Goal: Learn optimal recommendation strategies for users with stable preferences

Core Question: Given users' true preferences, how can recommendations guide 
              users to discover and recognize their latent interests?

Not Learning: How to adapt to changing user preferences
But Learning: How to provide optimal recommendations for stable preferences
```

## 🎯 Model Advantages

1. **Scientific Experimental Design**: Controls user variables, focuses on algorithm optimization
2. **Scalable Architecture**: Batch processing with efficient neural networks
3. **Stable Training Process**: Gradient clipping + batch normalization + experience replay
4. **Comprehensive Monitoring**: Multi-level metrics with real-time tracking
5. **Modular Design**: Easy to extend and modify components

This design is particularly well-suited for studying recommendation algorithm optimization in stable user populations, making it ideal for research and development of recommendation system core algorithms.


Episode 1:
  ┌─────────────────────────────────────────┐
  │ reset_episode()                         │
  │ - U100: beliefs = [0.000, 0.000, ...]  │ ← 重置到初始值
  │ - U1000: beliefs = [0.000, 0.000, ...] │
  └─────────────────────────────────────────┘
  
  Round 1:
    U100:  [0.000, ...] → action → [0.031, ...] ✅
    U1000: [0.000, ...] → action → [0.032, ...] ✅
  
  next_round() ← 只增加计数，不重置beliefs
  
  Round 2:
    U100:  [0.031, ...] → action → [0.062, ...] ✅ 连续！
    U1000: [0.032, ...] → action → [0.064, ...] ✅ 连续！
  
  Round 3:
    U100:  [0.062, ...] → action → [0.093, ...] ✅ 连续！
    ...
  
  Round 50:
    U100:  [...] → action → [0.650, ...] ✅

Episode 2:
  ┌─────────────────────────────────────────┐
  │ reset_episode()                         │
  │ - U100: beliefs = [0.000, 0.000, ...]  │ ← 重新开始
  │ - U1000: beliefs = [0.000, 0.000, ...] │
  └─────────────────────────────────────────┘
  
  Round 1:
    U100:  [0.000, ...] → action → [0.031, ...] ✅ 从头开始
    ...