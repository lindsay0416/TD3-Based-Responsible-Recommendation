# Recommendation RL System using TD3

A reinforcement learning system that uses Twin Delayed Deep Deterministic Policy Gradient (TD3) to reduce filter bubbles in recommendation systems by optimizing user belief distributions.

## 🎯 Project Overview

This system trains a TD3 agent to generate virtual item representations that help users discover diverse content, moving their belief distributions closer to optimal (natural) distributions and reducing filter bubble effects.

### Key Features
- **TD3-based Recommendation**: Uses continuous action space to generate virtual item representations
- **Filter Bubble Reduction**: Measures and optimizes user belief distances from natural distributions
- **Pentagon Geometry**: Visualizes 5-cluster belief distributions as pentagon shapes
- **Progressive Rewards**: Only rewards improvements in belief diversity
- **Configurable Training**: Easy parameter adjustment via YAML configuration

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Validate data and setup
python setup_environment.py
```

### 2. Run Training (Single Command)
```bash
# Start training with default configuration
python run_recommendation_rl.py

# Or validate configuration first
python run_recommendation_rl.py --validate-only

# Use custom configuration
python run_recommendation_rl.py --config my_config.yaml
```

### 3. Monitor Results
Training results are saved in:
- `results/` - Training statistics and final user beliefs
- `models/` - Trained TD3 models
- `logs/` - Training logs

## 📊 System Architecture

### Data Flow
1. **State**: User beliefs (5D) + User embedding (64D) = 69D state
2. **Action**: TD3 outputs 64D virtual item representation
3. **Recommendations**: Cosine similarity matching → Top-20 items + pre-trained recommendations
4. **User Simulation**: 50% acceptance probability per item
5. **Belief Update**: Recalculate user beliefs after each round (10k episodes)
6. **Reward**: Progressive reward based on belief distance improvements

### Training Structure
- **50 Rounds** × **10,000 Episodes** each
- **50,000 Users** per episode
- **Belief updates** after each round
- **Progressive rewards** only for improvements

## ⚙️ Configuration

All parameters are configurable via `config.yaml`:

```yaml
training:
  rounds: 50                    # Number of training rounds
  episodes_per_round: 10000     # Episodes per round
  users_per_episode: 50000      # Users per episode

td3:
  discount: 0.95               # Gamma (discount factor)
  batch_size: 256              # Training batch size
  learning_rate: 3e-4          # Learning rate

environment:
  user_acceptance_rate: 0.5    # User acceptance probability
  
reward:
  max_reward: 10.0             # Maximum reward value
  use_progressive_reward: true # Only reward improvements
```

## 📁 Project Structure

```
├── run_recommendation_rl.py          # 🎯 MAIN ENTRY POINT
├── config.yaml                       # Configuration file
├── setup_environment.py              # Environment setup and validation
├── recommendation_trainer.py         # Main training loop
├── recommendation_environment.py     # RL environment implementation
├── requirements.txt                  # Python dependencies
│
├── TD3.py                            # TD3 algorithm implementation
├── utils.py                          # Replay buffer and utilities
│
├── data_preprocessing/               # Data processing scripts
│   ├── calculate_user_average_beliefs.py
│   ├── calculate_belief_central_distances.py
│   └── README.md
│
├── processed_data/                   # Processed data files
│   ├── user_average_beliefs.json    # User beliefs with PP1 distances
│   ├── cluster_embeddings.pkl       # Cluster embeddings
│   └── natural_belief_target.pkl    # Optimal belief distribution
│
├── embeddings/                       # Raw embedding data
│   ├── user_embedding.pkl
│   ├── item_embedding.pkl
│   └── recommendations.pkl
│
├── results/                          # Training results
├── models/                           # Saved models
└── logs/                            # Training logs
```

## 🔬 Key Concepts

### Filter Bubble Measurement
- **PP1 Distance**: Distance between user belief center and optimal center
- **Cluster Distances**: Individual deviations per cluster (AA1, BB1, CC1, DD1, EE1)
- **Pentagon Geometry**: 5D beliefs visualized as pentagon shapes

### Reward Function
```python
# Combined Engagement + Alignment reward system
engagement_score = min(belief_sum / target_engagement, 1.0)
alignment_score = cosine_similarity(normalized_beliefs, normalized_optimal)
combined_score = engagement_score * alignment_score

base_reward = combined_score * max_reward
score_change = current_combined_score - previous_combined_score

if score_change > improvement_threshold:
    reward = base_reward + exploration_bonus  # Significant improvement
elif score_change >= -improvement_threshold:
    reward = base_reward * maintenance_factor  # Maintenance (70% reward)
else:
    reward = base_reward * 0.1  # Small penalty for regression
```

### State-Action Space
- **State**: 69D (5D user beliefs + 64D user embedding)
- **Action**: 64D virtual item representation (continuous)
- **Recommendations**: Cosine similarity → real items

## 📊 Reward Metrics

The system tracks multiple reward types:

### Episode-Level Metrics
- **total_reward**: Sum of all round rewards in the episode (immediate performance)
- **avg_round_reward**: Average reward per round in the episode (total_reward / num_rounds)
- **belief_improvement_reward**: Sum of belief improvement rewards (learning progress)
- **belief_improvements**: Count of users who improved their beliefs
- **avg_belief_reward**: Average reward per user for belief improvements

### Round-Level Metrics  
- **round_reward**: Immediate recommendation performance for that round
- **belief_improvements**: Users who showed improvement in that round

## 📈 Expected Results

The system should progressively:
1. **Reduce PP1 distances** (users closer to optimal belief center)
2. **Decrease cluster distances** (more balanced content consumption)
3. **Increase belief diversity** (reduced filter bubble effects)
4. **Improve recommendation acceptance** while maintaining relevance

## 🛠️ Customization

### Modify Training Parameters
Edit `config.yaml` to change:
- Number of rounds/episodes
- TD3 hyperparameters
- Reward function parameters
- User acceptance rates

### Extend the System
- **Custom Reward Functions**: Modify `RecommendationEnvironment.calculate_round_rewards()`
- **Different Belief Updates**: Customize `_calculate_new_beliefs()`
- **Alternative Similarity Metrics**: Change recommendation matching in `_get_rl_recommendations()`

## 📚 Data Requirements

Ensure these files exist before training:
- `processed_data/user_average_beliefs.json` (50k users with beliefs)
- `embeddings/user_embedding.pkl` (user embeddings)
- `embeddings/item_embedding.pkl` (item embeddings)
- `processed_data/cluster_embeddings.pkl` (5 cluster embeddings)
- `embeddings/recommendations.pkl` (pre-trained recommendations)

## 🔍 Troubleshooting

### Common Issues
1. **Missing Data Files**: Run `python setup_environment.py` to validate
2. **Memory Issues**: Reduce `batch_size` in config.yaml
3. **Slow Training**: Reduce `episodes_per_round` for testing
4. **GPU Issues**: Set `device: "cpu"` in config.yaml

### Debug Mode
```bash
python run_recommendation_rl.py --verbose
```

---

## 📄 Original TD3 Implementation

This project builds upon the original TD3 implementation. For the original MuJoCo experiments, see the legacy files (main.py, run_experiments.sh).

### Original TD3 Paper
```
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}
```
