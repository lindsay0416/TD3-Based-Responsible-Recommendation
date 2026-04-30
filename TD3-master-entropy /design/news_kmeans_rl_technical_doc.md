# News K-Means RL 系统技术文档

## 1. 系统目标：打破信息茧房

用户在新闻推荐系统中会陷入"信息茧房"——只看自己喜欢的类型，越看越窄。

我们的目标：**用 RL 引导推荐系统，让用户的阅读分布趋近于"自然分布"**，即新闻内容本身的类别比例。

### 1.1 什么是 Cluster？

所有新闻 item（11,370 个）通过 K-Means 聚类分成 **5 个 cluster**：

| Cluster | 新闻数量 | 自然比例（target belief） |
|---------|---------|------------------------|
| 0       | 1,987   | 0.1748                 |
| 1       | 2,161   | 0.1901                 |
| 2       | 2,385   | 0.2098                 |
| 3       | 2,463   | 0.2166                 |
| 4       | 2,374   | 0.2088                 |

**Natural Belief Target** = `[0.1748, 0.1901, 0.2098, 0.2166, 0.2088]`

这个 target 就是我们希望每个用户最终达到的阅读比例 （越接近越好，不用一模一样）。

### 1.2 什么是 User Belief？

每个用户有一个 5 维向量 `beliefs[5]`，表示他在每个 cluster 上的"偏好程度"。

计算方式：
```
belief[i] = 用户接受的 cluster_i 的 item 数 / 用户接受的总 item 数
```

- TD3-master-entropy /processed_data/news/K_means/user_average_beliefs.json
里保存了用户对每一个 cluster的原始 belief，是用数据集计算的 （TD3-master-entropy /data/news/new_behaviors.tsv）

**当前 news 用户的问题**：平均 belief sum 只有 0.087，说明用户接受率很低，大部分 item 被拒绝。各 cluster 的平均 belief：
```
[0.0096, 0.0227, 0.0152, 0.0258, 0.0135]
```
离 target `[0.1748, 0.1901, 0.2098, 0.2166, 0.2088]` 差距很大。

**RL 的任务就是缩小这个差距。**

---

## 2. RL 框架总览

```
┌─────────────────────────────────────────────────────┐
│                    Training Loop                     │
│                                                      │
│  for episode in 1..100:                              │
│    reset_episode()  # 重置 beliefs 到初始值           │
│    for round in 1..50:                               │
│      for batch of users:                             │
│        state = [beliefs(5D), user_embedding(64D)]    │
│        action = TD3_Actor(state)  → virtual_item(64D)│
│        recommendations = find_similar_items(action)  │
│        accepted, rejected = simulate_interaction()   │
│        new_beliefs = update_beliefs(accepted)        │
│        reward = calculate_reward(new_beliefs)        │
│        store (s, a, s', r) in replay buffer          │
│      train TD3 from replay buffer                    │
└─────────────────────────────────────────────────────┘
```

### 2.1 关键维度

| 概念 | 维度 | 说明 |
|------|------|------|
| User Embedding | 64D | LightGCN 训练得到 |
| Item Embedding | 64D | LightGCN 训练得到 |
| Beliefs | 5D | 5 个 cluster 的偏好比例 |
| **State** | **69D** | beliefs(5) + user_embedding(64) |
| **Action** | **64D** | TD3 输出的 virtual item embedding |

---

## 3. TD3 网络结构

### 3.1 Actor（策略网络）

输入 state(69D)，输出 action(64D)——一个"虚拟 item"的 embedding。

```python
class Actor(nn.Module):
    def __init__(self, state_dim=69, action_dim=64, max_action=1.0):
        self.l1 = nn.Linear(69, 128)    # 输入层
        self.l2 = nn.Linear(128, 128)   # 隐藏层 1
        self.l3 = nn.Linear(128, 128)   # 隐藏层 2
        self.l4 = nn.Linear(128, 64)    # 输出层
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return max_action * torch.tanh(self.l4(a))  # 输出范围 [-1, 1]
```

### 3.2 Critic（价值网络）

Twin Q-networks，输入 state+action(69+64=133D)，输出 Q 值。

```python
class Critic(nn.Module):
    def __init__(self, state_dim=69, action_dim=64):
        # Q1
        self.l1 = nn.Linear(133, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2（结构相同，参数独立）
        self.l4 = nn.Linear(133, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
```

### 3.3 TD3 超参数

```yaml
discount: 0.95        # γ，未来奖励折扣
tau: 0.005            # 软更新 target network 的速率
policy_noise: 0.2     # target policy 的噪声
noise_clip: 0.5       # 噪声裁剪范围
policy_freq: 2        # 每 2 次 critic 更新才更新 1 次 actor
batch_size: 128       # 从 replay buffer 采样的 batch 大小
learning_rate: 3e-4   # Adam 优化器学习率
```

---

## 4. Item 选择机制（怎么从 virtual item 变成真实推荐）

### 4.1 Enhanced Item Embedding

每个 item 的 embedding 不是直接用 LightGCN 的原始 embedding，而是融合了 cluster 信息：

```python
# 对每个 item：
item_emb_norm = item_embedding / norm(item_embedding)        # L2 归一化
cluster_emb_norm = cluster_embedding / norm(cluster_embedding)  # L2 归一化
combined = item_emb_norm + cluster_emb_norm                    # 相加
enhanced_emb = combined / norm(combined)                       # 再次归一化
```

这样做的好处：同一个 cluster 的 item 在 embedding 空间中更接近，RL agent 更容易学到"推荐某个 cluster"的策略。

### 4.2 RL 推荐：Cosine Similarity

TD3 Actor 输出一个 64D 的 virtual item，然后找与它最相似的真实 item：

```python
def get_rl_recommendations(virtual_item, all_enhanced_embeddings):
    # 归一化 virtual item
    virtual_norm = virtual_item / norm(virtual_item)
    
    # 计算与所有 item 的 cosine similarity（因为 enhanced_emb 已经归一化，直接点积）
    similarities = dot(all_enhanced_embeddings, virtual_norm)
    
    # 取 top-40 最相似的 item
    top_40_indices = argsort(similarities)[-40:]
    return [item_ids[i] for i in top_40_indices]
```

### 4.3 最终推荐列表：RL + 预训练推荐交叉

最终推荐列表是 RL 推荐和 LightGCN 预训练推荐的**交叉合并**：

```python
def get_final_recommendations(rl_recs, pretrained_recs):
    final = []
    for i in range(max(len(rl_recs), len(pretrained_recs))):
        if i < len(rl_recs):
            final.append(rl_recs[i])        # RL 推荐的第 i 个
        if i < len(pretrained_recs):
            final.append(pretrained_recs[i]) # 预训练推荐的第 i 个
    return final
```

配置：`rl_list_size = 40`，所以最终列表大约 80 个 item（40 RL + 40 预训练）。

### 4.4 Forced Cluster Exploration（可选）

如果某个 cluster 的 belief 太低（冷启动问题），可以强制替换一部分推荐为该 cluster 的 item：

```yaml
forced_cluster3_ratio: 0.0  # 0.0 = 关闭，0.2 = 20% 强制替换
```

---

## 5. 用户交互模拟（Acceptance Probability）

用户不会接受所有推荐。每个 item 的接受概率：

```python
def acceptance_probability(user_id, item_id):
    user_beliefs = users[user_id]['beliefs']       # 5D
    item_cluster = item_to_cluster[item_id]        # 0-4
    user_belief = user_beliefs[item_cluster]        # 用户对该 cluster 的偏好
    distance = cluster_distances[item_token]        # item 到 cluster 中心的距离（归一化到 [0,1]）
    
    # 距离因子：离 cluster 中心越近，越容易被接受
    distance_factor = max(0, 1 - distance)
    
    # 最终概率 = 距离因子 × 用户偏好 + 基线
    acceptance_prob = distance_factor * user_belief + baseline_rate  # baseline = 0.2
    
    return clamp(acceptance_prob, 0, 1)
```

然后用 `random() < acceptance_prob` 决定是否接受。

**关键参数**：
- `baseline_acceptance_rate: 0.2` — 即使用户完全不喜欢，也有 20% 概率接受（保证探索）
- `max_cluster_distance: 4` — 距离归一化的最大值

---

## 6. Belief 更新机制

### 6.1 Step-level 更新（每个 batch step 后立即更新）

```python
def update_beliefs_step(user_id, accepted_items, recommended_items):
    user_counts = user_accepted_counts[user_id]
    
    # 统计每个 cluster 接受了多少 item
    for item in accepted_items:
        cluster = item_to_cluster[item]
        user_counts['cluster_counts'][cluster] += 1
    
    # 新 belief = 各 cluster 接受数 / 总接受数
    total_accepted = sum(user_counts['cluster_counts'])
    if total_accepted > 0:
        new_beliefs = user_counts['cluster_counts'] / total_accepted
    else:
        new_beliefs = zeros(5)
    
    return new_beliefs
```

### 6.2 Belief 约束

Belief 不能超过 natural target（防止过度偏向某个 cluster）：

```python
def apply_belief_constraints(beliefs):
    # 超出 target 的部分削减 50%
    excess = max(0, beliefs - natural_target)
    constrained = beliefs - 0.5 * excess
    
    # 确保非负
    constrained = max(0, constrained)
    
    # 如果总和 > 1，归一化
    if sum(constrained) > 1:
        constrained = constrained / sum(constrained)
    
    return constrained
```

---

## 7. Reward Function（核心）

Reward 由三部分组成：

```
total_reward = improvement_reward + termination_bonus + entropy_reward
```

### 7.1 Improvement Reward（belief 改善奖励）

对每个 cluster，如果 belief 离 target 更近了，就给奖励：

```python
def improvement_reward(new_beliefs, previous_beliefs, natural_target, max_distances):
    reward = 0
    for i in range(5):
        current_dist  = abs(natural_target[i] - new_beliefs[i])
        previous_dist = abs(natural_target[i] - previous_beliefs[i])
        
        if current_dist < previous_dist:  # 有改善！
            # 绝对表现：离 target 越近，奖励越高
            abs_performance = (max_distances[i] - current_dist) / max_distances[i]
            abs_performance = max(0, abs_performance)
            
            # 改善幅度
            improvement_bonus = previous_dist - current_dist
            
            # 该 cluster 的奖励
            reward += abs_performance + improvement_bonus
    
    return reward
```

其中 `max_distances[i]` 是 episode 开始时每个用户每个 cluster 的初始距离，固定不变，用于归一化。

### 7.2 Termination Bonus（达标奖励）

如果用户的平均距离低于阈值，给一个大奖励：

```python
avg_distance = mean(abs(natural_target - new_beliefs))
if avg_distance < 0.10:  # termination_threshold
    termination_bonus = 5.0
else:
    termination_bonus = 0.0
```

### 7.3 Entropy Reward（多样性奖励）

鼓励推荐列表覆盖所有 5 个 cluster：

```python
def entropy_reward(recommended_items):
    # 统计推荐列表中每个 cluster 的 item 数量
    cluster_counts = count_per_cluster(recommended_items)  # [n0, n1, n2, n3, n4]
    total = sum(cluster_counts)
    
    # 计算信息熵
    probabilities = cluster_counts / total  # 归一化为概率
    entropy = -sum(p * log(p) for p in probabilities if p > 0)
    
    # 最大熵 = log(5) ≈ 1.609（5 个 cluster 均匀分布时）
    return entropy * entropy_weight  # entropy_weight = 10.0
```

**Reward 保证非负**：`total_reward = max(0, total_reward)`

---

## 8. 训练循环详细流程

### 8.1 整体结构

```
100 episodes × 50 rounds/episode = 5,000 rounds 总计
每 round 处理 50,000 users（分 batch，每 batch 1,000 users）
```

### 8.2 单个 Round 的流程

```python
def run_round(round_num):
    shuffle(user_list)  # 每轮打乱用户顺序
    
    for batch_users in batches(user_list, batch_size=1000):
        # 1. 获取 batch 状态 [1000, 69]
        states = get_batch_states(batch_users)
        
        # 2. TD3 Actor 推理 → virtual items [1000, 64]
        with no_grad():
            actions = td3.actor(states)
            # 训练期间加探索噪声
            noise = normal(0, 0.1, shape=actions.shape)
            actions = clamp(actions + noise, -1, 1)
            # L2 归一化
            actions = normalize(actions, p=2, dim=1)
        
        # 3. 为每个用户生成推荐、模拟交互、更新 belief、计算 reward
        for user_id, action in zip(batch_users, actions):
            rl_recs = find_top40_similar(action, enhanced_embeddings)
            pretrained_recs = pretrained_recommendations[user_id]
            final_recs = interleave(rl_recs, pretrained_recs)
            
            accepted, rejected = simulate_interaction(user_id, final_recs)
            new_beliefs, step_reward = update_beliefs_step(user_id, accepted, rejected, final_recs)
            
            next_state = get_state(user_id)  # 更新后的状态
            done = (mean_distance_to_target < 0.10)
            
            replay_buffer.add(state, action, next_state, step_reward, done)
        
        # 4. 训练 TD3（warmup 后）
        if replay_buffer.size >= 10000:  # warmup_transitions
            for _ in range(10):  # 10 次更新
                td3.train(replay_buffer, batch_size=128)
```

### 8.3 Episode 重置

每个 episode 开始时：
- 所有用户的 beliefs 重置为初始值
- 交互记录清空
- `max_distances_tensor` 重新计算（用于 reward 归一化）

### 8.4 Replay Buffer

标准的 experience replay：
- 存储 `(state, action, next_state, reward, done)` 五元组
- 支持 batch 添加和随机采样
- Warmup：前 10,000 个 transition 只收集不训练

---

## 9. 数据文件清单

所有路径相对于项目根目录 `TD3-master-entropy /`：

| 文件 | 路径 | 说明 |
|------|------|------|
| Item Embeddings | `embeddings/news/K_means/LightGCN/item_embedding.pkl` | 11,370 items × 64D |
| User Embeddings | `embeddings/news/K_means/LightGCN/user_embedding.pkl` | ~50,000 users × 64D |
| Recommendations | `embeddings/news/K_means/LightGCN/recommendations.pkl` | LightGCN 预训练推荐 |
| Item Token Map | `embeddings/news/K_means/item_token_map.json` | N12345 → 数字 token |
| User Token Map | `embeddings/news/K_means/user_token_map.json` | U100 → 数字 token |
| Cluster Manifest | `embeddings/news/K_means/cluster_matrix_manifest_K5.pkl` | cluster→items 映射 |
| Cluster Centers | `embeddings/news/K_means/cluster_centroids.json` | 5 个 cluster 中心 |
| Cluster Distances | `embeddings/news/K_means/node_to_center_distance.pt` | item 到中心的距离 |
| Test Set | `embeddings/news/K_means/test_set.csv` | 225,449 对 user-item |
| Cluster Embeddings | `processed_data/news/K_means/cluster_embeddings.pkl` | 5 clusters × 64D |
| Natural Target | `processed_data/news/K_means/natural_belief_target.pkl` | [0.175, 0.190, 0.210, 0.217, 0.209] |
| User Beliefs | `processed_data/news/K_means/user_average_beliefs.pkl` | 49,999 users × 5D |

---

## 10. 配置文件

运行命令：
```bash
cd "TD3-master-entropy "
python run_recommendation_rl.py --config config_news.yaml
```

关键配置差异（vs books）：
```yaml
# News 用 64D embedding（books 用 128D）
environment:
  state_dim: 69    # 5 + 64（books 是 133 = 5 + 128）
  action_dim: 64   # （books 是 128）
```

---

## 11. 评估指标

训练过程中跟踪的指标：

- **Acceptance Rate**：用户接受推荐的比例
- **PP1 Distance**：`sum(|belief[i] - target[i]|)` — 越小越好
- **Belief Improvements**：belief 有改善的用户数
- **Termination Bonuses**：达到 target 阈值的用户数
- **Step Reward**：每步的即时奖励
- **Round Reward**：每轮的累计奖励
- **Entropy**：推荐列表的 cluster 分布熵 — 越高越多样

---

## 12. 一句话总结

TD3 Actor 学习输出一个 64D 的"理想 item"向量 → 找到最相似的真实 item 推荐给用户 → 用户按概率接受/拒绝 → 根据 belief 是否更接近自然分布来给 reward → Critic 评估 Actor 的策略好不好 → Actor 逐渐学会推荐能让用户阅读更均衡的 item。

重点是 belief 需要和target接近，现在的问题是 belief不接近，而且 RL不收敛。