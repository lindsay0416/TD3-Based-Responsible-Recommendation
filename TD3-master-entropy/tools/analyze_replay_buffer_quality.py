#!/usr/bin/env python3
"""
Analyze quality of experiences in Replay Buffer
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

class ReplayBufferAnalyzer:
    """Analyze quality of experiences in replay buffer"""
    
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer
        
    def analyze_quality(self):
        """Comprehensive analysis of experience quality in buffer"""
        print("="*80)
        print("REPLAY BUFFER QUALITY ANALYSIS")
        print("="*80)
        
        # 1. Basic statistics
        self.analyze_basic_stats()
        
        # 2. Reward distribution
        self.analyze_reward_distribution()
        
        # 3. State diversity
        self.analyze_state_diversity()
        
        # 4. Learning progress
        self.analyze_learning_progress()
        
        # 5. Anomaly detection
        self.detect_anomalies()
        
        # 6. Quality scoring
        self.calculate_quality_score()
    
    def analyze_basic_stats(self):
        """Basic statistics"""
        print("\n📊 1. Basic Statistics")
        print("-"*80)
        
        size = self.buffer.size
        max_size = self.buffer.max_size
        
        print(f"Buffer使用率:     {size:,} / {max_size:,} ({size/max_size*100:.1f}%)")
        
        if size == 0:
            print("⚠️  Buffer为空，无法分析")
            return
        
        # Analyze rewards
        rewards = self.buffer.reward[:size].flatten()
        
        print(f"\nReward统计:")
        print(f"  平均值:         {rewards.mean():.4f}")
        print(f"  中位数:         {np.median(rewards):.4f}")
        print(f"  标准差:         {rewards.std():.4f}")
        print(f"  最小值:         {rewards.min():.4f}")
        print(f"  Maximum:        {rewards.max():.4f}")
        
        # Analyze done flags
        dones = self.buffer.not_done[:size].flatten()
        done_count = np.sum(dones == 0)  # not_done=0 means done=True
        
        print(f"\n完成状态:")
        print(f"  达到目标的经验: {done_count:,} ({done_count/size*100:.2f}%)")
        print(f"  未完成的经验:   {size-done_count:,} ({(size-done_count)/size*100:.2f}%)")
    
    def analyze_reward_distribution(self):
        """分析reward分布"""
        print("\n📈 2. Reward分布分析")
        print("-"*80)
        
        size = self.buffer.size
        if size == 0:
            return
        
        rewards = self.buffer.reward[:size].flatten()
        
        # Categorize rewards
        negative = np.sum(rewards < 0)
        zero = np.sum(rewards == 0)
        small_positive = np.sum((rewards > 0) & (rewards < 1))
        medium_positive = np.sum((rewards >= 1) & (rewards < 5))
        large_positive = np.sum(rewards >= 5)
        
        print(f"Reward Distribution:")
        print(f"  Negative rewards (< 0):     {negative:,} ({negative/size*100:.2f}%)")
        print(f"  Zero rewards (= 0):         {zero:,} ({zero/size*100:.2f}%)")
        print(f"  Small positive (0-1):       {small_positive:,} ({small_positive/size*100:.2f}%)")
        print(f"  Medium positive (1-5):      {medium_positive:,} ({medium_positive/size*100:.2f}%)")
        print(f"  Large positive (≥ 5):       {large_positive:,} ({large_positive/size*100:.2f}%)")
        
        # Quality assessment
        print(f"\nQuality Assessment:")
        if zero / size > 0.5:
            print(f"  ⚠️  Too many zero rewards ({zero/size*100:.1f}%) - may lack learning signal")
        elif small_positive / size > 0.3:
            print(f"  ✅ Sufficient positive rewards ({(small_positive+medium_positive+large_positive)/size*100:.1f}%)")
        
        if large_positive > 0:
            print(f"  ✅ {large_positive:,} high-quality experiences (reward ≥ 5)")
        else:
            print(f"  ⚠️  No high-quality experiences (reward ≥ 5)")
        
        # Percentiles
        print(f"\nReward Percentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(rewards, p)
            print(f"  {p}%: {val:.4f}")
    
    def analyze_state_diversity(self):
        """Analyze state diversity"""
        print("\n🎨 3. State Diversity Analysis")
        print("-"*80)
        
        size = self.buffer.size
        if size == 0:
            return
        
        states = self.buffer.state[:size]
        
        # Analyze belief part (first 5 dimensions)
        beliefs = states[:, :5]
        
        print(f"Belief统计 (前5维):")
        for i in range(5):
            cluster_beliefs = beliefs[:, i]
            print(f"  Cluster {i}:")
            print(f"    平均值: {cluster_beliefs.mean():.6f}")
            print(f"    标准差: {cluster_beliefs.std():.6f}")
            print(f"    范围:   [{cluster_beliefs.min():.6f}, {cluster_beliefs.max():.6f}]")
        
        # Check diversity
        print(f"\nDiversity Assessment:")
        
        # Calculate overall variance of beliefs
        total_variance = beliefs.var(axis=0).sum()
        print(f"  Total variance: {total_variance:.6f}")
        
        if total_variance < 0.001:
            print(f"  ⚠️  方差太小 - states lack diversity")
        elif total_variance < 0.01:
            print(f"  ⚠️  方差较小 - state多样性不足")
        else:
            print(f"  ✅ Reasonable variance - states have good diversity")
        
        # Check for stuck states (unchanged beliefs)
        unique_states = len(np.unique(beliefs, axis=0))
        print(f"\n  Unique state count: {unique_states:,} / {size:,} ({unique_states/size*100:.1f}%)")
        
        if unique_states / size < 0.1:
            print(f"  ⚠️  唯一state占比过低 - 可能存在stuck states")
        else:
            print(f"  ✅ 唯一state占比合理")
    
    def analyze_learning_progress(self):
        """Analyze learning progress"""
        print("\n📚 4. 学习进度分析")
        print("-"*80)
        
        size = self.buffer.size
        if size == 0:
            return
        
        states = self.buffer.state[:size]
        next_states = self.buffer.next_state[:size]
        rewards = self.buffer.reward[:size].flatten()
        
        # Calculate belief changes
        beliefs = states[:, :5]
        next_beliefs = next_states[:, :5]
        belief_changes = next_beliefs - beliefs
        
        print(f"Belief变化统计:")
        print(f"  平均变化量: {np.abs(belief_changes).mean():.6f}")
        print(f"  Maximum change: {np.abs(belief_changes).max():.6f}")
        
        # Count effective learning experiences
        significant_changes = np.sum(np.abs(belief_changes).sum(axis=1) > 0.001)
        print(f"\nEffective learning experiences:")
        print(f"  有显著变化: {significant_changes:,} ({significant_changes/size*100:.2f}%)")
        print(f"  无显著变化: {size-significant_changes:,} ({(size-significant_changes)/size*100:.2f}%)")
        
        if significant_changes / size < 0.3:
            print(f"  ⚠️  有效经验占比过低 - 学习信号不足")
        else:
            print(f"  ✅ Reasonable proportion of effective experiences")
        
        # Analyze positive progress
        # Assume target is [0.7, 0.07, 0.07, 0.07, 0.08]
        target = np.array([0.7, 0.07, 0.07, 0.07, 0.08])
        
        current_distances = np.abs(beliefs - target).sum(axis=1)
        next_distances = np.abs(next_beliefs - target).sum(axis=1)
        
        improvements = current_distances > next_distances
        improvement_count = np.sum(improvements)
        
        print(f"\n朝向目标的进步:")
        print(f"  改进的经验: {improvement_count:,} ({improvement_count/size*100:.2f}%)")
        print(f"  退步的经验: {size-improvement_count:,} ({(size-improvement_count)/size*100:.2f}%)")
        
        if improvement_count / size < 0.3:
            print(f"  ⚠️  改进经验占比过低 - 策略可能不够好")
        elif improvement_count / size > 0.6:
            print(f"  ✅ 改进经验占比很好 - 策略正在学习")
        else:
            print(f"  ✅ 改进经验占比合理")
    
    def detect_anomalies(self):
        """检测异常经验"""
        print("\n🔍 5. 异常检测")
        print("-"*80)
        
        size = self.buffer.size
        if size == 0:
            return
        
        anomalies = []
        
        # Detect anomalous rewards
        rewards = self.buffer.reward[:size].flatten()
        
        # Use IQR method to detect outliers
        q1, q3 = np.percentile(rewards, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        reward_outliers = np.sum((rewards < lower_bound) | (rewards > upper_bound))
        
        if reward_outliers > 0:
            anomalies.append(f"发现{reward_outliers:,}个异常reward值")
        
        # Detect NaN or Inf
        states = self.buffer.state[:size]
        next_states = self.buffer.next_state[:size]
        
        nan_states = np.sum(np.isnan(states))
        nan_next_states = np.sum(np.isnan(next_states))
        nan_rewards = np.sum(np.isnan(rewards))
        
        if nan_states > 0:
            anomalies.append(f"发现{nan_states:,}个NaN state值")
        if nan_next_states > 0:
            anomalies.append(f"发现{nan_next_states:,}个NaN next_state值")
        if nan_rewards > 0:
            anomalies.append(f"发现{nan_rewards:,}个NaN reward值")
        
        # Detect beliefs out of range
        beliefs = states[:, :5]
        out_of_range = np.sum((beliefs < 0) | (beliefs > 1))
        
        if out_of_range > 0:
            anomalies.append(f"发现{out_of_range:,}个超出[0,1]范围的belief值")
        
        if anomalies:
            print("⚠️  发现异常:")
            for anomaly in anomalies:
                print(f"  - {anomaly}")
        else:
            print("✅ 未发现异常")
    
    def calculate_quality_score(self):
        """计算总体质量评分"""
        print("\n⭐ 6. 总体质量评分")
        print("-"*80)
        
        size = self.buffer.size
        if size == 0:
            print("无法计算评分 - buffer为空")
            return
        
        score = 0
        max_score = 100
        
        # Scoring criteria
        rewards = self.buffer.reward[:size].flatten()
        states = self.buffer.state[:size]
        next_states = self.buffer.next_state[:size]
        
        # 1. Reward quality (30分)
        positive_ratio = np.sum(rewards > 0) / size
        score += min(30, positive_ratio * 30)
        
        # 2. State diversity (20分)
        beliefs = states[:, :5]
        unique_ratio = len(np.unique(beliefs, axis=0)) / size
        score += min(20, unique_ratio * 20)
        
        # 3. Learning effectiveness (30分)
        belief_changes = next_states[:, :5] - beliefs
        significant_changes = np.sum(np.abs(belief_changes).sum(axis=1) > 0.001) / size
        score += min(30, significant_changes * 30)
        
        # 4. Progress ratio (20分)
        target = np.array([0.7, 0.07, 0.07, 0.07, 0.08])
        current_distances = np.abs(beliefs - target).sum(axis=1)
        next_distances = np.abs(next_states[:, :5] - target).sum(axis=1)
        improvement_ratio = np.sum(current_distances > next_distances) / size
        score += min(20, improvement_ratio * 20)
        
        print(f"总体评分: {score:.1f} / {max_score}")
        print(f"\n评分细节:")
        print(f"  Reward质量:   {min(30, positive_ratio * 30):.1f} / 30")
        print(f"  State多样性:  {min(20, unique_ratio * 20):.1f} / 20")
        print(f"  学习效果:     {min(30, significant_changes * 30):.1f} / 30")
        print(f"  Progress ratio:     {min(20, improvement_ratio * 20):.1f} / 20")
        
        # Rating
        if score >= 80:
            grade = "A (Excellent)"
            comment = "✅ Buffer质量很好，包含高质量的学习经验"
        elif score >= 60:
            grade = "B (Good)"
            comment = "✅ Buffer质量不错，可以继续训练"
        elif score >= 40:
            grade = "C (Average)"
            comment = "⚠️  Buffer质量一般，建议检查训练策略"
        else:
            grade = "D (Poor)"
            comment = "❌ Buffer质量较差，需要改进训练过程"
        
        print(f"\n评级: {grade}")
        print(f"{comment}")
    
    def save_analysis_report(self, filename="buffer_quality_report.json"):
        """保存分析报告"""
        size = self.buffer.size
        if size == 0:
            return
        
        rewards = self.buffer.reward[:size].flatten()
        states = self.buffer.state[:size]
        next_states = self.buffer.next_state[:size]
        beliefs = states[:, :5]
        
        report = {
            "buffer_size": int(size),
            "buffer_capacity": int(self.buffer.max_size),
            "usage_ratio": float(size / self.buffer.max_size),
            "reward_stats": {
                "mean": float(rewards.mean()),
                "median": float(np.median(rewards)),
                "std": float(rewards.std()),
                "min": float(rewards.min()),
                "max": float(rewards.max()),
            },
            "reward_distribution": {
                "negative": int(np.sum(rewards < 0)),
                "zero": int(np.sum(rewards == 0)),
                "small_positive": int(np.sum((rewards > 0) & (rewards < 1))),
                "medium_positive": int(np.sum((rewards >= 1) & (rewards < 5))),
                "large_positive": int(np.sum(rewards >= 5)),
            },
            "belief_stats": {
                f"cluster_{i}": {
                    "mean": float(beliefs[:, i].mean()),
                    "std": float(beliefs[:, i].std()),
                    "min": float(beliefs[:, i].min()),
                    "max": float(beliefs[:, i].max()),
                }
                for i in range(5)
            },
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 分析报告已保存到: {filename}")


def main():
    """主函数 - 演示如何使用"""
    print("="*80)
    print("REPLAY BUFFER 质量分析工具")
    print("="*80)
    
    print("\n使用方法:")
    print("-"*80)
    print("""
在训练过程中，可以这样使用：

```python
from analyze_replay_buffer_quality import ReplayBufferAnalyzer

# 在 recommendation_trainer.py 中
def analyze_buffer_quality(self):
    analyzer = ReplayBufferAnalyzer(self.replay_buffer)
    analyzer.analyze_quality()
    analyzer.save_analysis_report()
```

或者在训练的特定时刻调用：

```python
# 在 run_episode() 中
if episode_num % 5 == 0:  # 每5个episode分析一次
    print(f"\\n分析Episode {episode_num}的Buffer质量:")
    analyzer = ReplayBufferAnalyzer(self.replay_buffer)
    analyzer.analyze_quality()
```
    """)
    
    print("\n建议的分析时机:")
    print("-"*80)
    print("1. 训练开始后第1个episode - 检查初始数据质量")
    print("2. 每5-10个episodes - 监控学习进度")
    print("3. 训练结束时 - 评估最终buffer质量")
    print("4. 发现训练问题时 - 诊断原因")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
