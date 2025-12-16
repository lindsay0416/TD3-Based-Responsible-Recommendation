#!/usr/bin/env python3
"""
Results Analysis: Recommended Items Cluster Distribution and Episode Rewards
Analyzes cluster distribution entropy and reward progression across episodes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ResultsAnalyzer:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load cluster mapping
        self.load_cluster_mapping()
        
        # Storage for analysis results
        self.episode_entropy_data = {}
        self.episode_reward_data = {}
        self.cluster_distribution_data = {}
        
    def load_cluster_mapping(self):
        """Load item to cluster mapping."""
        print("Loading cluster mapping...")
        with open("embeddings/cluster_matrix_manifest_K5_topk5_balanced.json", "r") as f:
            cluster_data = json.load(f)
        
        # Create item to cluster mapping
        self.item_to_cluster = {}
        for cluster_id, items in cluster_data['cluster_to_items'].items():
            for item in items:
                self.item_to_cluster[item] = int(cluster_id)
        
        print(f"Loaded cluster mapping for {len(self.item_to_cluster)} items")
    
    def calculate_entropy(self, cluster_ids: List[int], num_clusters: int = 5) -> float:
        """Calculate normalized entropy for cluster distribution."""
        if not cluster_ids:
            return 0.0
        
        cluster_counts = Counter(cluster_ids)
        total = len(cluster_ids)
        
        # Calculate probabilities
        probabilities = np.array([cluster_counts.get(i, 0) / total for i in range(num_clusters)])
        
        # Calculate entropy: H(X) = -Σ p(x) log(p(x))
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy += -p * np.log(p)
        
        # Normalize by max entropy (log(5) for 5 clusters)
        max_entropy = np.log(num_clusters)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def analyze_recommendations_episode(self, episode: int) -> Dict:
        """Analyze recommendations for a single episode."""
        rec_file = self.results_dir / f"recommendations_ep{episode}.json"
        
        if not rec_file.exists():
            print(f"Warning: {rec_file} not found")
            return None
        
        print(f"Analyzing episode {episode} recommendations...")
        
        # Load recommendations data
        with open(rec_file, 'r') as f:
            data = json.load(f)
        
        episode_entropies = []
        episode_cluster_counts = {i: 0 for i in range(5)}
        total_recommendations = 0
        
        # Process each round in the episode
        for round_key, round_data in data['rounds'].items():
            round_entropies = []
            
            # Process each user's recommendations
            for user_rec in round_data['recommendations']:
                recommended_items = user_rec['recommended_items']
                
                # Map items to clusters
                cluster_ids = []
                for item_id in recommended_items:
                    # Handle different item ID formats
                    item_str = str(item_id)
                    if item_str in self.item_to_cluster:
                        cluster_id = self.item_to_cluster[item_str]
                        cluster_ids.append(cluster_id)
                        episode_cluster_counts[cluster_id] += 1
                        total_recommendations += 1
                
                # Calculate entropy for this user's recommendations
                if cluster_ids:
                    user_entropy = self.calculate_entropy(cluster_ids)
                    round_entropies.append(user_entropy)
            
            # Add round average entropy
            if round_entropies:
                episode_entropies.extend(round_entropies)
        
        # Calculate episode statistics
        avg_entropy = np.mean(episode_entropies) if episode_entropies else 0.0
        std_entropy = np.std(episode_entropies) if episode_entropies else 0.0
        
        # Calculate cluster distribution
        cluster_distribution = {}
        for cluster_id in range(5):
            cluster_distribution[cluster_id] = episode_cluster_counts[cluster_id] / total_recommendations if total_recommendations > 0 else 0.0
        
        return {
            'episode': episode,
            'avg_entropy': avg_entropy,
            'std_entropy': std_entropy,
            'total_users': len(episode_entropies),
            'total_recommendations': total_recommendations,
            'cluster_distribution': cluster_distribution,
            'all_entropies': episode_entropies
        }
    
    def analyze_episode_rewards(self) -> Dict:
        """Analyze episode rewards from training data."""
        print("Analyzing episode rewards...")
        
        training_file = self.results_dir / "training_all_episodes.json"
        if not training_file.exists():
            print(f"Warning: {training_file} not found")
            return {}
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        episode_rewards = {}
        
        for episode_data in training_data:
            episode = episode_data['episode']
            
            # Calculate episode totals
            total_reward = 0.0
            round_rewards = []
            
            for round_data in episode_data['rounds']:
                round_reward = round_data['round_reward']
                total_reward += round_reward
                round_rewards.append(round_reward)
            
            # Calculate average reward per user per round
            num_rounds = len(round_rewards)
            users_per_round = 50000  # From config
            avg_reward_per_user = total_reward / (num_rounds * users_per_round) if num_rounds > 0 else 0.0
            
            episode_rewards[episode] = {
                'total_reward': total_reward,
                'avg_reward_per_user': avg_reward_per_user,
                'round_rewards': round_rewards,
                'num_rounds': num_rounds
            }
        
        return episode_rewards
    
    def run_analysis(self, max_episodes: int = 20):
        """Run complete analysis for all episodes."""
        print(f"Starting analysis for episodes 1-{max_episodes}...")
        
        # Analyze recommendations for each episode
        for episode in range(1, max_episodes + 1):
            result = self.analyze_recommendations_episode(episode)
            if result:
                self.episode_entropy_data[episode] = result
                self.cluster_distribution_data[episode] = result['cluster_distribution']
        
        # Analyze episode rewards
        self.episode_reward_data = self.analyze_episode_rewards()
        
        print(f"Analysis completed for {len(self.episode_entropy_data)} episodes")
    
    def plot_entropy_analysis(self):
        """Plot entropy analysis results."""
        if not self.episode_entropy_data:
            print("No entropy data to plot")
            return
        
        episodes = sorted(self.episode_entropy_data.keys())
        avg_entropies = [self.episode_entropy_data[ep]['avg_entropy'] for ep in episodes]
        std_entropies = [self.episode_entropy_data[ep]['std_entropy'] for ep in episodes]
        
        # Plot 1: Entropy trend across episodes
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(episodes, avg_entropies, 'b-o', linewidth=2, markersize=6)
        plt.fill_between(episodes, 
                        [avg - std for avg, std in zip(avg_entropies, std_entropies)],
                        [avg + std for avg, std in zip(avg_entropies, std_entropies)],
                        alpha=0.3)
        plt.xlabel('Episode')
        plt.ylabel('Normalized Entropy')
        plt.title('Recommendation Entropy Across Episodes')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Entropy distribution boxplot
        plt.subplot(2, 2, 2)
        entropy_data = []
        episode_labels = []
        for ep in episodes[::5]:  # Every 5th episode to avoid crowding
            if ep in self.episode_entropy_data:
                entropy_data.append(self.episode_entropy_data[ep]['all_entropies'])
                episode_labels.append(f'Ep {ep}')
        
        if entropy_data:
            plt.boxplot(entropy_data, labels=episode_labels)
            plt.ylabel('Normalized Entropy')
            plt.title('Entropy Distribution by Episode')
            plt.xticks(rotation=45)
        
        # Plot 3: Cluster distribution heatmap
        plt.subplot(2, 2, 3)
        cluster_matrix = []
        for ep in episodes:
            if ep in self.cluster_distribution_data:
                row = [self.cluster_distribution_data[ep][i] for i in range(5)]
                cluster_matrix.append(row)
        
        if cluster_matrix:
            cluster_matrix = np.array(cluster_matrix)
            sns.heatmap(cluster_matrix.T, 
                       xticklabels=[f'Ep{ep}' for ep in episodes],
                       yticklabels=[f'Cluster {i}' for i in range(5)],
                       annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title('Cluster Distribution Across Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Cluster')
        
        # Plot 4: Summary statistics
        plt.subplot(2, 2, 4)
        total_recs = [self.episode_entropy_data[ep]['total_recommendations'] for ep in episodes]
        plt.plot(episodes, total_recs, 'g-s', linewidth=2, markersize=6)
        plt.xlabel('Episode')
        plt.ylabel('Total Recommendations')
        plt.title('Total Recommendations per Episode')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'recommendation_entropy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Entropy analysis plot saved to {self.plots_dir / 'recommendation_entropy_analysis.png'}")
    
    def plot_reward_analysis(self):
        """Plot reward analysis results."""
        if not self.episode_reward_data:
            print("No reward data to plot")
            return
        
        episodes = sorted(self.episode_reward_data.keys())
        total_rewards = [self.episode_reward_data[ep]['total_reward'] for ep in episodes]
        avg_rewards_per_user = [self.episode_reward_data[ep]['avg_reward_per_user'] for ep in episodes]
        
        # Plot reward trends
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Total episode rewards
        plt.subplot(2, 3, 1)
        plt.plot(episodes, total_rewards, 'r-o', linewidth=2, markersize=6)
        plt.xlabel('Episode')
        plt.ylabel('Total Episode Reward')
        plt.title('Total Reward per Episode')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average reward per user
        plt.subplot(2, 3, 2)
        plt.plot(episodes, avg_rewards_per_user, 'b-s', linewidth=2, markersize=6)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward per User')
        plt.title('Average Reward per User per Episode')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Round-by-round rewards for first few episodes
        plt.subplot(2, 3, 3)
        for ep in episodes[:5]:  # First 5 episodes
            round_rewards = self.episode_reward_data[ep]['round_rewards']
            rounds = list(range(1, len(round_rewards) + 1))
            plt.plot(rounds, round_rewards, '-o', label=f'Episode {ep}', alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Round Reward')
        plt.title('Round Rewards (Episodes 1-5)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Reward improvement over episodes
        plt.subplot(2, 3, 4)
        if len(total_rewards) > 1:
            reward_changes = [total_rewards[i] - total_rewards[i-1] for i in range(1, len(total_rewards))]
            plt.plot(episodes[1:], reward_changes, 'g-^', linewidth=2, markersize=6)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Episode')
            plt.ylabel('Reward Change from Previous Episode')
            plt.title('Episode-to-Episode Reward Change')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Reward statistics
        plt.subplot(2, 3, 5)
        reward_stats = {
            'Mean': np.mean(total_rewards),
            'Std': np.std(total_rewards),
            'Min': np.min(total_rewards),
            'Max': np.max(total_rewards)
        }
        bars = plt.bar(reward_stats.keys(), reward_stats.values(), color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylabel('Reward Value')
        plt.title('Reward Statistics Across Episodes')
        for bar, value in zip(bars, reward_stats.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(reward_stats.values())*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # Plot 6: Cumulative rewards
        plt.subplot(2, 3, 6)
        cumulative_rewards = np.cumsum(total_rewards)
        plt.plot(episodes, cumulative_rewards, 'm-o', linewidth=2, markersize=6)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Across Episodes')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'episode_reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Reward analysis plot saved to {self.plots_dir / 'episode_reward_analysis.png'}")
    
    def save_analysis_results(self):
        """Save analysis results to files."""
        # Save entropy results
        entropy_results = {
            'episodes': {},
            'summary': {
                'total_episodes': len(self.episode_entropy_data),
                'avg_entropy_across_episodes': np.mean([data['avg_entropy'] for data in self.episode_entropy_data.values()]),
                'entropy_trend': 'increasing' if len(self.episode_entropy_data) > 1 and 
                               list(self.episode_entropy_data.values())[-1]['avg_entropy'] > 
                               list(self.episode_entropy_data.values())[0]['avg_entropy'] else 'decreasing'
            }
        }
        
        for ep, data in self.episode_entropy_data.items():
            entropy_results['episodes'][ep] = {
                'avg_entropy': data['avg_entropy'],
                'std_entropy': data['std_entropy'],
                'total_users': data['total_users'],
                'total_recommendations': data['total_recommendations'],
                'cluster_distribution': data['cluster_distribution']
            }
        
        with open(self.plots_dir / 'entropy_analysis_results.json', 'w') as f:
            json.dump(entropy_results, f, indent=2)
        
        # Save reward results
        reward_results = {
            'episodes': self.episode_reward_data,
            'summary': {
                'total_episodes': len(self.episode_reward_data),
                'total_cumulative_reward': sum([data['total_reward'] for data in self.episode_reward_data.values()]),
                'avg_episode_reward': np.mean([data['total_reward'] for data in self.episode_reward_data.values()]),
                'reward_trend': 'increasing' if len(self.episode_reward_data) > 1 and 
                              list(self.episode_reward_data.values())[-1]['total_reward'] > 
                              list(self.episode_reward_data.values())[0]['total_reward'] else 'decreasing'
            }
        }
        
        with open(self.plots_dir / 'reward_analysis_results.json', 'w') as f:
            json.dump(reward_results, f, indent=2)
        
        print(f"Analysis results saved to {self.plots_dir}")


def main():
    """Main analysis function."""
    print("="*80)
    print("RESULTS ANALYSIS: CLUSTER DISTRIBUTION & EPISODE REWARDS")
    print("="*80)
    
    analyzer = ResultsAnalyzer()
    
    # Run analysis
    analyzer.run_analysis(max_episodes=20)
    
    # Generate plots
    analyzer.plot_entropy_analysis()
    analyzer.plot_reward_analysis()
    
    # Save results
    analyzer.save_analysis_results()
    
    # Print summary
    if analyzer.episode_entropy_data:
        episodes = sorted(analyzer.episode_entropy_data.keys())
        first_entropy = analyzer.episode_entropy_data[episodes[0]]['avg_entropy']
        last_entropy = analyzer.episode_entropy_data[episodes[-1]]['avg_entropy']
        
        print(f"\nENTROPY ANALYSIS SUMMARY:")
        print(f"  Episodes analyzed: {len(episodes)}")
        print(f"  Entropy Episode 1: {first_entropy:.4f}")
        print(f"  Entropy Episode {episodes[-1]}: {last_entropy:.4f}")
        print(f"  Entropy change: {((last_entropy - first_entropy) / first_entropy * 100):+.2f}%")
    
    if analyzer.episode_reward_data:
        episodes = sorted(analyzer.episode_reward_data.keys())
        first_reward = analyzer.episode_reward_data[episodes[0]]['total_reward']
        last_reward = analyzer.episode_reward_data[episodes[-1]]['total_reward']
        
        print(f"\nREWARD ANALYSIS SUMMARY:")
        print(f"  Episodes analyzed: {len(episodes)}")
        print(f"  Total reward Episode 1: {first_reward:,.0f}")
        print(f"  Total reward Episode {episodes[-1]}: {last_reward:,.0f}")
        print(f"  Reward change: {((last_reward - first_reward) / first_reward * 100):+.2f}%")
    
    print(f"\nAll plots and results saved to: {Path('plots').absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()