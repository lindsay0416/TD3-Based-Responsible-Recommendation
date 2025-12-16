#!/usr/bin/env python3
"""
Clean Round Rewards PDF Generator: Round Rewards for Key Episodes with Optimized Layout
Generates a clean PDF with better positioning of statistics descriptions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class CleanRewardsPDFGenerator:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Key episodes to analyze
        self.key_episodes = [1, 10, 20, 30, 40, 50]
        
        # Storage for analysis results
        self.episode_reward_data = {}
        
    def analyze_episode_rewards(self) -> Dict:
        """Analyze episode rewards from training data for all 50 episodes."""
        print("Analyzing episode rewards for key episodes...")
        
        training_file = self.results_dir / "training_all_episodes.json"
        if not training_file.exists():
            print(f"Warning: {training_file} not found")
            return {}
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        episode_rewards = {}
        
        for episode_data in training_data:
            episode = episode_data['episode']
            
            # Only process key episodes
            if episode not in self.key_episodes:
                continue
            
            # Calculate round-by-round rewards per user
            round_rewards_per_user = []
            total_reward = 0.0
            
            for round_data in episode_data['rounds']:
                round_reward = round_data['round_reward']
                users_in_round = 50000  # From config
                reward_per_user = round_reward / users_in_round
                round_rewards_per_user.append(reward_per_user)
                total_reward += round_reward
            
            episode_rewards[episode] = {
                'round_rewards_per_user': round_rewards_per_user,
                'total_reward': total_reward,
                'avg_reward_per_user_per_episode': total_reward / (len(round_rewards_per_user) * 50000) if round_rewards_per_user else 0.0,
                'num_rounds': len(round_rewards_per_user)
            }
        
        return episode_rewards
    
    def run_analysis(self):
        """Run analysis for key episodes."""
        print(f"Starting analysis for key episodes: {self.key_episodes}")
        
        # Analyze episode rewards
        self.episode_reward_data = self.analyze_episode_rewards()
        
        print(f"Analysis completed for {len(self.episode_reward_data)} episodes")
    
    def generate_clean_rewards_pdf(self):
        """Generate clean PDF with round rewards for key episodes."""
        if not self.episode_reward_data:
            print("No reward data to plot")
            return
        
        # Create PDF
        pdf_path = self.plots_dir / 'round_rewards_key_episodes_clean.pdf'
        
        with PdfPages(pdf_path) as pdf:
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            axes = axes.flatten()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, episode in enumerate(self.key_episodes):
                if episode in self.episode_reward_data:
                    round_rewards = self.episode_reward_data[episode]['round_rewards_per_user']
                    rounds = list(range(1, len(round_rewards) + 1))
                    
                    # Plot the data
                    axes[i].plot(rounds, round_rewards, 'o-', color=colors[i], 
                               linewidth=2.5, markersize=6, markerfacecolor='white',
                               markeredgecolor=colors[i], markeredgewidth=2, alpha=0.9)
                    
                    # Set title and labels
                    axes[i].set_title(f'Episode {episode} - Round Rewards per User', 
                                    fontsize=16, fontweight='bold', pad=15)
                    axes[i].set_xlabel('Round', fontsize=13, fontweight='bold')
                    axes[i].set_ylabel('Average Reward per User', fontsize=13, fontweight='bold')
                    
                    # Customize grid
                    axes[i].grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                    axes[i].set_axisbelow(True)
                    
                    # Calculate statistics
                    avg_reward = np.mean(round_rewards)
                    std_reward = np.std(round_rewards)
                    min_reward = np.min(round_rewards)
                    max_reward = np.max(round_rewards)
                    min_round = rounds[np.argmin(round_rewards)]
                    max_round = rounds[np.argmax(round_rewards)]
                    
                    # Position statistics in the bottom right corner
                    stats_x, stats_y = 0.98, 0.02
                    ha, va = 'right', 'bottom'
                    
                    stats_text = f"Statistics\n" + "─"*12 + "\n"
                    stats_text += f"Avg: {avg_reward:.3f}\n"
                    stats_text += f"Std: {std_reward:.3f}\n"
                    stats_text += f"Min: {min_reward:.3f} (R{min_round})\n"
                    stats_text += f"Max: {max_reward:.3f} (R{max_round})\n"
                    stats_text += f"Range: {max_reward - min_reward:.3f}"
                    
                    axes[i].text(stats_x, stats_y, stats_text, 
                               transform=axes[i].transAxes, 
                               verticalalignment=va, horizontalalignment=ha,
                               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                        edgecolor=colors[i], alpha=0.95, linewidth=1.5),
                               fontsize=11, fontfamily='monospace')
                    
                    # Highlight min and max points
                    axes[i].scatter([min_round], [min_reward], color='red', s=80, 
                                  zorder=5, edgecolor='white', linewidth=1.5, alpha=0.8)
                    axes[i].scatter([max_round], [max_reward], color='green', s=80, 
                                  zorder=5, edgecolor='white', linewidth=1.5, alpha=0.8)
                    
                    # Customize tick parameters
                    axes[i].tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
                    
                    # Customize spines
                    for spine in axes[i].spines.values():
                        spine.set_linewidth(1.2)
                        spine.set_color('#333333')
                    
                    # Set axis limits with padding
                    reward_range = max_reward - min_reward
                    y_padding = reward_range * 0.1 if reward_range > 0 else 0.1
                    axes[i].set_ylim(min_reward - y_padding, max_reward + y_padding)
                    axes[i].set_xlim(0.5, len(rounds) + 0.5)
                    
                else:
                    # Handle missing data
                    axes[i].text(0.5, 0.5, f'Episode {episode}\nData not available', 
                               transform=axes[i].transAxes, ha='center', va='center',
                               fontsize=14, style='italic', color='gray')
                    axes[i].set_title(f'Episode {episode} - No Data', fontsize=16, color='gray')
                    axes[i].set_xlabel('Round', fontsize=13)
                    axes[i].set_ylabel('Average Reward per User', fontsize=13)
                    axes[i].grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle('Round Rewards per User Across Key Episodes', 
                        fontsize=22, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for suptitle
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
        
        print(f"Clean rewards PDF saved to {pdf_path}")
        
        # Also save as high-quality PNG
        png_path = self.plots_dir / 'round_rewards_key_episodes_clean.png'
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        for i, episode in enumerate(self.key_episodes):
            if episode in self.episode_reward_data:
                round_rewards = self.episode_reward_data[episode]['round_rewards_per_user']
                rounds = list(range(1, len(round_rewards) + 1))
                
                axes[i].plot(rounds, round_rewards, 'o-', color=colors[i], 
                           linewidth=2.5, markersize=6, markerfacecolor='white',
                           markeredgecolor=colors[i], markeredgewidth=2, alpha=0.9)
                
                axes[i].set_title(f'Episode {episode} - Round Rewards per User', 
                                fontsize=16, fontweight='bold', pad=15)
                axes[i].set_xlabel('Round', fontsize=13, fontweight='bold')
                axes[i].set_ylabel('Average Reward per User', fontsize=13, fontweight='bold')
                
                axes[i].grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                axes[i].set_axisbelow(True)
                
                # Statistics positioning in bottom right corner
                avg_reward = np.mean(round_rewards)
                std_reward = np.std(round_rewards)
                min_reward = np.min(round_rewards)
                max_reward = np.max(round_rewards)
                min_round = rounds[np.argmin(round_rewards)]
                max_round = rounds[np.argmax(round_rewards)]
                
                # Always position in bottom right corner
                stats_x, stats_y = 0.98, 0.02
                ha, va = 'right', 'bottom'
                
                stats_text = f"Statistics\n" + "─"*12 + "\n"
                stats_text += f"Avg: {avg_reward:.3f}\n"
                stats_text += f"Std: {std_reward:.3f}\n"
                stats_text += f"Min: {min_reward:.3f} (R{min_round})\n"
                stats_text += f"Max: {max_reward:.3f} (R{max_round})\n"
                stats_text += f"Range: {max_reward - min_reward:.3f}"
                
                axes[i].text(stats_x, stats_y, stats_text, 
                           transform=axes[i].transAxes, 
                           verticalalignment=va, horizontalalignment=ha,
                           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                    edgecolor=colors[i], alpha=0.95, linewidth=1.5),
                           fontsize=11, fontfamily='monospace')
                
                axes[i].scatter([min_round], [min_reward], color='red', s=80, 
                              zorder=5, edgecolor='white', linewidth=1.5, alpha=0.8)
                axes[i].scatter([max_round], [max_reward], color='green', s=80, 
                              zorder=5, edgecolor='white', linewidth=1.5, alpha=0.8)
                
                axes[i].tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
                
                for spine in axes[i].spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_color('#333333')
                
                reward_range = max_reward - min_reward
                y_padding = reward_range * 0.1 if reward_range > 0 else 0.1
                axes[i].set_ylim(min_reward - y_padding, max_reward + y_padding)
                axes[i].set_xlim(0.5, len(rounds) + 0.5)
                
            else:
                axes[i].text(0.5, 0.5, f'Episode {episode}\nData not available', 
                           transform=axes[i].transAxes, ha='center', va='center',
                           fontsize=14, style='italic', color='gray')
                axes[i].set_title(f'Episode {episode} - No Data', fontsize=16, color='gray')
                axes[i].set_xlabel('Round', fontsize=13)
                axes[i].set_ylabel('Average Reward per User', fontsize=13)
                axes[i].grid(True, alpha=0.3)
        
        fig.suptitle('Round Rewards per User Across Key Episodes', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Clean rewards PNG saved to {png_path}")


def main():
    """Main function to generate clean rewards PDF."""
    print("="*80)
    print("CLEAN ROUND REWARDS PDF GENERATOR: KEY EPISODES")
    print("="*80)
    
    generator = CleanRewardsPDFGenerator()
    
    # Run analysis
    generator.run_analysis()
    
    # Generate clean PDF
    generator.generate_clean_rewards_pdf()
    
    # Print summary
    if generator.episode_reward_data:
        print(f"\nANALYSIS SUMMARY:")
        print(f"="*50)
        print(f"Episodes analyzed: {list(generator.episode_reward_data.keys())}")
        
        for ep in generator.key_episodes:
            if ep in generator.episode_reward_data:
                avg_reward = generator.episode_reward_data[ep]['avg_reward_per_user_per_episode']
                num_rounds = generator.episode_reward_data[ep]['num_rounds']
                print(f"Episode {ep}: {avg_reward:.3f} avg reward/user ({num_rounds} rounds)")
    
    print(f"\nGenerated Files:")
    print(f"  📄 plots/round_rewards_key_episodes_clean.pdf")
    print(f"  🖼️  plots/round_rewards_key_episodes_clean.png")
    print("="*80)


if __name__ == "__main__":
    main()