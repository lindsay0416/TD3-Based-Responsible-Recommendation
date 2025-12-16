#!/usr/bin/env python3
"""
Simple Average Rewards PDF Generator: Clean plot without statistics
Generates a single-page PDF with just the average reward per user trend.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Set style for better plots
plt.style.use('default')

class SimpleAvgRewardsPDFGenerator:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Storage for analysis results
        self.episode_reward_data = {}
        
    def analyze_episode_rewards(self):
        """Analyze average rewards per user for all episodes."""
        print("Analyzing average rewards per user for all episodes...")
        
        training_file = self.results_dir / "training_all_episodes.json"
        if not training_file.exists():
            print(f"Warning: {training_file} not found")
            return
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        for episode_data in training_data:
            episode = episode_data['episode']
            
            # Calculate total reward and total users for this episode
            total_reward = 0.0
            total_users = 0
            
            for round_data in episode_data['rounds']:
                round_reward = round_data['round_reward']
                users_per_round = 50000  # From config
                total_reward += round_reward
                total_users += users_per_round
            
            # Calculate average reward per user
            avg_reward_per_user = total_reward / total_users if total_users > 0 else 0.0
            self.episode_reward_data[episode] = avg_reward_per_user
        
        print(f"Analysis completed for {len(self.episode_reward_data)} episodes")
    
    def generate_simple_pdf(self):
        """Generate simple PDF with just the average reward per user plot."""
        if not self.episode_reward_data:
            print("No reward data to plot")
            return
        
        episodes = sorted(self.episode_reward_data.keys())
        avg_rewards = [self.episode_reward_data[ep] for ep in episodes]
        
        # Create PDF
        pdf_path = self.plots_dir / 'avg_rewards_per_user_simple.pdf'
        
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Main plot - clean line with markers
            ax.plot(episodes, avg_rewards, 'o-', color='#8E44AD', linewidth=3, 
                   markersize=6, alpha=0.9, markerfacecolor='white', 
                   markeredgecolor='#8E44AD', markeredgewidth=2)
            
            # Set title and labels
            ax.set_title('Average Reward per User per Episode Across All 50 Episodes', 
                        fontsize=20, fontweight='bold', pad=25)
            ax.set_xlabel('Episode', fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Reward per User', fontsize=16, fontweight='bold')
            
            # Customize grid
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Set axis limits
            ax.set_xlim(0, 51)
            
            # Customize tick parameters
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            ax.set_xticks(np.arange(0, 51, 5), minor=False)
            ax.set_xticks(np.arange(0, 51, 1), minor=True)
            
            # Customize spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
        
        print(f"Simple average rewards per user PDF saved to {pdf_path}")


def main():
    """Main function."""
    print("="*60)
    print("SIMPLE AVERAGE REWARDS PER USER PDF GENERATOR")
    print("="*60)
    
    generator = SimpleAvgRewardsPDFGenerator()
    generator.analyze_episode_rewards()
    generator.generate_simple_pdf()
    
    if generator.episode_reward_data:
        episodes = sorted(generator.episode_reward_data.keys())
        first_reward = generator.episode_reward_data[episodes[0]]
        last_reward = generator.episode_reward_data[episodes[-1]]
        
        print(f"\nGenerated clean plot for {len(episodes)} episodes")
        print(f"Episode 1 avg reward per user: {first_reward:.4f}")
        print(f"Episode 50 avg reward per user: {last_reward:.4f}")
        print(f"File: plots/avg_rewards_per_user_simple.pdf")
    
    print("="*60)


if __name__ == "__main__":
    main()