#!/usr/bin/env python3
"""
Simple Total Rewards PDF Generator: Clean plot without statistics
Generates a single-page PDF with just the total reward trend.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Set style for better plots
plt.style.use('default')

class SimpleTotalRewardsPDFGenerator:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Storage for analysis results
        self.episode_reward_data = {}
        
    def analyze_episode_rewards(self):
        """Analyze total rewards for all episodes."""
        print("Analyzing total rewards for all episodes...")
        
        training_file = self.results_dir / "training_all_episodes.json"
        if not training_file.exists():
            print(f"Warning: {training_file} not found")
            return
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        for episode_data in training_data:
            episode = episode_data['episode']
            total_reward = sum(round_data['round_reward'] for round_data in episode_data['rounds'])
            self.episode_reward_data[episode] = total_reward
        
        print(f"Analysis completed for {len(self.episode_reward_data)} episodes")
    
    def generate_simple_pdf(self):
        """Generate simple PDF with total reward plot and average line."""
        if not self.episode_reward_data:
            print("No reward data to plot")
            return
        
        episodes = sorted(self.episode_reward_data.keys())
        total_rewards = [self.episode_reward_data[ep] for ep in episodes]
        
        # Calculate average reward
        avg_reward = np.mean(total_rewards)
        
        # Create PDF
        pdf_path = self.plots_dir / 'total_rewards_simple.pdf'
        
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Main plot - clean line with markers
            ax.plot(episodes, total_rewards, 'o-', color='#2E86AB', linewidth=3, 
                   markersize=6, alpha=0.9, markerfacecolor='white', 
                   markeredgecolor='#2E86AB', markeredgewidth=2, label='Episode Rewards')
            
            # Add average line
            ax.axhline(y=avg_reward, color='#E74C3C', linestyle='--', linewidth=2.5, 
                      alpha=0.8, label=f'Average: {avg_reward/1e6:.2f}M')
            
            # Set title and labels
            ax.set_title('Total Reward per Episode Across All 50 Episodes', 
                        fontsize=20, fontweight='bold', pad=25)
            ax.set_xlabel('Episode', fontsize=16, fontweight='bold')
            ax.set_ylabel('Total Episode Reward', fontsize=16, fontweight='bold')
            
            # Add legend
            legend = ax.legend(fontsize=14, loc='upper right', frameon=True, 
                             fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            # Customize grid
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Set axis limits
            ax.set_xlim(0, 51)
            
            # Format y-axis to show values in millions
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            
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
        
        print(f"Simple total rewards PDF with average line saved to {pdf_path}")


def main():
    """Main function."""
    print("="*60)
    print("SIMPLE TOTAL REWARDS PDF GENERATOR")
    print("="*60)
    
    generator = SimpleTotalRewardsPDFGenerator()
    generator.analyze_episode_rewards()
    generator.generate_simple_pdf()
    
    if generator.episode_reward_data:
        episodes = sorted(generator.episode_reward_data.keys())
        total_rewards = [generator.episode_reward_data[ep] for ep in episodes]
        avg_reward = np.mean(total_rewards)
        
        print(f"\nGenerated clean plot for {len(episodes)} episodes")
        print(f"Average reward across all episodes: {avg_reward:,.0f}")
        print(f"File: plots/total_rewards_simple.pdf")
    
    print("="*60)


if __name__ == "__main__":
    main()