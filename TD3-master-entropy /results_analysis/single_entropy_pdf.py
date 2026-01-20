#!/usr/bin/env python3
"""
Single Entropy PDF Generator: Clean Average Entropy Plot for All 50 Episodes
Generates a single-page PDF with optimized layout and statistics positioning.
"""

import json
import yaml
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

class SingleEntropyPDFGenerator:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load config to get dataset and topk
        self.load_config()
        
        # Load cluster mapping
        self.load_cluster_mapping()
        
        # All episodes to analyze
        self.all_episodes = list(range(1, 51))  # Episodes 1-50
        
        # Storage for analysis results
        self.episode_entropy_data = {}
    
    def load_config(self):
        """Load configuration from config.yaml."""
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        self.dataset = config['experiment_params']['dataset']
        self.topk = config['experiment_params']['topk']
        print(f"Using dataset: {self.dataset}, topk: {self.topk}")
        
    def load_cluster_mapping(self):
        """Load item to cluster mapping."""
        print("Loading cluster mapping...")
        cluster_path = f"embeddings/{self.dataset}/Topk{self.topk}/cluster_matrix_manifest_K5_topk{self.topk}.json"
        print(f"Loading from: {cluster_path}")
        with open(cluster_path, "r") as f:
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
    
    def analyze_episode_recommendations(self, episode: int) -> Dict:
        """Analyze recommendations for a single episode."""
        rec_file = self.results_dir / f"recommendations_ep{episode}.json"
        
        if not rec_file.exists():
            print(f"Warning: {rec_file} not found")
            return None
        
        if episode % 10 == 0 or episode in [1, 5]:
            print(f"Analyzing episode {episode} recommendations...")
        
        # Load recommendations data
        with open(rec_file, 'r') as f:
            data = json.load(f)
        
        episode_entropies = []
        total_recommendations = 0
        
        # Process each round in the episode
        for round_key in sorted(data['rounds'].keys(), key=lambda x: int(x.split('_')[1])):
            round_data = data['rounds'][round_key]
            
            # Process each user's recommendations in this round
            for user_rec in round_data['recommendations']:
                recommended_items = user_rec['recommended_items']
                
                # Map items to clusters
                cluster_ids = []
                for item_id in recommended_items:
                    item_str = str(item_id)
                    if item_str in self.item_to_cluster:
                        cluster_id = self.item_to_cluster[item_str]
                        cluster_ids.append(cluster_id)
                        total_recommendations += 1
                
                # Calculate entropy for this user's recommendations
                if cluster_ids:
                    user_entropy = self.calculate_entropy(cluster_ids)
                    episode_entropies.append(user_entropy)
        
        # Calculate episode statistics
        avg_entropy = np.mean(episode_entropies) if episode_entropies else 0.0
        
        return {
            'episode': episode,
            'avg_entropy': avg_entropy,
            'total_recommendations': total_recommendations
        }
    
    def analyze_all_episodes(self):
        """Analyze all 50 episodes for entropy."""
        print(f"Starting entropy analysis for all 50 episodes...")
        
        for episode in self.all_episodes:
            result = self.analyze_episode_recommendations(episode)
            if result:
                self.episode_entropy_data[episode] = result
        
        print(f"Analysis completed for {len(self.episode_entropy_data)} episodes")
    
    def generate_single_entropy_pdf(self):
        """Generate a single-page PDF with clean entropy plot and centered statistics."""
        if not self.episode_entropy_data:
            print("No entropy data available")
            return
        
        episodes = sorted(self.episode_entropy_data.keys())
        entropies = [self.episode_entropy_data[ep]['avg_entropy'] for ep in episodes]
        
        # Create PDF
        pdf_path = self.plots_dir / 'average_entropy_50episodes_clean.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Create figure with optimal size
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Main entropy plot
            ax.plot(episodes, entropies, 'o-', color='#2E86AB', linewidth=3, 
                   markersize=6, alpha=0.9, markerfacecolor='white', 
                   markeredgecolor='#2E86AB', markeredgewidth=2)
            
            # Add trend line
            z = np.polyfit(episodes, entropies, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "--", color='#E74C3C', alpha=0.8, linewidth=2.5)
            
            # Set title and labels
            ax.set_title('Average Entropy Across All 50 Episodes', 
                        fontsize=20, fontweight='bold', pad=25)
            ax.set_xlabel('Episode', fontsize=16, fontweight='bold')
            ax.set_ylabel('Normalized Entropy', fontsize=16, fontweight='bold')
            
            # Customize grid
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Calculate statistics
            avg_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            min_entropy = np.min(entropies)
            max_entropy = np.max(entropies)
            min_episode = episodes[np.argmin(entropies)]
            max_episode = episodes[np.argmax(entropies)]
            
            # Handle zero entropy case to avoid division by zero
            if entropies[0] > 0:
                improvement = ((entropies[-1] - entropies[0]) / entropies[0] * 100)
            else:
                # If starting entropy is 0, show absolute change instead
                improvement = entropies[-1] * 100 if entropies[-1] > 0 else 0.0
            
            # Highlight first and last episodes with larger markers
            ax.scatter([episodes[0]], [entropies[0]], color='#27AE60', s=150, 
                      zorder=5, edgecolor='white', linewidth=2, 
                      label=f'Episode 1: {entropies[0]:.4f}')
            ax.scatter([episodes[-1]], [entropies[-1]], color='#F39C12', s=150, 
                      zorder=5, edgecolor='white', linewidth=2,
                      label=f'Episode 50: {entropies[-1]:.4f}')
            
            # Add trend line to legend
            ax.plot([], [], "--", color='#E74C3C', alpha=0.8, linewidth=2.5,
                   label=f'Trend Line (slope: {z[0]:.6f})')
            
            # Customize legend
            legend = ax.legend(fontsize=12, loc='lower right', frameon=True, 
                             fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            # Set axis limits with some padding
            ax.set_xlim(0, 51)
            entropy_range = max_entropy - min_entropy
            y_padding = entropy_range * 0.15
            ax.set_ylim(min_entropy - y_padding, max_entropy + y_padding)
            
            # Customize tick parameters
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            ax.tick_params(axis='both', which='minor', width=1, length=3)
            
            # Set minor ticks
            ax.set_xticks(np.arange(0, 51, 5), minor=False)
            ax.set_xticks(np.arange(0, 51, 1), minor=True)
            
            # Customize spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
        
        print(f"Clean entropy PDF saved to {pdf_path}")
        
        # Also save as high-quality PNG
        png_path = self.plots_dir / 'average_entropy_50episodes_clean.png'
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Recreate the same plot for PNG
        ax.plot(episodes, entropies, 'o-', color='#2E86AB', linewidth=3, 
               markersize=6, alpha=0.9, markerfacecolor='white', 
               markeredgecolor='#2E86AB', markeredgewidth=2)
        
        ax.plot(episodes, p(episodes), "--", color='#E74C3C', alpha=0.8, linewidth=2.5)
        
        ax.set_title('Average Entropy Across All 50 Episodes', 
                    fontsize=20, fontweight='bold', pad=25)
        ax.set_xlabel('Episode', fontsize=16, fontweight='bold')
        ax.set_ylabel('Normalized Entropy', fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        ax.scatter([episodes[0]], [entropies[0]], color='#27AE60', s=150, 
                  zorder=5, edgecolor='white', linewidth=2, 
                  label=f'Episode 1: {entropies[0]:.4f}')
        ax.scatter([episodes[-1]], [entropies[-1]], color='#F39C12', s=150, 
                  zorder=5, edgecolor='white', linewidth=2,
                  label=f'Episode 50: {entropies[-1]:.4f}')
        
        ax.plot([], [], "--", color='#E74C3C', alpha=0.8, linewidth=2.5,
               label=f'Trend Line (slope: {z[0]:.6f})')
        
        legend = ax.legend(fontsize=12, loc='upper left', frameon=True, 
                         fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax.set_xlim(0, 51)
        ax.set_ylim(min_entropy - y_padding, max_entropy + y_padding)
        
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        ax.set_xticks(np.arange(0, 51, 5), minor=False)
        ax.set_xticks(np.arange(0, 51, 1), minor=True)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#333333')
        
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Clean entropy PNG saved to {png_path}")


def main():
    """Main function to generate clean entropy PDF."""
    print("="*80)
    print("SINGLE ENTROPY PDF GENERATOR: CLEAN 50 EPISODES PLOT")
    print("="*80)
    
    generator = SingleEntropyPDFGenerator()
    
    # Analyze all episodes
    generator.analyze_all_episodes()
    
    # Generate clean PDF
    generator.generate_single_entropy_pdf()
    
    # Print summary
    if generator.episode_entropy_data:
        episodes = sorted(generator.episode_entropy_data.keys())
        first_entropy = generator.episode_entropy_data[episodes[0]]['avg_entropy']
        last_entropy = generator.episode_entropy_data[episodes[-1]]['avg_entropy']
        
        print(f"\nANALYSIS SUMMARY:")
        print(f"="*50)
        print(f"Episodes analyzed: {len(episodes)}")
        print(f"Entropy Episode 1: {first_entropy:.4f}")
        print(f"Entropy Episode 50: {last_entropy:.4f}")
        if first_entropy > 0:
            print(f"Entropy improvement: {((last_entropy - first_entropy) / first_entropy * 100):+.2f}%")
        else:
            print(f"Entropy improvement: N/A (starting entropy was 0)")
        print(f"Average entropy: {np.mean([generator.episode_entropy_data[ep]['avg_entropy'] for ep in episodes]):.4f}")
        print(f"Standard deviation: {np.std([generator.episode_entropy_data[ep]['avg_entropy'] for ep in episodes]):.4f}")
    
    print(f"\nGenerated Files:")
    print(f"  📄 plots/average_entropy_50episodes_clean.pdf")
    print(f"  🖼️  plots/average_entropy_50episodes_clean.png")
    print("="*80)


if __name__ == "__main__":
    main()