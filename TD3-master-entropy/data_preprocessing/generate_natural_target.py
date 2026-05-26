#!/usr/bin/env python3
"""
Single script to generate natural_belief_target.json from cluster manifest.

Usage:
    python data_preprocessing/generate_natural_target.py
    
Or with custom config:
    python data_preprocessing/generate_natural_target.py --config custom_config.yaml
"""

import json
import numpy as np
import pickle
import argparse
import os
import yaml

class NaturalTargetGenerator:
    def __init__(self, config_path="data_preprocessing/natural_target_config.yaml"):
        """Initialize generator with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded configuration from {config_path}")
        return config
    
    def setup_logging(self):
        """Setup logging based on config."""
        self.verbose = self.config.get('logging', {}).get('verbose', True)
        self.show_cluster_details = self.config.get('logging', {}).get('show_cluster_details', True)
        self.show_calculations = self.config.get('logging', {}).get('show_calculations', True)
    
    def log(self, message):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def calculate_cluster_beliefs(self):
        """Calculate belief distributions from cluster manifest."""
        self.log("🔄 Step 1: Calculating cluster size beliefs...")
        
        manifest_path = self.config['input_files']['cluster_manifest']
        
        # Load cluster information
        with open(manifest_path, 'r') as f:
            cluster_data = json.load(f)
        
        cluster_to_items = cluster_data['cluster_to_items']
        expected_clusters = self.config['processing']['expected_clusters']
        
        # Validate cluster count
        if len(cluster_to_items) != expected_clusters:
            self.log(f"⚠️  Warning: Expected {expected_clusters} clusters, found {len(cluster_to_items)}")
        
        # Get cluster sizes
        cluster_sizes = {}
        total_items = 0
        
        if self.show_cluster_details:
            self.log("Cluster sizes:")
        
        for cluster_id, items in cluster_to_items.items():
            cluster_size = len(items)
            cluster_sizes[cluster_id] = cluster_size
            total_items += cluster_size
            if self.show_cluster_details:
                self.log(f"  Cluster {cluster_id}: {cluster_size:,} items")
        
        self.log(f"  Total items: {total_items:,}")
        
        # Calculate proportional beliefs (natural distribution)
        proportional_beliefs = self.calculate_proportional_beliefs(cluster_sizes, total_items, expected_clusters)
        
        if self.show_calculations:
            self.log(f"\nNatural belief distribution (proportional): {proportional_beliefs}")
            self.log(f"Sum: {np.sum(proportional_beliefs):.6f}")
            
            self.log("Interpretation:")
            for i in range(expected_clusters):
                percentage = proportional_beliefs[i] * 100
                self.log(f"  Cluster {i}: {percentage:.1f}% of all items")
        
        return {
            'selected_beliefs': proportional_beliefs,
            'cluster_sizes': cluster_sizes,
            'total_items': total_items
        }
    
    def calculate_proportional_beliefs(self, cluster_sizes, total_items, num_clusters):
        """Calculate proportional beliefs based on cluster sizes."""
        proportional_beliefs = np.zeros(num_clusters)
        for i in range(num_clusters):
            cluster_size = cluster_sizes[str(i)]
            proportional_beliefs[i] = cluster_size / total_items
        
        return proportional_beliefs

    def save_natural_target(self, belief_data):
        """Save natural belief as target for RL."""
        self.log("\n🔄 Step 2: Saving natural belief target...")
        
        natural_belief = belief_data['selected_beliefs']
        
        # Create target data structure
        target_data = {
            'natural_belief_target': natural_belief.tolist(),
            'description': 'Natural belief distribution based on cluster sizes (proportional method) - target for filter bubble reduction',
            'cluster_sizes': belief_data['cluster_sizes'],
            'total_items': belief_data['total_items'],
            'usage': 'Use this as target belief in RL reward function to guide users away from filter bubbles'
        }
        
        # Get output paths from config
        json_path = self.config['output_files']['natural_belief_target_json']
        pkl_path = self.config['output_files']['natural_belief_target_pkl']
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        
        # Save as JSON (main target file)
        with open(json_path, 'w') as f:
            json.dump(target_data, f, indent=2)
        
        # Save as pickle for fast loading in RL
        with open(pkl_path, 'wb') as f:
            pickle.dump(natural_belief, f)
        
        self.log("✅ Generated files:")
        self.log(f"  - {json_path} (main target file)")
        self.log(f"  - {pkl_path} (for RL environment)")
        
        return natural_belief
    


    def create_utility_functions(self, natural_belief):
        """Create utility functions for filter bubble measurement."""
        if not self.config['output']['generate_utilities']:
            return
            
        self.log("\n🔄 Step 3: Creating utility functions...")
        
        utilities_path = self.config['output_files']['filter_bubble_utilities']
        include_examples = self.config['output']['include_examples']
        
        utility_code = f'''#!/usr/bin/env python3
"""
Utility functions for measuring filter bubble effects and calculating RL rewards.
Generated automatically from natural belief target.
"""

import numpy as np

# Natural belief target (proportional to cluster sizes)
NATURAL_BELIEF_TARGET = np.array({natural_belief.tolist()})

def calculate_filter_bubble_score(user_belief):
    """
    Calculate how much user is in a filter bubble.
    
    Args:
        user_belief: User's current belief distribution (numpy array)
        
    Returns:
        float: Filter bubble score (0 = no bubble, higher = more bubble)
               Uses KL divergence from natural distribution
    """
    epsilon = 1e-8
    user_belief_safe = np.array(user_belief) + epsilon
    natural_belief_safe = NATURAL_BELIEF_TARGET + epsilon
    
    kl_div = np.sum(user_belief_safe * np.log(user_belief_safe / natural_belief_safe))
    return kl_div

def calculate_diversity_score(user_belief):
    """
    Calculate how diverse user's interests are.
    
    Args:
        user_belief: User's current belief distribution (numpy array)
        
    Returns:
        float: Diversity score (0 = focused, 1 = perfectly diverse)
               Uses cosine similarity to natural distribution
    """
    user_belief = np.array(user_belief)
    dot_product = np.dot(user_belief, NATURAL_BELIEF_TARGET)
    norm_user = np.linalg.norm(user_belief)
    norm_natural = np.linalg.norm(NATURAL_BELIEF_TARGET)
    
    if norm_user == 0 or norm_natural == 0:
        return 0.0
    
    return dot_product / (norm_user * norm_natural)

def calculate_rl_reward(user_belief_before, user_belief_after):
    """
    Calculate RL reward based on movement toward natural distribution.
    
    Args:
        user_belief_before: User's belief before recommendation
        user_belief_after: User's belief after recommendation
        
    Returns:
        float: Reward (positive if moved closer to natural distribution)
    """
    bubble_score_before = calculate_filter_bubble_score(user_belief_before)
    bubble_score_after = calculate_filter_bubble_score(user_belief_after)
    
    # Reward for reducing filter bubble (lower KL divergence is better)
    reward = bubble_score_before - bubble_score_after
    return reward

def get_natural_target():
    """Get the natural belief target."""
    return NATURAL_BELIEF_TARGET.copy()

def analyze_user_belief(user_belief, user_id=None):
    """
    Comprehensive analysis of a user's belief distribution.
    
    Args:
        user_belief: User's belief distribution
        user_id: Optional user identifier for display
        
    Returns:
        dict: Analysis results
    """
    user_belief = np.array(user_belief)
    
    # Calculate metrics
    filter_bubble_score = calculate_filter_bubble_score(user_belief)
    diversity_score = calculate_diversity_score(user_belief)
    
    # Find dominant clusters
    user_dominant = np.argmax(user_belief)
    natural_dominant = np.argmax(NATURAL_BELIEF_TARGET)
    
    # Calculate concentration (entropy)
    entropy = -np.sum(user_belief * np.log(user_belief + 1e-8))
    max_entropy = -np.log(1.0 / len(user_belief))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    analysis = {{
        'user_id': user_id,
        'belief_distribution': user_belief.tolist(),
        'filter_bubble_score': filter_bubble_score,
        'diversity_score': diversity_score,
        'dominant_cluster': int(user_dominant),
        'natural_dominant_cluster': int(natural_dominant),
        'is_filter_bubble': user_dominant != natural_dominant,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'concentration_level': 1.0 - normalized_entropy  # 0 = diverse, 1 = concentrated
    }}
    
    return analysis

'''
        
        # Add examples if requested
        if include_examples:
            utility_code += f'''
# Example usage:
if __name__ == "__main__":
    # Example user belief (concentrated on cluster 0)
    example_user_belief = [0.8, 0.05, 0.05, 0.05, 0.05]
    
    print("=== EXAMPLE ANALYSIS ===")
    print(f"Natural target: {{NATURAL_BELIEF_TARGET}}")
    print(f"Example user belief: {{example_user_belief}}")
    
    analysis = analyze_user_belief(example_user_belief, "example_user")
    
    print(f"\\nAnalysis results:")
    for key, value in analysis.items():
        if key != 'belief_distribution':
            print(f"  {{key}}: {{value}}")
    
    # Show what happens with a recommendation
    new_belief = [0.7, 0.1, 0.1, 0.05, 0.05]  # Slightly more diverse
    reward = calculate_rl_reward(example_user_belief, new_belief)
    print(f"\\nRL reward for moving to {{new_belief}}: {{reward:.4f}}")
'''
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(utilities_path), exist_ok=True)
        
        with open(utilities_path, 'w') as f:
            f.write(utility_code)
        
        self.log(f"✅ Created utility functions:")
        self.log(f"  - {utilities_path}")
    
    def process(self):
        """Run the complete processing pipeline."""
        self.log("🚀 Starting Natural Belief Target Generation")
        self.log("=" * 60)
        
        try:
            # Check if input files exist
            manifest_path = self.config['input_files']['cluster_manifest']
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Cluster manifest not found: {manifest_path}")
            
            # Step 1: Calculate cluster beliefs
            belief_data = self.calculate_cluster_beliefs()
            
            # Step 2: Save natural target
            natural_belief = self.save_natural_target(belief_data)
            
            # Step 3: Create utility functions (if enabled)
            self.create_utility_functions(natural_belief)
            
            self.log("\n" + "=" * 60)
            self.log("✅ Natural belief target generation complete!")
            
            self.log("\n📁 Generated files:")
            json_path = self.config['output_files']['natural_belief_target_json']
            pkl_path = self.config['output_files']['natural_belief_target_pkl']
            utilities_path = self.config['output_files']['filter_bubble_utilities']
            
            if os.path.exists(json_path):
                self.log(f"  - {json_path}")
            if os.path.exists(pkl_path):
                self.log(f"  - {pkl_path}")
            if os.path.exists(utilities_path):
                self.log(f"  - {utilities_path}")
            
            self.log("\n🎯 Ready for use in:")
            self.log("  - RL environment reward calculation")
            self.log("  - Filter bubble measurement")
            self.log("  - User belief analysis")
            
        except Exception as e:
            self.log(f"❌ Error: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Generate natural belief target from configuration")
    parser.add_argument("--config", 
                       default="data_preprocessing/natural_target_config.yaml",
                       help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    try:
        generator = NaturalTargetGenerator(args.config)
        generator.process()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())