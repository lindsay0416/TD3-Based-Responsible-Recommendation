import pickle
import json
import numpy as np

def save_natural_belief_as_target():
    """
    Save the natural (proportional) belief as the target for RL optimization
    """
    # Load the calculated beliefs
    with open('processed_data/cluster_size_beliefs.pkl', 'rb') as f:
        belief_methods = pickle.load(f)
    
    # Extract the natural (proportional) distribution
    natural_belief = belief_methods['proportional']
    
    print("=== NATURAL BELIEF TARGET ===")
    print(f"Natural belief vector: {natural_belief}")
    print(f"Sum: {np.sum(natural_belief):.6f}")
    
    # Show interpretation
    print(f"\nInterpretation:")
    for i in range(5):
        percentage = natural_belief[i] * 100
        print(f"  Cluster {i}: {percentage:.1f}% (natural exposure probability)")
    
    # Save as target for RL
    target_data = {
        'natural_belief_target': natural_belief.tolist(),
        'description': 'Natural belief distribution based on cluster sizes - target for filter bubble reduction',
        'cluster_sizes': belief_methods['cluster_sizes'],
        'total_items': belief_methods['total_items'],
        'usage': 'Use this as target belief in RL reward function to guide users away from filter bubbles'
    }
    
    # Save as pickle for RL environment
    with open('processed_data/natural_belief_target.pkl', 'wb') as f:
        pickle.dump(natural_belief, f)
    
    # Save detailed info as JSON
    with open('processed_data/natural_belief_target.json', 'w') as f:
        json.dump(target_data, f, indent=2)
    
    print(f"\n✅ Saved natural belief target:")
    print(f"  - natural_belief_target.pkl (for RL environment)")
    print(f"  - natural_belief_target.json (with metadata)")
    
    return natural_belief

def calculate_filter_bubble_metrics(natural_belief):
    """
    Create utility functions for measuring filter bubble effect
    """
    print(f"\n=== FILTER BUBBLE MEASUREMENT UTILITIES ===")
    
    # Example: Calculate KL divergence from natural distribution
    def kl_divergence_from_natural(user_belief, natural_belief):
        """Calculate KL divergence: how far user belief is from natural"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        user_belief_safe = user_belief + epsilon
        natural_belief_safe = natural_belief + epsilon
        
        kl_div = np.sum(user_belief_safe * np.log(user_belief_safe / natural_belief_safe))
        return kl_div
    
    # Example: Calculate cosine similarity
    def cosine_similarity_to_natural(user_belief, natural_belief):
        """Calculate cosine similarity: how aligned user belief is with natural"""
        dot_product = np.dot(user_belief, natural_belief)
        norm_user = np.linalg.norm(user_belief)
        norm_natural = np.linalg.norm(natural_belief)
        return dot_product / (norm_user * norm_natural)
    
    # Save utility functions as code template
    utility_code = f'''
import numpy as np

# Natural belief target (load this in your RL environment)
NATURAL_BELIEF_TARGET = np.array({natural_belief.tolist()})

def calculate_filter_bubble_score(user_belief):
    """
    Calculate how much user is in a filter bubble (0 = no bubble, higher = more bubble)
    Uses KL divergence from natural distribution
    """
    epsilon = 1e-8
    user_belief_safe = user_belief + epsilon
    natural_belief_safe = NATURAL_BELIEF_TARGET + epsilon
    
    kl_div = np.sum(user_belief_safe * np.log(user_belief_safe / natural_belief_safe))
    return kl_div

def calculate_diversity_score(user_belief):
    """
    Calculate how diverse user's interests are (0 = focused, 1 = diverse)
    Uses cosine similarity to natural distribution
    """
    dot_product = np.dot(user_belief, NATURAL_BELIEF_TARGET)
    norm_user = np.linalg.norm(user_belief)
    norm_natural = np.linalg.norm(NATURAL_BELIEF_TARGET)
    return dot_product / (norm_user * norm_natural)

def calculate_rl_reward(user_belief_before, user_belief_after):
    """
    Calculate RL reward based on movement toward natural distribution
    Positive reward if user moves closer to natural distribution
    """
    bubble_score_before = calculate_filter_bubble_score(user_belief_before)
    bubble_score_after = calculate_filter_bubble_score(user_belief_after)
    
    # Reward for reducing filter bubble (lower KL divergence is better)
    reward = bubble_score_before - bubble_score_after
    return reward

# Example usage in RL environment:
# reward = calculate_rl_reward(old_user_belief, new_user_belief)
# filter_bubble_level = calculate_filter_bubble_score(current_user_belief)
# diversity_level = calculate_diversity_score(current_user_belief)
'''
    
    with open('processed_data/filter_bubble_utilities.py', 'w') as f:
        f.write(utility_code)
    
    print(f"  - filter_bubble_utilities.py (utility functions for RL)")
    
    # Test with sample user beliefs
    print(f"\n=== EXAMPLE FILTER BUBBLE ANALYSIS ===")
    
    # Load some sample user beliefs for demonstration
    try:
        with open('processed_data/user_belief_vectors.pkl', 'rb') as f:
            user_beliefs = pickle.load(f)
        
        sample_users = list(user_beliefs.keys())[:3]
        
        for userid in sample_users:
            user_belief = user_beliefs[userid]
            
            # Calculate metrics
            kl_div = kl_divergence_from_natural(user_belief, natural_belief)
            cos_sim = cosine_similarity_to_natural(user_belief, natural_belief)
            
            print(f"  User {userid}:")
            print(f"    Belief: {user_belief}")
            print(f"    Filter bubble score (KL div): {kl_div:.4f}")
            print(f"    Diversity score (cos sim): {cos_sim:.4f}")
            
            # Identify dominant cluster
            dominant_cluster = np.argmax(user_belief)
            natural_dominant = np.argmax(natural_belief)
            
            if dominant_cluster != natural_dominant:
                print(f"    ⚠️  Filter bubble detected: prefers cluster {dominant_cluster} over natural cluster {natural_dominant}")
            else:
                print(f"    ✅ Aligned with natural preference for cluster {dominant_cluster}")
    
    except FileNotFoundError:
        print("  (User beliefs not loaded - run this after user belief processing)")

if __name__ == "__main__":
    print("=== SETTING UP NATURAL BELIEF TARGET FOR RL ===")
    
    # Save natural belief as target
    natural_belief = save_natural_belief_as_target()
    
    # Create filter bubble measurement utilities
    calculate_filter_bubble_metrics(natural_belief)
    
    print(f"\n🎯 READY FOR RL IMPLEMENTATION!")
    print(f"Your RL agent can now:")
    print(f"  1. Measure filter bubble effect using KL divergence")
    print(f"  2. Reward recommendations that move users toward natural distribution")
    print(f"  3. Promote diversity while respecting natural item availability")