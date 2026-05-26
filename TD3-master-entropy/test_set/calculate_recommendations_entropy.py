#!/usr/bin/env python3
"""
Calculate Entropy of Recommendation Lists
Uses the same entropy calculation as in the RL reward function.
"""

import pickle
import json
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List


def calculate_entropy(cluster_counts, num_clusters=5):
    """
    Calculate entropy of cluster distribution.
    Same algorithm as entropy reward in recommendation_environment.py
    
    Args:
        cluster_counts: Counter object with cluster_id -> count mapping
        num_clusters: Total number of clusters (default: 5)
        
    Returns:
        entropy: Entropy value (higher = more diverse)
    """
    total = sum(cluster_counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for i in range(num_clusters):
        p = cluster_counts.get(i, 0) / total
        if p > 0:
            entropy += -p * np.log(p)
    
    return entropy


def load_cluster_mapping(cluster_file: str) -> Dict[str, int]:
    """Load item to cluster mapping from cluster manifest file."""
    with open(cluster_file, 'r') as f:
        cluster_data = json.load(f)
    
    item_to_cluster = {}
    for cluster_id, items in cluster_data['cluster_to_items'].items():
        for item in items:
            item_to_cluster[item] = int(cluster_id)
    
    return item_to_cluster


def calculate_recommendations_entropy(recommendations_file: str, item_to_cluster: Dict[str, int], 
                                     token_to_item: Dict[str, str]) -> Dict:
    """
    Calculate entropy for all recommendation lists in a recommendations.pkl file.
    
    Args:
        recommendations_file: Path to recommendations.pkl
        item_to_cluster: Mapping from item_id to cluster_id
        token_to_item: Mapping from token to item_id
        
    Returns:
        Dictionary with entropy statistics
    """
    # Load recommendations
    with open(recommendations_file, 'rb') as f:
        recommendations_list = pickle.load(f)
    
    user_entropies = []
    user_cluster_distributions = {}
    
    for rec in recommendations_list:
        user_id = rec['user_id']
        recommended_items = rec['recommended_items']
        
        # Convert to list if numpy array
        if hasattr(recommended_items, 'tolist'):
            recommended_items = recommended_items.tolist()
        
        # Map items to clusters
        cluster_ids = []
        for item_token in recommended_items:
            item_token_str = str(item_token)
            
            # Convert token to item_id
            item_id = token_to_item.get(item_token_str)
            
            if item_id is not None:
                # Get cluster for this item
                cluster_id = item_to_cluster.get(item_id)
                
                if cluster_id is not None and 0 <= cluster_id < 5:
                    cluster_ids.append(cluster_id)
        
        # Calculate entropy for this user's recommendations
        if cluster_ids:
            cluster_counts = Counter(cluster_ids)
            entropy = calculate_entropy(cluster_counts)
            user_entropies.append(entropy)
            user_cluster_distributions[user_id] = dict(cluster_counts)
    
    # Calculate statistics
    results = {
        'num_users': len(recommendations_list),
        'num_users_with_valid_clusters': len(user_entropies),
        'entropy_stats': {
            'mean': float(np.mean(user_entropies)) if user_entropies else 0.0,
            'std': float(np.std(user_entropies)) if user_entropies else 0.0,
            'min': float(np.min(user_entropies)) if user_entropies else 0.0,
            'max': float(np.max(user_entropies)) if user_entropies else 0.0,
            'median': float(np.median(user_entropies)) if user_entropies else 0.0,
        },
        'max_possible_entropy': float(np.log(5)),  # log(5) for 5 clusters
        'normalized_entropy_stats': {
            'mean': float(np.mean(user_entropies) / np.log(5)) if user_entropies else 0.0,
            'std': float(np.std(user_entropies) / np.log(5)) if user_entropies else 0.0,
        },
        'user_entropies': user_entropies,
        'user_cluster_distributions': user_cluster_distributions
    }
    
    return results


def main():
    """Calculate entropy for ENMF recommendations."""
    # Paths
    base_path = Path("embeddings/news/Topk5")
    cluster_file = base_path / "cluster_matrix_manifest_K5_topk5.json"
    item_token_map_file = base_path / "item_token_map.json"
    
    # Models to analyze
    models = ["ENMF", "LightGCN", "NCL", "NGCF", "SGL"]
    
    # Load cluster mapping
    print("Loading cluster mapping...")
    item_to_cluster = load_cluster_mapping(str(cluster_file))
    print(f"Loaded cluster mapping for {len(item_to_cluster)} items")
    
    # Load item token mapping and create reverse mapping
    print("Loading item token mapping...")
    with open(item_token_map_file, 'r') as f:
        item_token_map = json.load(f)
    
    # Create reverse mapping: token -> item_id
    token_to_item = {str(token): item_id for item_id, token in item_token_map.items()}
    print(f"Loaded token mapping for {len(token_to_item)} items\n")
    
    # Analyze each model
    all_results = {}
    
    for model in models:
        recommendations_file = base_path / model / "recommendations.pkl"
        
        if not recommendations_file.exists():
            print(f"⚠️  {model}: recommendations.pkl not found")
            continue
        
        print(f"Analyzing {model}...")
        results = calculate_recommendations_entropy(str(recommendations_file), item_to_cluster, token_to_item)
        all_results[model] = results
        
        # Print results
        print(f"  Total users: {results['num_users']}")
        print(f"  Users with valid clusters: {results['num_users_with_valid_clusters']}")
        print(f"  Mean entropy (avg across users): {results['entropy_stats']['mean']:.4f} ± {results['entropy_stats']['std']:.4f}")
        print(f"  Entropy range: [{results['entropy_stats']['min']:.4f}, {results['entropy_stats']['max']:.4f}]")
        print(f"  Median entropy: {results['entropy_stats']['median']:.4f}")
        print(f"  Normalized mean entropy: {results['normalized_entropy_stats']['mean']:.4f} (max=1.0)")
        print(f"  Max possible entropy per user: {results['max_possible_entropy']:.4f} (log(5))")
        print()
    
    # Save results
    output_file = Path("test_set") / "recommendations_entropy_results.json"
    
    # Prepare output (exclude large arrays)
    output_data = {}
    for model, results in all_results.items():
        output_data[model] = {
            'num_users': results['num_users'],
            'num_users_with_valid_clusters': results['num_users_with_valid_clusters'],
            'entropy_stats': results['entropy_stats'],
            'max_possible_entropy': results['max_possible_entropy'],
            'normalized_entropy_stats': results['normalized_entropy_stats'],
        }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON ACROSS MODELS (Mean = Average Entropy Across All Users)")
    print("="*70)
    print(f"{'Model':<12} {'Mean Entropy':<15} {'Normalized':<12} {'Std':<10}")
    print("-"*70)
    
    for model in models:
        if model in all_results:
            results = all_results[model]
            mean_entropy = results['entropy_stats']['mean']
            normalized = results['normalized_entropy_stats']['mean']
            std = results['entropy_stats']['std']
            print(f"{model:<12} {mean_entropy:<15.4f} {normalized:<12.4f} {std:<10.4f}")


if __name__ == "__main__":
    main()
