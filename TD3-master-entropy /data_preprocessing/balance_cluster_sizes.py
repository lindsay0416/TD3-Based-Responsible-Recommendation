#!/usr/bin/env python3
"""
Balance cluster sizes by sampling Cluster 0 to match the size of other clusters.
This helps the RL agent learn all clusters more equally.

Current distribution:
- Cluster 0: 13,383 items (70%)
- Cluster 1: 1,355 items (7%)
- Cluster 2: 1,413 items (7%)
- Cluster 3: 1,381 items (7%)
- Cluster 4: 1,598 items (8%)

Strategy: Sample Cluster 0 down to ~3,000 items (2x the average of other clusters)
This maintains Cluster 0's importance while reducing dominance.
"""

import json
import random
import numpy as np
from pathlib import Path


def load_cluster_data():
    """Load cluster assignments."""
    with open('embeddings/cluster_matrix_manifest_K5_topk5.json', 'r') as f:
        return json.load(f)


def balance_clusters(cluster_data, target_cluster0_size=3000, seed=42):
    """
    Balance cluster sizes by sampling Cluster 0.
    
    Args:
        cluster_data: Original cluster data
        target_cluster0_size: Target size for Cluster 0 (default: 3000)
        seed: Random seed for reproducibility
    
    Returns:
        Balanced cluster data
    """
    random.seed(seed)
    np.random.seed(seed)
    
    cluster_to_items = cluster_data['cluster_to_items']
    
    # Print original distribution
    print("Original Cluster Distribution:")
    print("="*60)
    total_items = sum(len(items) for items in cluster_to_items.values())
    for cluster_id in sorted(cluster_to_items.keys(), key=int):
        count = len(cluster_to_items[cluster_id])
        percentage = count / total_items * 100
        print(f"  Cluster {cluster_id}: {count:5d} items ({percentage:5.2f}%)")
    print(f"  Total: {total_items} items")
    
    # Sample Cluster 0
    cluster0_items = cluster_to_items['0']
    original_cluster0_size = len(cluster0_items)
    
    if original_cluster0_size > target_cluster0_size:
        print(f"\nSampling Cluster 0 from {original_cluster0_size} to {target_cluster0_size} items...")
        sampled_cluster0 = random.sample(cluster0_items, target_cluster0_size)
        cluster_to_items['0'] = sampled_cluster0
    else:
        print(f"\nCluster 0 size ({original_cluster0_size}) already <= target ({target_cluster0_size}), no sampling needed")
        sampled_cluster0 = cluster0_items
    
    # Calculate cluster sizes for metadata
    cluster_sizes_dict = {str(i): len(cluster_to_items[str(i)]) for i in range(5)}
    new_total = sum(cluster_sizes_dict.values())
    natural_belief = [cluster_sizes_dict[str(i)] / new_total for i in range(5)]
    
    # Create balanced data with all required metadata
    balanced_data = {
        'cluster_to_items': cluster_to_items,
        'cluster_sizes': cluster_sizes_dict,
        'num_clusters': 5,
        'num_nodes': new_total,
        'avg_belief': natural_belief,
        'avg_belief_cluster_order': list(range(5)),
        'avg_belief_raw': natural_belief,
        'tag': 'balanced_K5_topk5',
        'labels_path': cluster_data.get('labels_path', ''),
        'matrix': cluster_data.get('matrix', ''),
        'original_cluster0_size': original_cluster0_size,
        'sampled_cluster0_size': len(sampled_cluster0),
        'sampling_ratio': len(sampled_cluster0) / original_cluster0_size,
        'seed': seed,
        'note': 'Cluster 0 sampled to reduce dominance and improve RL training balance'
    }
    
    # Print new distribution
    print("\nBalanced Cluster Distribution:")
    print("="*60)
    new_total = sum(len(items) for items in cluster_to_items.values())
    for cluster_id in sorted(cluster_to_items.keys(), key=int):
        count = len(cluster_to_items[cluster_id])
        percentage = count / new_total * 100
        print(f"  Cluster {cluster_id}: {count:5d} items ({percentage:5.2f}%)")
    print(f"  Total: {new_total} items")
    
    # Calculate new natural belief target
    print("\nNew Natural Belief Target:")
    print("="*60)
    for cluster_id in sorted(cluster_to_items.keys(), key=int):
        count = len(cluster_to_items[cluster_id])
        target = count / new_total
        print(f"  Cluster {cluster_id}: {target:.4f} ({target*100:.2f}%)")
    
    return balanced_data


def save_balanced_data(balanced_data, output_path='embeddings/cluster_matrix_manifest_K5_topk5_balanced.json'):
    """Save balanced cluster data."""
    with open(output_path, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    print(f"\n✅ Saved balanced cluster data to: {output_path}")


def update_natural_belief_target(balanced_data):
    """Calculate and save new natural belief target based on balanced clusters."""
    cluster_to_items = balanced_data['cluster_to_items']
    
    # Calculate cluster sizes
    cluster_sizes = [len(cluster_to_items[str(i)]) for i in range(5)]
    total_items = sum(cluster_sizes)
    
    # Calculate natural belief target (proportional to cluster size)
    natural_belief_target = [size / total_items for size in cluster_sizes]
    
    # Create target data
    target_data = {
        'natural_belief_target': natural_belief_target,
        'description': 'Natural belief distribution based on BALANCED cluster sizes',
        'cluster_sizes': cluster_sizes,
        'total_items': total_items,
        'usage': 'Use this as target belief in RL reward function (balanced version)',
        'note': 'Cluster 0 was sampled down to reduce dominance'
    }
    
    # Save
    output_path = 'processed_data/natural_belief_target_balanced.json'
    with open(output_path, 'w') as f:
        json.dump(target_data, f, indent=2)
    
    print(f"✅ Saved balanced natural belief target to: {output_path}")
    
    return target_data


def main():
    """Main function."""
    print("="*60)
    print("Cluster Size Balancing Tool")
    print("="*60)
    
    # Load data
    print("\nLoading cluster data...")
    cluster_data = load_cluster_data()
    
    # Balance clusters
    # Options for target_cluster0_size:
    # - 3000: Moderate reduction (Cluster 0 still largest but not dominant)
    # - 2000: Aggressive reduction (Cluster 0 similar to others)
    # - 1500: Equal distribution (all clusters similar size)
    
    target_size = 3000  # Change this value to adjust balance
    balanced_data = balance_clusters(cluster_data, target_cluster0_size=target_size)
    
    # Save balanced data
    save_balanced_data(balanced_data)
    
    # Update natural belief target
    target_data = update_natural_belief_target(balanced_data)
    
    print("\n" + "="*60)
    print("Balancing Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Update config.yaml to use balanced cluster file:")
    print("   - Change cluster file path in environment initialization")
    print("2. Update config.yaml to use balanced natural belief target:")
    print("   - data_paths.natural_beliefs: 'processed_data/natural_belief_target_balanced.json'")
    print("3. Retrain the model with balanced data")
    print("\nNote: Keep original files as backup!")


if __name__ == "__main__":
    main()
