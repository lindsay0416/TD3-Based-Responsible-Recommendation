#!/usr/bin/env python3
"""
Preprocessing script for movies dataset.
Generates all required processed_data files for movies/Topk5.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

# Configuration
DATASET = "movies"
TOPK = 5
EMBEDDINGS_BASE = f"embeddings/{DATASET}/Topk{TOPK}"
PROCESSED_BASE = f"processed_data/{DATASET}/Topk{TOPK}"
DATA_BASE = f"data/{DATASET}"

def create_directories():
    """Create necessary directories."""
    Path(PROCESSED_BASE).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {PROCESSED_BASE}")

def generate_cluster_embeddings():
    """Generate cluster embeddings from item embeddings."""
    print("\n" + "="*60)
    print("1. Generating Cluster Embeddings")
    print("="*60)
    
    # Load item embeddings (from any RS model, they should be similar)
    item_emb_path = f"{EMBEDDINGS_BASE}/NGCF/item_embedding.pkl"
    with open(item_emb_path, 'rb') as f:
        item_embeddings = pickle.load(f)
    print(f"Loaded {len(item_embeddings)} item embeddings")
    
    # Load cluster manifest
    cluster_manifest_path = f"{EMBEDDINGS_BASE}/cluster_matrix_manifest_K5_topk{TOPK}.json"
    with open(cluster_manifest_path, 'r') as f:
        cluster_data = json.load(f)
    
    # Load item token map
    item_token_map_path = f"{EMBEDDINGS_BASE}/item_token_map.json"
    with open(item_token_map_path, 'r') as f:
        item_token_map = json.load(f)
    
    # Calculate cluster embeddings
    cluster_embeddings = {}
    
    for cluster_id, items in cluster_data['cluster_to_items'].items():
        embeddings = []
        
        for item_id in items:
            token = item_token_map.get(item_id)
            if token is not None:
                token_str = str(token)
                if token_str in item_embeddings:
                    embeddings.append(item_embeddings[token_str])
        
        if embeddings:
            # Average embeddings for this cluster
            cluster_embeddings[cluster_id] = np.mean(embeddings, axis=0).astype(np.float32)
            print(f"  Cluster {cluster_id}: {len(embeddings)} items")
    
    # Save cluster embeddings
    output_path = f"{PROCESSED_BASE}/cluster_embeddings.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(cluster_embeddings, f)
    
    print(f"✓ Saved: {output_path}")
    return cluster_embeddings

def generate_natural_belief_target():
    """Generate natural belief target from cluster sizes."""
    print("\n" + "="*60)
    print("2. Generating Natural Belief Target")
    print("="*60)
    
    # Load cluster manifest
    cluster_manifest_path = f"{EMBEDDINGS_BASE}/cluster_matrix_manifest_K5_topk{TOPK}.json"
    with open(cluster_manifest_path, 'r') as f:
        cluster_data = json.load(f)
    
    # Calculate cluster sizes
    cluster_sizes = {}
    total_items = 0
    
    for cluster_id, items in cluster_data['cluster_to_items'].items():
        cluster_sizes[cluster_id] = len(items)
        total_items += len(items)
    
    # Calculate natural belief target (proportional to cluster size)
    natural_belief_target = []
    for i in range(5):
        cluster_id = str(i)
        proportion = cluster_sizes.get(cluster_id, 0) / total_items
        natural_belief_target.append(proportion)
    
    print(f"Total items: {total_items}")
    print(f"Natural belief target: {natural_belief_target}")
    
    # Save as JSON
    output_data = {
        'natural_belief_target': natural_belief_target,
        'cluster_sizes': cluster_sizes,
        'total_items': total_items
    }
    
    output_path = f"{PROCESSED_BASE}/natural_belief_target.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved: {output_path}")
    return natural_belief_target

def generate_user_beliefs():
    """Generate user average beliefs from behaviors."""
    print("\n" + "="*60)
    print("3. Generating User Average Beliefs")
    print("="*60)
    
    # Load cluster manifest
    cluster_manifest_path = f"{EMBEDDINGS_BASE}/cluster_matrix_manifest_K5_topk{TOPK}.json"
    with open(cluster_manifest_path, 'r') as f:
        cluster_data = json.load(f)
    
    # Create item to cluster mapping
    item_to_cluster = {}
    for cluster_id, items in cluster_data['cluster_to_items'].items():
        for item in items:
            item_to_cluster[item] = int(cluster_id)
    
    # Load behaviors file
    behaviors_path = f"{DATA_BASE}/new_behaviors.tsv"
    
    if not os.path.exists(behaviors_path):
        print(f"Warning: {behaviors_path} not found")
        print("Creating minimal user beliefs from user_token_map...")
        
        # Fallback: create minimal beliefs from user token map
        user_token_map_path = f"{EMBEDDINGS_BASE}/user_token_map.json"
        with open(user_token_map_path, 'r') as f:
            user_token_map = json.load(f)
        
        user_beliefs = {}
        for user_id in user_token_map.keys():
            # Initialize with small uniform beliefs
            user_beliefs[user_id] = {
                'beliefs': [0.0002] * 5,
                'pp1_distance': 0.2,
                'cluster_distances': [0.2] * 5
            }
        
        print(f"Created minimal beliefs for {len(user_beliefs)} users")
    else:
        # Parse behaviors file
        user_interactions = {}
        
        with open(behaviors_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                user_id = parts[1]
                history = parts[3].split()
                
                if user_id not in user_interactions:
                    user_interactions[user_id] = []
                
                user_interactions[user_id].extend(history)
        
        # Calculate beliefs
        user_beliefs = {}
        
        for user_id, items in user_interactions.items():
            cluster_counts = [0] * 5
            
            for item_id in items:
                cluster = item_to_cluster.get(item_id)
                if cluster is not None:
                    cluster_counts[cluster] += 1
            
            total = sum(cluster_counts)
            if total > 0:
                beliefs = [count / total for count in cluster_counts]
            else:
                beliefs = [0.0002] * 5
            
            # Calculate distance to natural target (placeholder)
            pp1_distance = 0.2
            cluster_distances = [0.2] * 5
            
            user_beliefs[user_id] = {
                'beliefs': beliefs,
                'pp1_distance': pp1_distance,
                'cluster_distances': cluster_distances
            }
        
        print(f"Calculated beliefs for {len(user_beliefs)} users")
    
    # Save user beliefs
    output_path = f"{PROCESSED_BASE}/user_average_beliefs.json"
    with open(output_path, 'w') as f:
        json.dump(user_beliefs, f, indent=2)
    
    print(f"✓ Saved: {output_path}")
    return user_beliefs

def main():
    """Main preprocessing function."""
    print("="*60)
    print(f"PREPROCESSING: {DATASET} / Topk{TOPK}")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Run preprocessing steps
    try:
        cluster_embeddings = generate_cluster_embeddings()
        natural_target = generate_natural_belief_target()
        user_beliefs = generate_user_beliefs()
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nGenerated files in: {PROCESSED_BASE}/")
        print("  - cluster_embeddings.pkl")
        print("  - natural_belief_target.json")
        print("  - user_average_beliefs.json")
        print("\nYou can now run training with:")
        print("  python run_recommendation_rl.py")
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
