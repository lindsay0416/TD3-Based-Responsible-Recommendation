#!/usr/bin/env python3
"""
Regenerate user_average_beliefs.json with correct formula:
belief[i] = cluster_accepted_count[i] / total_items_in_dataset

Usage:
    python data_preprocessing/regenerate_user_beliefs_configurable.py --config config.yaml
"""

import pandas as pd
import json
import numpy as np
import ast
import os
import pickle
import yaml
import argparse
from collections import defaultdict


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def regenerate_user_beliefs(config):
    print("="*60)
    print("Regenerating User Average Beliefs")
    print("="*60)

    # Load cluster data
    cluster_manifest_path = config['input_files']['cluster_manifest']
    print(f"\nLoading cluster data from {cluster_manifest_path}...")
    with open(cluster_manifest_path, 'r') as f:
        cluster_data = json.load(f)

    num_clusters = config['processing']['expected_clusters']
    cluster_sizes = np.array([
        int(cluster_data['cluster_sizes'][str(i)]) for i in range(num_clusters)
    ], dtype=np.float32)

    total_items_in_dataset = np.sum(cluster_sizes)

    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Total items in dataset: {total_items_in_dataset}")

    # Create item to cluster mapping
    print("\nCreating item to cluster mapping...")
    item_to_cluster = {}
    for cluster_id, items in cluster_data['cluster_to_items'].items():
        for item in items:
            item_to_cluster[item] = int(cluster_id)

    print(f"Mapped {len(item_to_cluster)} items to clusters")

    # Load natural belief target
    natural_belief_path = config['input_files']['natural_belief_target']
    print(f"\nLoading natural belief target from {natural_belief_path}...")
    with open(natural_belief_path, 'r') as f:
        target_data = json.load(f)
    natural_belief_target = np.array(target_data['natural_belief_target'])
    print(f"Natural belief target: {natural_belief_target}")

    # Load behaviors file
    behaviors_path = config['input_files']['behaviors_file']
    separator = config['input_files'].get('separator', '\t')
    print(f"\nLoading behaviors from {behaviors_path}...")
    df = pd.read_csv(behaviors_path, sep=separator)
    print(f"Loaded {len(df)} records")
    print(f"Unique users: {df['userid'].nunique()}")

    # Parse impressions and collect accepted items per user
    print("\nProcessing user accepted items...")
    user_accepted_items = defaultdict(list)

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(df)} records...")
        
        user_id = row['userid']
        impression_str = row['impression']
        
        if pd.isna(impression_str):
            continue
            
        # Parse impression column: "item1-1 item2-0 item3-1" format
        tokens = str(impression_str).split()
        for token in tokens:
            if token.endswith('-1'):  # Accepted item
                item_id = token[:-2]  # Remove "-1" suffix
                if item_id in item_to_cluster:
                    user_accepted_items[user_id].append(item_id)

    print(f"\nCollected accepted items for {len(user_accepted_items)} users")

    # Calculate beliefs for each user
    print("\nCalculating beliefs...")
    user_beliefs_data = {}
    users_with_no_items = 0
    users_with_items = 0

    for user_id, accepted_items in user_accepted_items.items():
        # Count items per cluster
        cluster_counts = np.zeros(num_clusters, dtype=np.float32)
        
        for item_id in accepted_items:
            if item_id in item_to_cluster:
                cluster = item_to_cluster[item_id]
                cluster_counts[cluster] += 1
        
        # Calculate beliefs: count / total_items_in_dataset (NOT normalized to sum=1)
        beliefs = cluster_counts / total_items_in_dataset
        
        # Calculate distance to natural target
        pp1_distance = np.abs(beliefs - natural_belief_target).sum()
        
        # Cluster distances (same as natural_belief_target)
        cluster_distances = natural_belief_target.copy()
        
        user_beliefs_data[user_id] = {
            'beliefs': beliefs.tolist(),
            'pp1_distance': float(pp1_distance),
            'cluster_distances': cluster_distances.tolist()
        }
        
        if cluster_counts.sum() > 0:
            users_with_items += 1
        else:
            users_with_no_items += 1

    print(f"Calculated beliefs for {len(user_beliefs_data)} users")
    print(f"  Users with accepted items: {users_with_items}")
    print(f"  Users with no accepted items: {users_with_no_items}")

    # Save to JSON
    output_json_path = config['output_files']['user_average_beliefs_json']
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    print(f"\nSaving to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(user_beliefs_data, f, indent=2)
    print(f"Saved JSON! File size: {os.path.getsize(output_json_path) / 1024 / 1024:.2f} MB")

    # Save to PKL
    output_pkl_path = config['output_files']['user_average_beliefs_pkl']
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    print(f"\nSaving to {output_pkl_path}...")
    pkl_data = {user_id: data['beliefs'] for user_id, data in user_beliefs_data.items()}
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"Saved PKL! File size: {os.path.getsize(output_pkl_path) / 1024 / 1024:.2f} MB")

    # Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    all_beliefs = np.array([data['beliefs'] for data in user_beliefs_data.values()])
    print(f"Shape: {all_beliefs.shape}")
    print(f"Mean per cluster: {np.mean(all_beliefs, axis=0)}")
    print(f"Mean sum of beliefs: {np.sum(all_beliefs, axis=1).mean():.6f}")

    print("\n✓ Regeneration Complete!")


def main():
    parser = argparse.ArgumentParser(description="Regenerate user beliefs from configuration")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    regenerate_user_beliefs(config)


if __name__ == "__main__":
    main()
