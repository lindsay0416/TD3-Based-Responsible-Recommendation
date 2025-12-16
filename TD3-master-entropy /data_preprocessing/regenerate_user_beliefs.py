#!/usr/bin/env python3
"""
Regenerate user_average_beliefs.json with correct formula:
belief[i] = cluster_accepted_count[i] / total_items_in_dataset
"""

import pandas as pd
import json
import numpy as np
import ast
import os
import pickle
from collections import defaultdict

print("="*60)
print("Regenerating User Average Beliefs")
print("="*60)

# Load cluster data
print("\nLoading cluster data...")
with open('embeddings/cluster_matrix_manifest_K5_topk5.json', 'r') as f:
    cluster_data = json.load(f)

cluster_sizes = np.array([
    int(cluster_data['cluster_sizes']['0']),
    int(cluster_data['cluster_sizes']['1']),
    int(cluster_data['cluster_sizes']['2']),
    int(cluster_data['cluster_sizes']['3']),
    int(cluster_data['cluster_sizes']['4'])
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
print("\nLoading natural belief target...")
with open('processed_data/natural_belief_target.json', 'r') as f:
    target_data = json.load(f)
natural_belief_target = np.array(target_data['natural_belief_target'])
print(f"Natural belief target: {natural_belief_target}")

# Load user beliefs TSV
print("\nLoading user_beliefs_K5_topk5.tsv...")
df = pd.read_csv('embeddings/user_beliefs_K5_topk5.tsv', sep='\t')
print(f"Loaded {len(df)} records")
print(f"Unique users: {df['userid'].nunique()}")

# Collect all accepted items per user
print("\nProcessing user accepted items...")
user_accepted_items = defaultdict(list)

for idx, row in df.iterrows():
    if idx % 10000 == 0:
        print(f"  Processed {idx}/{len(df)} records...")
    
    user_id = row['userid']
    
    # Parse impression_1 (accepted items)
    impression_1_str = row['impression_1']
    if pd.notna(impression_1_str) and isinstance(impression_1_str, str):
        try:
            accepted_items = ast.literal_eval(impression_1_str)
            if isinstance(accepted_items, list):
                user_accepted_items[user_id].extend(accepted_items)
        except:
            pass

print(f"\nCollected accepted items for {len(user_accepted_items)} users")

# Calculate beliefs for each user
print("\nCalculating beliefs...")
user_beliefs_data = {}
users_with_no_items = 0
users_with_items = 0

for user_id, accepted_items in user_accepted_items.items():
    # Count items per cluster
    cluster_counts = np.zeros(5, dtype=np.float32)
    
    for item_id in accepted_items:
        if item_id in item_to_cluster:
            cluster = item_to_cluster[item_id]
            cluster_counts[cluster] += 1
    
    # Calculate beliefs using correct formula
    beliefs = cluster_counts / total_items_in_dataset
    
    # Calculate distance to natural target
    pp1_distance = np.abs(beliefs - natural_belief_target).sum()
    
    # Calculate cluster distances (same as natural_belief_target for now)
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
output_json_path = 'processed_data/user_average_beliefs.json'
print(f"\nSaving to {output_json_path}...")
with open(output_json_path, 'w') as f:
    json.dump(user_beliefs_data, f, indent=2)

print(f"Saved JSON! File size: {len(json.dumps(user_beliefs_data)) / 1024 / 1024:.2f} MB")

# Also save to PKL for compatibility
output_pkl_path = 'processed_data/user_average_beliefs.pkl'
print(f"\nSaving to {output_pkl_path}...")

# Convert to simple dict format for PKL (just beliefs array per user)
pkl_data = {user_id: data['beliefs'] for user_id, data in user_beliefs_data.items()}

with open(output_pkl_path, 'wb') as f:
    pickle.dump(pkl_data, f)

print(f"Saved PKL! File size: {os.path.getsize(output_pkl_path) / 1024 / 1024:.2f} MB")

# Display sample results
print("\n" + "="*60)
print("Sample User Beliefs")
print("="*60)

sample_users = ['U100', 'U1000', 'U10001']
for user_id in sample_users:
    if user_id in user_beliefs_data:
        data = user_beliefs_data[user_id]
        beliefs = np.array(data['beliefs'])
        
        print(f"\n{user_id}:")
        print(f"  Beliefs: {beliefs}")
        print(f"  Sum: {beliefs.sum():.6f}")
        print(f"  PP1 distance: {data['pp1_distance']:.6f}")
        
        # Calculate implied counts
        counts = beliefs * total_items_in_dataset
        print(f"  Implied counts: {counts}")
        print(f"  Total accepted items: {counts.sum():.0f}")
        
        # Distance to target
        distance = np.abs(natural_belief_target - beliefs)
        print(f"  Distance to target: {distance}")
        print(f"  Average distance: {distance.mean():.6f}")

# Summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

all_beliefs = np.array([data['beliefs'] for data in user_beliefs_data.values()])
print(f"Shape: {all_beliefs.shape}")
print(f"Mean per cluster: {np.mean(all_beliefs, axis=0)}")
print(f"Std per cluster: {np.std(all_beliefs, axis=0)}")
print(f"Min per cluster: {np.min(all_beliefs, axis=0)}")
print(f"Max per cluster: {np.max(all_beliefs, axis=0)}")
print(f"Mean sum of beliefs: {np.sum(all_beliefs, axis=1).mean():.6f}")
print(f"Max sum of beliefs: {np.sum(all_beliefs, axis=1).max():.6f}")

print("\n" + "="*60)
print("✓ Regeneration Complete!")
print("="*60)
