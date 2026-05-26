#!/usr/bin/env python3
"""
Unified dataset preprocessing script.
Generates all processed data files from a single config.

Usage:
    # Generate all files for books
    python data_preprocessing/preprocess_dataset.py --config data_preprocessing/dataset_config_books.yaml
    
    # Generate all files for movies  
    python data_preprocessing/preprocess_dataset.py --config data_preprocessing/dataset_config_movies.yaml
    
    # Generate specific component only
    python data_preprocessing/preprocess_dataset.py --config dataset_config_books.yaml --only natural_target
    python data_preprocessing/preprocess_dataset.py --config dataset_config_books.yaml --only user_beliefs
    python data_preprocessing/preprocess_dataset.py --config dataset_config_books.yaml --only cluster_embeddings
    
    # Override topk from command line
    python data_preprocessing/preprocess_dataset.py --config dataset_config_books.yaml --topk 15
"""

import os
import json
import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict


def load_config(config_path, topk_override=None):
    """Load config and resolve {topk} placeholders."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    topk = topk_override if topk_override else config.get('topk', 5)
    config['topk'] = topk
    
    # Resolve all {topk} placeholders recursively
    def resolve_placeholders(obj):
        if isinstance(obj, str):
            return obj.format(topk=topk)
        elif isinstance(obj, dict):
            return {k: resolve_placeholders(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_placeholders(item) for item in obj]
        return obj
    
    return resolve_placeholders(config)


def ensure_dir(filepath):
    """Ensure directory exists for filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


# ============================================
# 1. Natural Belief Target Generation
# ============================================
def generate_natural_target(config):
    """Generate natural_belief_target.json from cluster manifest."""
    print("\n" + "="*60)
    print("1. Generating Natural Belief Target")
    print("="*60)
    
    cfg = config['natural_target']
    manifest_path = cfg['input']['cluster_manifest']
    
    # Load cluster manifest
    print(f"Loading cluster manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        cluster_data = json.load(f)
    
    cluster_to_items = cluster_data['cluster_to_items']
    num_clusters = cfg['options']['expected_clusters']
    
    # Calculate cluster sizes and proportional beliefs
    cluster_sizes = {}
    total_items = 0
    
    for cluster_id, items in cluster_to_items.items():
        cluster_size = len(items)
        cluster_sizes[cluster_id] = cluster_size
        total_items += cluster_size
        print(f"  Cluster {cluster_id}: {cluster_size:,} items")
    
    print(f"  Total items: {total_items:,}")
    
    # Calculate proportional beliefs
    natural_belief = np.zeros(num_clusters)
    for i in range(num_clusters):
        natural_belief[i] = cluster_sizes[str(i)] / total_items
    
    print(f"\nNatural belief target: {natural_belief}")
    
    # Save JSON
    target_data = {
        'natural_belief_target': natural_belief.tolist(),
        'description': 'Natural belief distribution based on cluster sizes - target for filter bubble reduction',
        'cluster_sizes': cluster_sizes,
        'total_items': total_items,
        'usage': 'Use this as target belief in RL reward function'
    }
    
    json_path = cfg['output']['json']
    ensure_dir(json_path)
    with open(json_path, 'w') as f:
        json.dump(target_data, f, indent=2)
    print(f"✅ Saved: {json_path}")
    
    # Save PKL
    pkl_path = cfg['output']['pkl']
    ensure_dir(pkl_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(natural_belief, f)
    print(f"✅ Saved: {pkl_path}")
    
    return natural_belief


# ============================================
# 2. User Average Beliefs Generation
# ============================================
def generate_user_beliefs(config):
    """Generate user_average_beliefs.json from behaviors data."""
    print("\n" + "="*60)
    print("2. Generating User Average Beliefs")
    print("="*60)
    
    cfg = config['user_beliefs']
    num_clusters = cfg['options']['expected_clusters']
    
    # Load cluster manifest
    manifest_path = cfg['input']['cluster_manifest']
    print(f"Loading cluster manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        cluster_data = json.load(f)
    
    cluster_sizes = np.array([
        int(cluster_data['cluster_sizes'][str(i)]) for i in range(num_clusters)
    ], dtype=np.float32)
    total_items = np.sum(cluster_sizes)
    print(f"Total items in dataset: {total_items}")
    
    # Create item to cluster mapping
    item_to_cluster = {}
    for cluster_id, items in cluster_data['cluster_to_items'].items():
        for item in items:
            item_to_cluster[item] = int(cluster_id)
    print(f"Mapped {len(item_to_cluster)} items to clusters")
    
    # Load natural belief target
    target_path = cfg['input']['natural_belief_target']
    print(f"Loading natural belief target: {target_path}")
    with open(target_path, 'r') as f:
        target_data = json.load(f)
    natural_belief_target = np.array(target_data['natural_belief_target'])
    
    # Load behaviors
    behaviors_path = cfg['input']['behaviors_file']
    separator = cfg['input'].get('separator', '\t')
    print(f"Loading behaviors: {behaviors_path}")
    df = pd.read_csv(behaviors_path, sep=separator)
    print(f"Loaded {len(df)} records, {df['userid'].nunique()} unique users")
    
    # Collect accepted items per user
    print("Processing user accepted items...")
    user_accepted_items = defaultdict(list)
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} records...")
        
        user_id = row['userid']
        impression_str = row['impression']
        
        if pd.isna(impression_str):
            continue
        
        tokens = str(impression_str).split()
        for token in tokens:
            if token.endswith('-1'):
                item_id = token[:-2]
                if item_id in item_to_cluster:
                    user_accepted_items[user_id].append(item_id)
    
    print(f"Collected accepted items for {len(user_accepted_items)} users")
    
    # Calculate beliefs
    print("Calculating beliefs...")
    user_beliefs_data = {}
    
    for user_id, accepted_items in user_accepted_items.items():
        cluster_counts = np.zeros(num_clusters, dtype=np.float32)
        
        for item_id in accepted_items:
            cluster = item_to_cluster[item_id]
            cluster_counts[cluster] += 1
        
        # belief = count / total_items (NOT normalized to sum=1)
        beliefs = cluster_counts / total_items
        pp1_distance = np.abs(beliefs - natural_belief_target).sum()
        
        user_beliefs_data[user_id] = {
            'beliefs': beliefs.tolist(),
            'pp1_distance': float(pp1_distance),
            'cluster_distances': natural_belief_target.tolist()
        }
    
    print(f"Calculated beliefs for {len(user_beliefs_data)} users")
    
    # Save JSON
    json_path = cfg['output']['json']
    ensure_dir(json_path)
    with open(json_path, 'w') as f:
        json.dump(user_beliefs_data, f, indent=2)
    print(f"✅ Saved: {json_path}")
    
    # Save PKL
    pkl_path = cfg['output']['pkl']
    ensure_dir(pkl_path)
    pkl_data = {uid: data['beliefs'] for uid, data in user_beliefs_data.items()}
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)
    print(f"✅ Saved: {pkl_path}")
    
    # Summary
    all_beliefs = np.array([data['beliefs'] for data in user_beliefs_data.values()])
    print(f"\nSummary:")
    print(f"  Mean sum of beliefs: {np.sum(all_beliefs, axis=1).mean():.6f}")
    print(f"  Mean per cluster: {np.mean(all_beliefs, axis=0)}")


# ============================================
# 3. Cluster Embeddings Generation
# ============================================
def generate_cluster_embeddings(config):
    """Generate cluster_embeddings.pkl by averaging item embeddings."""
    print("\n" + "="*60)
    print("3. Generating Cluster Embeddings")
    print("="*60)
    
    cfg = config['cluster_embeddings']
    num_clusters = cfg['options']['expected_clusters']
    
    # Load item embeddings
    item_emb_path = cfg['input']['item_embeddings']
    print(f"Loading item embeddings: {item_emb_path}")
    with open(item_emb_path, 'rb') as f:
        item_embeddings = pickle.load(f)
    print(f"Loaded {len(item_embeddings)} item embeddings")
    
    # Load cluster manifest
    manifest_path = cfg['input']['cluster_manifest']
    print(f"Loading cluster manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        cluster_data = json.load(f)
    cluster_to_items = cluster_data['cluster_to_items']
    
    # Load item token map
    token_map_path = cfg['input']['item_token_map']
    print(f"Loading item token map: {token_map_path}")
    with open(token_map_path, 'r') as f:
        item_token_map = json.load(f)
    
    # Calculate cluster embeddings
    cluster_embeddings = {}
    
    for cluster_id, item_list in cluster_to_items.items():
        print(f"\nProcessing cluster {cluster_id} ({len(item_list)} items)")
        
        embeddings = []
        for item_id in item_list:
            if item_id in item_token_map:
                token = str(item_token_map[item_id])
                if token in item_embeddings:
                    emb = item_embeddings[token]
                    if isinstance(emb, list):
                        emb = np.array(emb)
                    embeddings.append(emb)
        
        if embeddings:
            cluster_embedding = np.mean(embeddings, axis=0)
            cluster_embeddings[cluster_id] = cluster_embedding
            print(f"  Used {len(embeddings)}/{len(item_list)} items")
            print(f"  Embedding norm: {np.linalg.norm(cluster_embedding):.4f}")
        else:
            print(f"  ⚠️ No valid embeddings found!")
    
    # Save PKL
    pkl_path = cfg['output']['pkl']
    ensure_dir(pkl_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(cluster_embeddings, f)
    print(f"\n✅ Saved: {pkl_path}")
    
    # Save NPY
    if len(cluster_embeddings) == num_clusters:
        npy_path = cfg['output']['npy']
        ensure_dir(npy_path)
        cluster_array = np.array([cluster_embeddings[str(i)] for i in range(num_clusters)])
        np.save(npy_path, cluster_array)
        print(f"✅ Saved: {npy_path}")
    
    # Save info JSON
    info_path = cfg['output']['info_json']
    ensure_dir(info_path)
    info_data = {}
    for cid, emb in cluster_embeddings.items():
        info_data[cid] = {
            'embedding_shape': list(emb.shape),
            'embedding_norm': float(np.linalg.norm(emb)),
            'first_5_values': emb[:5].tolist(),
            'last_5_values': emb[-5:].tolist()
        }
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    print(f"✅ Saved: {info_path}")


# ============================================
# 4. Cluster Node Distances with Tokens
# ============================================
def generate_cluster_node_distances_tokens(config):
    """Generate cluster_node_distances_tokens CSV by adding token column."""
    print("\n" + "="*60)
    print("4. Generating Cluster Node Distances with Tokens")
    print("="*60)
    
    cfg = config['cluster_node_distances_tokens']
    
    # Load cluster node distances CSV
    input_csv = cfg['input']['cluster_node_distances_csv']
    print(f"Loading cluster node distances: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")
    
    # Detect item column name (could be 'item' or 'node')
    item_col = 'item' if 'item' in df.columns else 'node'
    print(f"Item column: {item_col}")
    
    # Load item token map
    token_map_path = cfg['input']['item_token_map']
    print(f"Loading item token map: {token_map_path}")
    with open(token_map_path, 'r') as f:
        item_token_map = json.load(f)
    
    # Add token column
    df['token'] = df[item_col].map(item_token_map)
    
    # Check for missing tokens
    missing = df['token'].isna().sum()
    if missing > 0:
        print(f"⚠️ {missing} items have no token mapping")
    
    # Reorder columns: cluster, item, token, distance_to_center
    if item_col == 'node':
        df = df.rename(columns={'node': 'item'})
    cols = ['cluster', 'item', 'token', 'distance_to_center']
    df = df[cols]
    
    # Save output
    output_csv = cfg['output']['csv']
    ensure_dir(output_csv)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")
    print(f"  Total rows: {len(df)}")


# ============================================
# 5. Cluster Distances PT (PyTorch tensor)
# ============================================
def generate_cluster_distances_pt(config):
    """Generate cluster_distances.pt from cluster_node_distances CSV."""
    print("\n" + "="*60)
    print("5. Generating Cluster Distances PT")
    print("="*60)
    
    try:
        import torch
    except ImportError:
        print("⚠️ PyTorch not installed, skipping cluster_distances.pt generation")
        return
    
    cfg = config['cluster_distances_pt']
    
    # Load cluster node distances CSV
    input_csv = cfg['input']['cluster_node_distances_csv']
    print(f"Loading cluster node distances: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")
    
    # Load item token map
    token_map_path = cfg['input']['item_token_map']
    print(f"Loading item token map: {token_map_path}")
    with open(token_map_path, 'r') as f:
        item_token_map = json.load(f)
    
    # Detect item column name
    item_col = 'item' if 'item' in df.columns else 'node'
    
    # Map item to token and get distances
    df['token'] = df[item_col].map(item_token_map)
    df = df.dropna(subset=['token'])
    df['token'] = df['token'].astype(int)
    
    # Create tensor: index by token, value is distance
    max_token = df['token'].max() + 1
    distances = torch.zeros(max_token, dtype=torch.float32)
    
    for _, row in df.iterrows():
        token = int(row['token'])
        distance = float(row['distance_to_center'])
        distances[token] = distance
    
    # Save output
    output_pt = cfg['output']['pt']
    ensure_dir(output_pt)
    torch.save(distances, output_pt)
    print(f"✅ Saved: {output_pt}")
    print(f"  Shape: {distances.shape}")
    print(f"  Max distance: {distances.max().item()}")


# ============================================
# 6. Test Set CSV
# ============================================
def generate_test_set(config):
    """Generate test_set.csv from behaviors data."""
    print("\n" + "="*60)
    print("6. Generating Test Set")
    print("="*60)
    
    cfg = config['test_set']
    
    # Load behaviors
    behaviors_path = cfg['input']['behaviors_file']
    separator = cfg['input'].get('separator', '\t')
    print(f"Loading behaviors: {behaviors_path}")
    df = pd.read_csv(behaviors_path, sep=separator)
    print(f"Loaded {len(df)} records")
    
    # Load user token map
    user_token_map_path = cfg['input']['user_token_map']
    print(f"Loading user token map: {user_token_map_path}")
    with open(user_token_map_path, 'r') as f:
        user_token_map = json.load(f)
    
    # Load item token map
    item_token_map_path = cfg['input']['item_token_map']
    print(f"Loading item token map: {item_token_map_path}")
    with open(item_token_map_path, 'r') as f:
        item_token_map = json.load(f)
    
    # Extract user-item pairs from impressions (accepted items only)
    test_pairs = []
    
    for _, row in df.iterrows():
        user_id = row['userid']
        impression_str = row['impression']
        
        if pd.isna(impression_str):
            continue
        
        # Get user token
        if user_id not in user_token_map:
            continue
        user_token = user_token_map[user_id]
        
        # Parse impressions
        tokens = str(impression_str).split()
        for token in tokens:
            if token.endswith('-1'):  # Accepted items
                item_id = token[:-2]
                if item_id in item_token_map:
                    item_token = item_token_map[item_id]
                    test_pairs.append({'user_id': user_token, 'item_id': item_token})
    
    # Create DataFrame and save
    test_df = pd.DataFrame(test_pairs)
    
    output_csv = cfg['output']['csv']
    ensure_dir(output_csv)
    test_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")
    print(f"  Total pairs: {len(test_df)}")
    print(f"  Unique users: {test_df['user_id'].nunique()}")
    print(f"  Unique items: {test_df['item_id'].nunique()}")
    
    # Save test set info
    info_path = cfg['output']['info_json']
    ensure_dir(info_path)
    info_data = {
        'user_ids': test_df['user_id'].unique().tolist(),
        'total_pairs': len(test_df),
        'unique_users': test_df['user_id'].nunique(),
        'unique_items': test_df['item_id'].nunique()
    }
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    print(f"✅ Saved: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified dataset preprocessing")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument("--only", choices=[
        'natural_target', 'user_beliefs', 'cluster_embeddings', 
        'node_distances_tokens', 'cluster_distances_pt', 'test_set'
    ], help="Generate only specific component")
    parser.add_argument("--topk", type=int, help="Override topk value from config")
    
    args = parser.parse_args()
    
    config = load_config(args.config, args.topk)
    print(f"Dataset: {config['dataset']}, Topk: {config['topk']}")
    
    if args.only:
        if args.only == 'natural_target':
            generate_natural_target(config)
        elif args.only == 'user_beliefs':
            generate_user_beliefs(config)
        elif args.only == 'cluster_embeddings':
            generate_cluster_embeddings(config)
        elif args.only == 'node_distances_tokens':
            generate_cluster_node_distances_tokens(config)
        elif args.only == 'cluster_distances_pt':
            generate_cluster_distances_pt(config)
        elif args.only == 'test_set':
            generate_test_set(config)
    else:
        # Generate all in order
        generate_natural_target(config)
        generate_user_beliefs(config)
        generate_cluster_embeddings(config)
        generate_cluster_node_distances_tokens(config)
        generate_cluster_distances_pt(config)
        generate_test_set(config)
    
    print("\n" + "="*60)
    print("✅ Preprocessing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
