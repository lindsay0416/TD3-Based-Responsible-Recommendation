#!/usr/bin/env python3
"""
Extract unique user_ids from test_set.csv
Reads dataset config from config.yaml
Output: processed_data/{dataset}/unique_test_users.json

Note: test_set.csv contains token IDs, we convert them back to original user IDs
"""

import pandas as pd
import json
import yaml
from pathlib import Path

def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load config
    config = load_config()
    dataset = config['experiment_params']['dataset']
    topk = config['experiment_params']['topk']
    
    print(f"Extracting unique user_ids for dataset: {dataset}, topk: {topk}")
    
    # Read test_set.csv
    test_set_path = f'embeddings/{dataset}/Topk{topk}/test_set.csv'
    print(f"Reading from: {test_set_path}")
    
    df = pd.read_csv(test_set_path)
    
    print(f"Total rows in test_set.csv: {len(df)}")
    
    # Get unique token IDs from test set
    unique_token_ids = df['user_id'].unique().tolist()
    print(f"Unique token IDs in test set: {len(unique_token_ids)}")
    
    # Load user_token_map to convert token IDs back to original user IDs
    user_token_map_path = f'embeddings/{dataset}/Topk{topk}/user_token_map.json'
    print(f"Loading user token map from: {user_token_map_path}")
    
    with open(user_token_map_path, 'r') as f:
        user_token_map = json.load(f)  # {"U10000": 0, "U10003": 1, ...}
    
    # Create reverse mapping: token_id -> original_user_id
    token_to_user = {v: k for k, v in user_token_map.items()}
    
    # Convert token IDs to original user IDs
    unique_users = []
    for token_id in unique_token_ids:
        if token_id in token_to_user:
            unique_users.append(token_to_user[token_id])
        else:
            print(f"Warning: token_id {token_id} not found in user_token_map")
    
    print(f"Converted to {len(unique_users)} original user IDs")
    
    # Ensure output directory exists
    output_dir = Path(f'processed_data/{dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output = {
        "unique_user_ids": unique_users,
        "count": len(unique_users),
        "source": test_set_path
    }
    
    output_path = output_dir / 'unique_test_users.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to: {output_path}")
    print(f"Sample user_ids: {unique_users[:10]}")

if __name__ == "__main__":
    main()
