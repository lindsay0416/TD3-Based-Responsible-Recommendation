#!/usr/bin/env python3
"""
Calculate average beliefs for each user from user_beliefs_K5_topk5.tsv
Save as pickle file for fast loading in the RL environment.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from collections import defaultdict

def load_user_beliefs(file_path):
    """Load user beliefs from TSV file."""
    print(f"Loading user beliefs from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    return df

def calculate_user_average_beliefs(df):
    """Calculate average belief vector for each user."""
    print("Calculating average beliefs per user...")
    
    # Parse belief column which contains list strings
    print("Parsing belief vectors...")
    
    import ast
    
    # Parse belief strings and collect data
    parsed_data = []
    failed_parses = 0
    
    for idx, row in df.iterrows():
        try:
            belief_str = row['belief']
            if isinstance(belief_str, str):
                belief_vector = ast.literal_eval(belief_str)
            else:
                belief_vector = belief_str
            
            parsed_data.append({
                'userid': row['userid'],
                'belief_vector': belief_vector
            })
        except Exception as e:
            failed_parses += 1
            if failed_parses <= 5:  # Show first 5 errors
                print(f"Error parsing belief for user {row['userid']}: {e}")
    
    print(f"Successfully parsed {len(parsed_data)} records, failed: {failed_parses}")
    
    # Group by userid and calculate average belief vector
    user_beliefs_dict = defaultdict(list)
    
    for item in parsed_data:
        user_beliefs_dict[item['userid']].append(item['belief_vector'])
    
    # Calculate averages
    user_avg_beliefs = {}
    for user_id, belief_list in user_beliefs_dict.items():
        # Convert to numpy array and calculate mean
        belief_array = np.array(belief_list)
        avg_belief = np.mean(belief_array, axis=0)
        user_avg_beliefs[user_id] = avg_belief
    
    print(f"Calculated average beliefs for {len(user_avg_beliefs)} users")
    if user_avg_beliefs:
        sample_belief = next(iter(user_avg_beliefs.values()))
        print(f"Belief vector dimension: {len(sample_belief)}")
    
    return user_avg_beliefs

def save_user_beliefs(user_beliefs, output_path):
    """Save user beliefs as pickle file and JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as pickle for fast loading
    with open(output_path, 'wb') as f:
        pickle.dump(user_beliefs, f)
    
    print(f"Saved user average beliefs to {output_path}")
    print(f"Pickle file size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    # Save as JSON for human readability
    json_path = output_path.replace('.pkl', '.json')
    json_data = {}
    
    for user_id, belief_vector in user_beliefs.items():
        # Convert numpy array to list for JSON serialization
        json_data[user_id] = belief_vector.tolist()
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved human-readable version to {json_path}")
    print(f"JSON file size: {os.path.getsize(json_path) / 1024:.2f} KB")

def main():
    # File paths
    input_file = "embeddings/user_beliefs_K5_topk5.tsv"
    output_file = "processed_data/user_average_beliefs.pkl"
    
    # Load data
    df = load_user_beliefs(input_file)
    
    # Calculate average beliefs
    user_avg_beliefs = calculate_user_average_beliefs(df)
    
    # Save results
    save_user_beliefs(user_avg_beliefs, output_file)
    
    # Display sample results
    print("\nSample user average beliefs:")
    sample_users = list(user_avg_beliefs.keys())[:5]
    for user_id in sample_users:
        belief_vector = user_avg_beliefs[user_id]
        print(f"User {user_id}: {belief_vector}")
    
    # Summary statistics
    print(f"\nSummary statistics:")
    all_beliefs = np.array(list(user_avg_beliefs.values()))
    print(f"Shape: {all_beliefs.shape}")
    print(f"Mean per dimension: {np.mean(all_beliefs, axis=0)}")
    print(f"Std per dimension: {np.std(all_beliefs, axis=0)}")
    print(f"Min per dimension: {np.min(all_beliefs, axis=0)}")
    print(f"Max per dimension: {np.max(all_beliefs, axis=0)}")

if __name__ == "__main__":
    main()