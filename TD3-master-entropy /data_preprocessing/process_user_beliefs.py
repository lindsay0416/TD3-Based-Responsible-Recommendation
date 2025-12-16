#!/usr/bin/env python3
"""
Single-file user belief processing pipeline.
Reads configuration from belief_config.yaml and generates user_average_beliefs.json

Usage:
    python data_preprocessing/process_user_beliefs.py
    
Or with custom config:
    python data_preprocessing/process_user_beliefs.py --config custom_config.yaml
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import pickle
import ast
from collections import OrderedDict, defaultdict

class UserBeliefProcessor:
    def __init__(self, config_path="data_preprocessing/belief_config.yaml"):
        """Initialize processor with configuration."""
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
        self.show_samples = self.config.get('logging', {}).get('show_samples', 5)
    
    def log(self, message):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_optimal_beliefs(self):
        """Load optimal beliefs for constrained normalization."""
        target_file = self.config['input_files']['natural_belief_target']
        
        try:
            with open(target_file, 'r') as f:
                target_data = json.load(f)
            optimal_beliefs = np.array(target_data['natural_belief_target'], dtype=np.float32)
            self.log(f"✅ Loaded optimal beliefs: {optimal_beliefs}")
            return optimal_beliefs
        except Exception as e:
            self.log(f"⚠️  Could not load optimal beliefs from {target_file}: {e}")
            fallback = self.config['processing']['fallback_uniform_beliefs']
            optimal_beliefs = np.array(fallback, dtype=np.float32)
            self.log(f"Using fallback uniform beliefs: {optimal_beliefs}")
            return optimal_beliefs
    
    def load_clusters_from_manifest(self):
        """Load cluster information from manifest file."""
        manifest_path = self.config['input_files']['cluster_manifest']
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if "cluster_to_items" not in manifest:
            raise KeyError("Manifest missing 'cluster_to_items' key")

        cluster_to_items = manifest["cluster_to_items"]
        clusters = {int(cid): set(items) for cid, items in cluster_to_items.items()}

        messages = set()
        message_cluster_map = {}
        for cid, items in clusters.items():
            for it in items:
                messages.add(it)
                message_cluster_map[it] = cid

        cluster_ids = sorted(clusters.keys())
        self.log(f"✅ Loaded {len(cluster_ids)} clusters, {len(messages)} items from manifest")
        return manifest, clusters, messages, message_cluster_map, cluster_ids
    
    def split_impressions(self, df, valid_ids):
        """Split impression column into clicked and non-clicked items."""
        def process(imp_str):
            if pd.isna(imp_str):
                return pd.Series([[], []])
            tokens = str(imp_str).split()
            filtered = [t for t in tokens if t.split('-')[0] in valid_ids]
            group_0 = [t.split("-0")[0] for t in filtered if t.endswith("-0")]
            group_1 = [t.split("-1")[0] for t in filtered if t.endswith("-1")]
            return pd.Series([group_0, group_1])

        df[["impression_0", "impression_1"]] = df["impression"].apply(process)
        before = len(df)
        df = df[~((df["impression_0"].str.len() == 0) & (df["impression_1"].str.len() == 0))].reset_index(drop=True)
        after = len(df)
        if after < before:
            self.log(f"Dropped {before - after} rows with no valid impressions")
        return df
    
    def layered_constrained_normalize_beliefs(self, smoothed, optimal_beliefs):
        """Apply constrained normalization to beliefs."""
        if len(smoothed) == 0:
            return smoothed
        
        smoothed = np.array(smoothed)
        optimal_beliefs = np.array(optimal_beliefs)
        
        base_ratio = self.config['processing']['base_ratio']
        max_ratio = self.config['processing']['max_ratio']
        
        # 1. Calculate relative preference strength [0, 1]
        if smoothed.std() > 1e-8:
            relative_prefs = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        else:
            relative_prefs = np.ones_like(smoothed) * 0.5
        
        # 2. Set base interest and maximum interest for each cluster
        base_interest = optimal_beliefs * base_ratio
        max_interest = optimal_beliefs * max_ratio
        
        # 3. Linear interpolation: base + preference * (max - base)
        adjusted_beliefs = base_interest + relative_prefs * (max_interest - base_interest)
        
        # 4. Normalize to sum to 1
        normalized_beliefs = adjusted_beliefs / adjusted_beliefs.sum()
        
        return normalized_beliefs
    
    def compute_user_beliefs(self):
        """Compute individual user beliefs and save to TSV."""
        self.log("🔄 Step 1: Computing individual user beliefs...")
        
        # Load configuration parameters
        positive_weight = self.config['processing']['positive_weight']
        negative_weight = self.config['processing']['negative_weight']
        alpha_smoothing = self.config['processing']['alpha_smoothing']
        use_constrained = self.config['processing']['use_constrained_normalization']
        
        # Load optimal beliefs if using constrained normalization
        if use_constrained:
            optimal_beliefs = self.load_optimal_beliefs()
        
        # Load clusters and manifest
        manifest, clusters, messages, message_cluster_map, cluster_ids = self.load_clusters_from_manifest()
        id_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
        num_clusters = len(cluster_ids)
        
        # Load behaviors CSV
        behaviors_csv = self.config['input_files']['behaviors_csv']
        if not os.path.exists(behaviors_csv):
            raise FileNotFoundError(f"Behaviors CSV not found: {behaviors_csv}")
        
        users = pd.read_csv(behaviors_csv)
        if "impression" not in users.columns:
            raise KeyError("Input CSV must contain an 'impression' column")
        
        users = self.split_impressions(users, messages)
        self.log(f"✅ Loaded {len(users)} user records from {behaviors_csv}")
        
        # Compute beliefs for each user
        user_beliefs = np.zeros((len(users), num_clusters), dtype=np.float64)
        belief_list = []
        
        for idx, row in users[["impression_1", "impression_0"]].iterrows():
            cluster_scores = np.zeros(num_clusters, dtype=np.float64)

            # Positive evidence from clicks
            for item_id in row["impression_1"]:
                cid = message_cluster_map.get(item_id)
                if cid is not None:
                    cluster_scores[id_to_idx[cid]] += positive_weight

            # Negative evidence from non-click exposures
            for item_id in row["impression_0"]:
                cid = message_cluster_map.get(item_id)
                if cid is not None:
                    cluster_scores[id_to_idx[cid]] += negative_weight

            # Apply smoothing
            smoothed = cluster_scores + alpha_smoothing
            np.clip(smoothed, 1e-6, None, out=smoothed)
            
            # Apply normalization
            if use_constrained:
                belief = self.layered_constrained_normalize_beliefs(smoothed, optimal_beliefs)
            else:
                # Simple normalization to sum=1
                belief = smoothed / smoothed.sum()

            user_beliefs[idx] = belief
            belief_list.append(belief.tolist())

        users["belief"] = belief_list
        
        # Save TSV file
        output_tsv = self.config['output_files']['user_beliefs_tsv']
        os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
        users.to_csv(output_tsv, sep="\t", index=False)
        self.log(f"✅ Saved individual user beliefs to {output_tsv}")
        
        return users
    
    def calculate_average_beliefs(self, df):
        """Calculate average beliefs per user from TSV data."""
        self.log("🔄 Step 2: Calculating average beliefs per user...")
        
        # Parse belief column
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
                if failed_parses <= 3:  # Show first 3 errors
                    self.log(f"⚠️  Error parsing belief for user {row['userid']}: {e}")
        
        self.log(f"✅ Successfully parsed {len(parsed_data)} records, failed: {failed_parses}")
        
        # Group by userid and calculate average
        user_beliefs_dict = defaultdict(list)
        
        for item in parsed_data:
            user_beliefs_dict[item['userid']].append(item['belief_vector'])
        
        # Calculate averages
        user_avg_beliefs = {}
        for user_id, belief_list in user_beliefs_dict.items():
            belief_array = np.array(belief_list)
            avg_belief = np.mean(belief_array, axis=0)
            user_avg_beliefs[user_id] = avg_belief
        
        self.log(f"✅ Calculated average beliefs for {len(user_avg_beliefs)} users")
        return user_avg_beliefs
    
    def save_results(self, user_avg_beliefs):
        """Save results as both pickle and JSON files."""
        self.log("🔄 Step 3: Saving results...")
        
        # Create output directory
        pkl_path = self.config['output_files']['user_average_beliefs_pkl']
        json_path = self.config['output_files']['user_average_beliefs_json']
        
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # Save pickle file
        with open(pkl_path, 'wb') as f:
            pickle.dump(user_avg_beliefs, f)
        self.log(f"✅ Saved pickle file: {pkl_path} ({os.path.getsize(pkl_path) / 1024:.2f} KB)")
        
        # Save JSON file
        json_data = {}
        for user_id, belief_vector in user_avg_beliefs.items():
            json_data[user_id] = belief_vector.tolist()
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.log(f"✅ Saved JSON file: {json_path} ({os.path.getsize(json_path) / 1024:.2f} KB)")
        
        # Display sample results
        if self.show_samples > 0:
            self.log(f"\n📊 Sample user average beliefs (first {self.show_samples}):")
            sample_users = list(user_avg_beliefs.keys())[:self.show_samples]
            for user_id in sample_users:
                belief_vector = user_avg_beliefs[user_id]
                self.log(f"  User {user_id}: {belief_vector}")
        
        # Summary statistics
        all_beliefs = np.array(list(user_avg_beliefs.values()))
        self.log(f"\n📈 Summary Statistics:")
        self.log(f"  Total users: {len(user_avg_beliefs)}")
        self.log(f"  Belief dimensions: {all_beliefs.shape[1]}")
        self.log(f"  Mean per dimension: {np.mean(all_beliefs, axis=0)}")
        self.log(f"  Std per dimension: {np.std(all_beliefs, axis=0)}")
    
    def process(self):
        """Run the complete processing pipeline."""
        print("🚀 Starting User Belief Processing Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Compute individual user beliefs
            df = self.compute_user_beliefs()
            
            # Step 2: Calculate average beliefs per user
            user_avg_beliefs = self.calculate_average_beliefs(df)
            
            # Step 3: Save results
            self.save_results(user_avg_beliefs)
            
            print("=" * 60)
            print("✅ Pipeline completed successfully!")
            print(f"📁 Output files:")
            print(f"   - {self.config['output_files']['user_average_beliefs_pkl']}")
            print(f"   - {self.config['output_files']['user_average_beliefs_json']}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Process user beliefs from configuration")
    parser.add_argument("--config", 
                       default="data_preprocessing/belief_config.yaml",
                       help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    processor = UserBeliefProcessor(args.config)
    processor.process()

if __name__ == "__main__":
    main()