#!/usr/bin/env python3
"""
Analyze cluster distribution of recommended items in each round
"""

import json
import os
import glob
from collections import Counter
import numpy as np

def load_item_to_cluster_mapping():
    """Load item to cluster mapping"""
    import pickle
    
    # Load graph with clusters
    with open('embeddings/graph_with_clusters_K5_topk5.pkl', 'rb') as f:
        graph_data = pickle.load(f)
    
    item_to_cluster = {}
    for node, data in graph_data.nodes(data=True):
        if 'cluster' in data:
            item_to_cluster[str(node)] = data['cluster']
    
    return item_to_cluster

def analyze_round_recommendations(rec_file, item_to_cluster):
    """Analyze cluster distribution of recommendations in a single round"""
    with open(rec_file, 'r') as f:
        data = json.load(f)
    
    episode = data['episode']
    round_num = data['round']
    recommendations = data['recommendations']
    
    # Count cluster distribution of all recommended items
    all_recommended_clusters = []
    all_accepted_clusters = []
    all_rejected_clusters = []
    
    for user_rec in recommendations:
        # Recommended items
        for item_id in user_rec['recommended_items']:
            item_id_str = str(item_id)
            # Try different formats
            cluster = item_to_cluster.get(item_id_str)
            if cluster is None and not item_id_str.startswith('N'):
                cluster = item_to_cluster.get(f"N{item_id_str}")
            if cluster is None and item_id_str.startswith('N'):
                cluster = item_to_cluster.get(item_id_str[1:])
            
            if cluster is not None:
                all_recommended_clusters.append(cluster)
        
        # Accepted items
        for item_id in user_rec['accepted_items']:
            item_id_str = str(item_id)
            cluster = item_to_cluster.get(item_id_str)
            if cluster is None and not item_id_str.startswith('N'):
                cluster = item_to_cluster.get(f"N{item_id_str}")
            if cluster is None and item_id_str.startswith('N'):
                cluster = item_to_cluster.get(item_id_str[1:])
            
            if cluster is not None:
                all_accepted_clusters.append(cluster)
        
        # Rejected items
        for item_id in user_rec['rejected_items']:
            item_id_str = str(item_id)
            cluster = item_to_cluster.get(item_id_str)
            if cluster is None and not item_id_str.startswith('N'):
                cluster = item_to_cluster.get(f"N{item_id_str}")
            if cluster is None and item_id_str.startswith('N'):
                cluster = item_to_cluster.get(item_id_str[1:])
            
            if cluster is not None:
                all_rejected_clusters.append(cluster)
    
    # Calculate distribution
    recommended_dist = Counter(all_recommended_clusters)
    accepted_dist = Counter(all_accepted_clusters)
    rejected_dist = Counter(all_rejected_clusters)
    
    # Convert to percentages
    total_recommended = len(all_recommended_clusters)
    total_accepted = len(all_accepted_clusters)
    total_rejected = len(all_rejected_clusters)
    
    recommended_pct = {k: v/total_recommended*100 for k, v in recommended_dist.items()} if total_recommended > 0 else {}
    accepted_pct = {k: v/total_accepted*100 for k, v in accepted_dist.items()} if total_accepted > 0 else {}
    rejected_pct = {k: v/total_rejected*100 for k, v in rejected_dist.items()} if total_rejected > 0 else {}
    
    return {
        'episode': episode,
        'round': round_num,
        'total_users': len(recommendations),
        'total_recommended': total_recommended,
        'total_accepted': total_accepted,
        'total_rejected': total_rejected,
        'recommended_distribution': recommended_pct,
        'accepted_distribution': accepted_pct,
        'rejected_distribution': rejected_pct,
        'recommended_counts': dict(recommended_dist),
        'accepted_counts': dict(accepted_dist),
        'rejected_counts': dict(rejected_dist)
    }

def analyze_round_data(key, round_data, item_to_cluster):
    """Analyze data for a single round (from dictionary)"""
    episode = round_data['episode']
    round_num = round_data['round']
    recommendations = round_data['recommendations']
    
    # Count cluster distribution of all recommended items
    all_recommended_clusters = []
    all_accepted_clusters = []
    all_rejected_clusters = []
    
    for user_rec in recommendations:
        # Recommended items
        for item_id in user_rec['recommended_items']:
            item_id_str = str(item_id)
            # Try different formats
            cluster = item_to_cluster.get(item_id_str)
            if cluster is None and not item_id_str.startswith('N'):
                cluster = item_to_cluster.get(f"N{item_id_str}")
            if cluster is None and item_id_str.startswith('N'):
                cluster = item_to_cluster.get(item_id_str[1:])
            
            if cluster is not None:
                all_recommended_clusters.append(cluster)
        
        # Accepted items
        for item_id in user_rec['accepted_items']:
            item_id_str = str(item_id)
            cluster = item_to_cluster.get(item_id_str)
            if cluster is None and not item_id_str.startswith('N'):
                cluster = item_to_cluster.get(f"N{item_id_str}")
            if cluster is None and item_id_str.startswith('N'):
                cluster = item_to_cluster.get(item_id_str[1:])
            
            if cluster is not None:
                all_accepted_clusters.append(cluster)
        
        # Rejected items
        for item_id in user_rec['rejected_items']:
            item_id_str = str(item_id)
            cluster = item_to_cluster.get(item_id_str)
            if cluster is None and not item_id_str.startswith('N'):
                cluster = item_to_cluster.get(f"N{item_id_str}")
            if cluster is None and item_id_str.startswith('N'):
                cluster = item_to_cluster.get(item_id_str[1:])
            
            if cluster is not None:
                all_rejected_clusters.append(cluster)
    
    # Calculate distribution
    recommended_dist = Counter(all_recommended_clusters)
    accepted_dist = Counter(all_accepted_clusters)
    rejected_dist = Counter(all_rejected_clusters)
    
    # Convert to percentages
    total_recommended = len(all_recommended_clusters)
    total_accepted = len(all_accepted_clusters)
    total_rejected = len(all_rejected_clusters)
    
    recommended_pct = {k: v/total_recommended*100 for k, v in recommended_dist.items()} if total_recommended > 0 else {}
    accepted_pct = {k: v/total_accepted*100 for k, v in accepted_dist.items()} if total_accepted > 0 else {}
    rejected_pct = {k: v/total_rejected*100 for k, v in rejected_dist.items()} if total_rejected > 0 else {}
    
    return {
        'key': key,
        'episode': episode,
        'round': round_num,
        'total_users': len(recommendations),
        'total_recommended': total_recommended,
        'total_accepted': total_accepted,
        'total_rejected': total_rejected,
        'recommended_distribution': recommended_pct,
        'accepted_distribution': accepted_pct,
        'rejected_distribution': rejected_pct,
        'recommended_counts': dict(recommended_dist),
        'accepted_counts': dict(accepted_dist),
        'rejected_counts': dict(rejected_dist)
    }

def main():
    """Main function"""
    print("Loading item-to-cluster mapping...")
    item_to_cluster = load_item_to_cluster_mapping()
    print(f"Loaded {len(item_to_cluster)} items")
    
    # Load recommendation data files
    rec_file = 'results/all_recommendations.json'
    if not os.path.exists(rec_file):
        print(f"Recommendations file not found: {rec_file}")
        return
    
    print(f"Loading recommendations from {rec_file}...")
    with open(rec_file, 'r') as f:
        all_rec_data = json.load(f)
    
    print(f"Found {len(all_rec_data)} episode-round combinations")
    
    if len(all_rec_data) == 0:
        print("No recommendation data to analyze")
        return
    
    # Analyze each episode-round
    all_analyses = []
    for key in sorted(all_rec_data.keys()):
        print(f"Analyzing {key}...")
        analysis = analyze_round_data(key, all_rec_data[key], item_to_cluster)
        all_analyses.append(analysis)
    
    # Save analysis results
    output_file = 'results/cluster_distribution_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    
    print(f"\nAnalysis saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("CLUSTER DISTRIBUTION SUMMARY")
    print("="*80)
    
    # Show comparison between first and last rounds
    if len(all_analyses) >= 2:
        first = all_analyses[0]
        last = all_analyses[-1]
        
        print(f"\nFirst Round (Episode {first['episode']}, Round {first['round']}):")
        print("  Recommended distribution:")
        for cluster in sorted(first['recommended_distribution'].keys()):
            pct = first['recommended_distribution'][cluster]
            count = first['recommended_counts'][cluster]
            print(f"    Cluster {cluster}: {pct:5.2f}% ({count} items)")
        
        print(f"\nLast Round (Episode {last['episode']}, Round {last['round']}):")
        print("  Recommended distribution:")
        for cluster in sorted(last['recommended_distribution'].keys()):
            pct = last['recommended_distribution'][cluster]
            count = last['recommended_counts'][cluster]
            print(f"    Cluster {cluster}: {pct:5.2f}% ({count} items)")
        
        print(f"\nAcceptance rates by cluster (Last Round):")
        for cluster in sorted(last['recommended_counts'].keys()):
            recommended = last['recommended_counts'].get(cluster, 0)
            accepted = last['accepted_counts'].get(cluster, 0)
            if recommended > 0:
                acceptance_rate = accepted / recommended * 100
                print(f"    Cluster {cluster}: {acceptance_rate:5.2f}% ({accepted}/{recommended})")

if __name__ == "__main__":
    main()
