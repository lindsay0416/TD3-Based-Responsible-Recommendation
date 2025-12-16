import pandas as pd
import numpy as np
import json
from collections import defaultdict

def recalculate_user_beliefs_raw():
    """
    Recalculate user beliefs using raw click counts (no normalization)
    This keeps beliefs in their natural form - if users clicked all items proportionally,
    they would match the natural distribution
    """
    print("=== RECALCULATING USER BELIEFS (RAW COUNTS, NO NORMALIZATION) ===")
    
    # Load cluster mapping
    with open('embeddings/cluster_matrix_manifest_K5_topk5.json', 'r') as f:
        cluster_data = json.load(f)
    
    item_to_cluster = {}
    cluster_to_items = cluster_data['cluster_to_items']
    
    for cluster_id, items in cluster_to_items.items():
        for item in items:
            item_to_cluster[item] = int(cluster_id)
    
    print(f"Loaded cluster mapping for {len(item_to_cluster)} items")
    
    # Read the existing TSV file
    df = pd.read_csv('embeddings/user_beliefs_K5_topk5.tsv', sep='\t')
    print(f"Loaded {len(df)} records from user_beliefs_K5_topk5.tsv")
    
    # Calculate raw click counts for each record
    new_beliefs = []
    
    for idx, row in df.iterrows():
        impression_str = row['impression']
        
        # Initialize raw click counts for 5 clusters
        raw_click_counts = np.zeros(5, dtype=int)
        
        if pd.notna(impression_str):
            # Parse impression string
            impression_items = impression_str.strip().split()
            
            for item_interaction in impression_items:
                if '-' in item_interaction:
                    item_id, interaction_flag = item_interaction.rsplit('-', 1)
                    
                    # Count ONLY clicks (-1), keep as raw counts
                    if interaction_flag == '1' and item_id in item_to_cluster:
                        cluster_id = item_to_cluster[item_id]
                        raw_click_counts[cluster_id] += 1
        
        # Store as list (no normalization!)
        new_beliefs.append(raw_click_counts.tolist())
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} records...")
    
    # Calculate beliefs as percentage of items clicked per cluster
    percentage_beliefs = []
    cluster_sizes = [len(cluster_to_items[str(i)]) for i in range(5)]
    
    for raw_counts in new_beliefs:
        # Calculate percentage of items clicked in each cluster
        percentages = []
        for i in range(5):
            clicks_in_cluster = raw_counts[i]
            total_items_in_cluster = cluster_sizes[i]
            percentage = clicks_in_cluster / total_items_in_cluster if total_items_in_cluster > 0 else 0
            percentages.append(percentage)
        
        percentage_beliefs.append(percentages)
    
    # Update the belief column with percentage beliefs
    df['belief'] = percentage_beliefs
    
    # Save back to the same file (overwrite)
    df.to_csv('embeddings/user_beliefs_K5_topk5.tsv', sep='\t', index=False)
    print(f"✅ Updated user_beliefs_K5_topk5.tsv with percentage-based beliefs")
    
    # Analyze both raw and percentage beliefs
    analyze_raw_beliefs(new_beliefs, cluster_to_items)
    analyze_percentage_beliefs(percentage_beliefs, cluster_to_items)
    
    return df

def analyze_raw_beliefs(beliefs_list, cluster_to_items):
    """
    Analyze the raw belief counts
    """
    print(f"\n=== RAW BELIEF ANALYSIS ===")
    
    # Convert to numpy array for analysis
    beliefs_array = np.array(beliefs_list)
    
    # Calculate statistics
    total_clicks_per_user = np.sum(beliefs_array, axis=1)
    total_clicks_per_cluster = np.sum(beliefs_array, axis=0)
    
    print(f"Total users: {len(beliefs_array)}")
    print(f"Users with clicks: {np.sum(total_clicks_per_user > 0)}")
    print(f"Average clicks per user: {np.mean(total_clicks_per_user):.2f}")
    print(f"Median clicks per user: {np.median(total_clicks_per_user):.2f}")
    print(f"Max clicks per user: {np.max(total_clicks_per_user)}")
    
    print(f"\nClicks per cluster (raw counts):")
    for i in range(5):
        cluster_size = len(cluster_to_items[str(i)])
        clicks = total_clicks_per_cluster[i]
        click_rate = clicks / cluster_size if cluster_size > 0 else 0
        print(f"  Cluster {i}: {clicks:,} clicks ({cluster_size:,} items, {click_rate:.4f} clicks/item)")
    
    # Calculate natural distribution for comparison
    cluster_sizes = [len(cluster_to_items[str(i)]) for i in range(5)]
    total_items = sum(cluster_sizes)
    natural_proportions = [size / total_items for size in cluster_sizes]
    
    # Calculate actual click proportions
    total_clicks = np.sum(total_clicks_per_cluster)
    if total_clicks > 0:
        actual_proportions = total_clicks_per_cluster / total_clicks
    else:
        actual_proportions = np.zeros(5)
    
    print(f"\nComparison with natural distribution:")
    print(f"Cluster    Natural%    Actual%    Ratio (Actual/Natural)")
    for i in range(5):
        natural_pct = natural_proportions[i] * 100
        actual_pct = actual_proportions[i] * 100
        ratio = actual_proportions[i] / natural_proportions[i] if natural_proportions[i] > 0 else 0
        print(f"   {i}        {natural_pct:6.1f}%     {actual_pct:6.1f}%     {ratio:6.2f}x")
    
    # Show sample users
    print(f"\nSample raw beliefs (first 10 users with clicks):")
    users_with_clicks = beliefs_array[total_clicks_per_user > 0][:10]
    for i, belief in enumerate(users_with_clicks):
        total = np.sum(belief)
        dominant_cluster = np.argmax(belief) if total > 0 else -1
        print(f"  User {i}: {belief} (total: {total}, dominant: cluster {dominant_cluster})")

def analyze_percentage_beliefs(percentage_beliefs, cluster_to_items):
    """
    Analyze the percentage-based beliefs
    """
    print(f"\n=== PERCENTAGE BELIEF ANALYSIS ===")
    
    # Convert to numpy array
    beliefs_array = np.array(percentage_beliefs)
    
    # Calculate average percentage beliefs
    avg_beliefs = np.mean(beliefs_array, axis=0)
    
    print(f"Average percentage beliefs (% of items clicked per cluster): {avg_beliefs}")
    
    # For comparison, if users clicked items proportionally to natural distribution
    cluster_sizes = [len(cluster_to_items[str(i)]) for i in range(5)]
    total_items = sum(cluster_sizes)
    natural_proportions = [size / total_items for size in cluster_sizes]
    
    # If users clicked the same percentage of items in each cluster proportionally
    # they would have the same percentage across all clusters
    total_clicks = sum([sum(belief) * cluster_sizes[i] for i, belief in enumerate(zip(*percentage_beliefs))])
    avg_click_rate = total_clicks / (len(percentage_beliefs) * total_items) if len(percentage_beliefs) > 0 else 0
    
    print(f"Natural proportions: {natural_proportions}")
    print(f"Average click rate across all items: {avg_click_rate:.6f}")
    
    print(f"\nComparison (percentage of items clicked):")
    print(f"Cluster    Items    Natural%    User%Clicked    Expected%Clicked")
    for i in range(5):
        items = cluster_sizes[i]
        natural_pct = natural_proportions[i] * 100
        user_pct_clicked = avg_beliefs[i] * 100
        expected_pct_clicked = avg_click_rate * 100
        print(f"   {i}      {items:5d}     {natural_pct:6.1f}%      {user_pct_clicked:8.4f}%        {expected_pct_clicked:8.4f}%")
    
    # Show sample percentage beliefs
    print(f"\nSample percentage beliefs (first 10 users):")
    for i, belief in enumerate(beliefs_array[:10]):
        max_pct = np.max(belief) if len(belief) > 0 else 0
        max_cluster = np.argmax(belief) if max_pct > 0 else -1
        print(f"  User {i}: {[f'{x:.6f}' for x in belief]} (max: {max_pct:.6f} in cluster {max_cluster})")

def create_natural_belief_target_raw():
    """
    Create the natural belief target as raw counts (for comparison)
    """
    print(f"\n=== CREATING NATURAL BELIEF TARGET (RAW COUNTS) ===")
    
    # Load cluster sizes
    with open('embeddings/cluster_matrix_manifest_K5_topk5.json', 'r') as f:
        cluster_data = json.load(f)
    
    cluster_sizes = [len(cluster_data['cluster_to_items'][str(i)]) for i in range(5)]
    
    print(f"Natural belief target (raw item counts): {cluster_sizes}")
    print(f"If a user clicked items proportionally to availability:")
    
    # Example: if user made 100 clicks proportionally
    example_clicks = 100
    proportional_clicks = []
    total_items = sum(cluster_sizes)
    
    for size in cluster_sizes:
        prop_clicks = int(example_clicks * size / total_items)
        proportional_clicks.append(prop_clicks)
    
    print(f"  With {example_clicks} total clicks: {proportional_clicks}")
    
    # Save natural target as raw counts
    natural_target_raw = {
        'cluster_sizes': cluster_sizes,
        'description': 'Natural belief target as raw item counts (no normalization)',
        'example_proportional_clicks_100': proportional_clicks
    }
    
    with open('processed_data/natural_belief_target_raw.json', 'w') as f:
        json.dump(natural_target_raw, f, indent=2)
    
    print(f"✅ Saved natural_belief_target_raw.json")
    
    # Save natural target as proportions (for direct comparison with percentage beliefs)
    natural_proportions = [size / total_items for size in cluster_sizes]
    
    natural_target_proportions = {
        'natural_proportions': natural_proportions,
        'description': 'Natural proportions of items per cluster (for comparison with percentage beliefs)',
        'cluster_sizes': cluster_sizes,
        'interpretation': 'If users clicked items proportionally, they would click the same % of items in each cluster'
    }
    
    with open('processed_data/natural_belief_target_proportions.json', 'w') as f:
        json.dump(natural_target_proportions, f, indent=2)
    
    print(f"✅ Saved natural_belief_target_proportions.json")

if __name__ == "__main__":
    print("=== RECALCULATING USER BELIEFS WITHOUT NORMALIZATION ===")
    print("This keeps beliefs in their natural form - raw click counts per cluster")
    print("If users clicked all items proportionally, they would match cluster sizes")
    print()
    
    # Recalculate beliefs
    df = recalculate_user_beliefs_raw()
    
    # Create natural target for comparison
    create_natural_belief_target_raw()
    
    print(f"\n✅ COMPLETE!")
    print(f"- Updated embeddings/user_beliefs_K5_topk5.tsv with percentage-based beliefs")
    print(f"- User beliefs = percentage of items clicked per cluster")
    print(f"- Natural distribution = proportion of items available per cluster")
    print(f"- Now both use consistent methodology based on item availability")
    print(f"\nUser beliefs now represent: 'What % of available items did users click in each cluster?'")