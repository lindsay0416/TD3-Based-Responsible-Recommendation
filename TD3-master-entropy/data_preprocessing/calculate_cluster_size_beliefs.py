import json
import numpy as np
import pickle

def calculate_cluster_size_beliefs():
    """
    Calculate optimal belief vector based on cluster sizes from cluster_to_items
    """
    # Load cluster information
    with open('embeddings/cluster_matrix_manifest_K5_topk5.json', 'r') as f:
        cluster_data = json.load(f)
    
    cluster_to_items = cluster_data['cluster_to_items']
    
    print("=== CLUSTER SIZE ANALYSIS ===")
    
    # Get cluster sizes
    cluster_sizes = {}
    total_items = 0
    
    for cluster_id, items in cluster_to_items.items():
        cluster_size = len(items)
        cluster_sizes[cluster_id] = cluster_size
        total_items += cluster_size
        print(f"Cluster {cluster_id}: {cluster_size} items")
    
    print(f"Total items across all clusters: {total_items}")
    
    # Calculate proportional beliefs (Method 1: Simple Proportional)
    proportional_beliefs = np.zeros(5)
    for i in range(5):
        cluster_size = cluster_sizes[str(i)]
        proportional_beliefs[i] = cluster_size / total_items
    
    print(f"\n=== METHOD 1: PROPORTIONAL BELIEFS ===")
    print(f"Belief vector: {proportional_beliefs}")
    print(f"Sum: {np.sum(proportional_beliefs):.6f}")
    
    # Calculate square root normalized beliefs (Method 2: Sqrt Normalization)
    # This reduces the dominance of large clusters
    sqrt_beliefs = np.zeros(5)
    sqrt_sizes = []
    for i in range(5):
        cluster_size = cluster_sizes[str(i)]
        sqrt_size = np.sqrt(cluster_size)
        sqrt_sizes.append(sqrt_size)
    
    total_sqrt = sum(sqrt_sizes)
    for i in range(5):
        sqrt_beliefs[i] = sqrt_sizes[i] / total_sqrt
    
    print(f"\n=== METHOD 2: SQRT NORMALIZED BELIEFS ===")
    print(f"Belief vector: {sqrt_beliefs}")
    print(f"Sum: {np.sum(sqrt_beliefs):.6f}")
    
    # Calculate log normalized beliefs (Method 3: Log Normalization)
    # This further reduces the dominance of large clusters
    log_beliefs = np.zeros(5)
    log_sizes = []
    for i in range(5):
        cluster_size = cluster_sizes[str(i)]
        log_size = np.log(cluster_size + 1)  # +1 to avoid log(0)
        log_sizes.append(log_size)
    
    total_log = sum(log_sizes)
    for i in range(5):
        log_beliefs[i] = log_sizes[i] / total_log
    
    print(f"\n=== METHOD 3: LOG NORMALIZED BELIEFS ===")
    print(f"Belief vector: {log_beliefs}")
    print(f"Sum: {np.sum(log_beliefs):.6f}")
    
    # Calculate balanced beliefs (Method 4: Balanced with minimum)
    # Ensures no cluster gets less than a minimum threshold
    min_belief = 0.05  # 5% minimum per cluster
    balanced_beliefs = proportional_beliefs.copy()
    
    # Ensure minimum belief for each cluster
    for i in range(5):
        if balanced_beliefs[i] < min_belief:
            balanced_beliefs[i] = min_belief
    
    # Renormalize
    balanced_beliefs = balanced_beliefs / np.sum(balanced_beliefs)
    
    print(f"\n=== METHOD 4: BALANCED BELIEFS (min 5%) ===")
    print(f"Belief vector: {balanced_beliefs}")
    print(f"Sum: {np.sum(balanced_beliefs):.6f}")
    
    return {
        'proportional': proportional_beliefs,
        'sqrt_normalized': sqrt_beliefs,
        'log_normalized': log_beliefs,
        'balanced': balanced_beliefs,
        'cluster_sizes': cluster_sizes,
        'total_items': total_items
    }

def analyze_methods(belief_methods):
    """
    Compare different belief calculation methods
    """
    print(f"\n=== COMPARISON OF METHODS ===")
    
    methods = ['proportional', 'sqrt_normalized', 'log_normalized', 'balanced']
    
    print(f"{'Method':<15} {'Cluster 0':<10} {'Cluster 1':<10} {'Cluster 2':<10} {'Cluster 3':<10} {'Cluster 4':<10} {'Max/Min Ratio':<12}")
    print("-" * 85)
    
    for method in methods:
        beliefs = belief_methods[method]
        max_belief = np.max(beliefs)
        min_belief = np.min(beliefs)
        ratio = max_belief / min_belief
        
        print(f"{method:<15} {beliefs[0]:<10.4f} {beliefs[1]:<10.4f} {beliefs[2]:<10.4f} {beliefs[3]:<10.4f} {beliefs[4]:<10.4f} {ratio:<12.2f}")
    
    # Show cluster sizes for reference
    cluster_sizes = belief_methods['cluster_sizes']
    print(f"\nCluster sizes for reference:")
    for i in range(5):
        size = cluster_sizes[str(i)]
        print(f"  Cluster {i}: {size} items")

def save_optimal_beliefs(belief_methods):
    """
    Save the calculated belief vectors
    """
    # Save all methods
    with open('processed_data/cluster_size_beliefs.pkl', 'wb') as f:
        pickle.dump(belief_methods, f)
    
    # Save as JSON for inspection
    json_data = {}
    for method, beliefs in belief_methods.items():
        if isinstance(beliefs, np.ndarray):
            json_data[method] = beliefs.tolist()
        else:
            json_data[method] = beliefs
    
    with open('processed_data/cluster_size_beliefs.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n=== SAVED FILES ===")
    print(f"  - cluster_size_beliefs.pkl (all methods)")
    print(f"  - cluster_size_beliefs.json (human readable)")

def recommend_method(belief_methods):
    """
    Provide recommendation on which method to use
    """
    print(f"\n=== RECOMMENDATIONS ===")
    
    cluster_sizes = belief_methods['cluster_sizes']
    largest_cluster = max(cluster_sizes.values())
    smallest_cluster = min(cluster_sizes.values())
    size_ratio = largest_cluster / smallest_cluster
    
    print(f"Cluster size ratio (largest/smallest): {size_ratio:.1f}")
    
    if size_ratio > 10:
        print(f"📊 RECOMMENDATION: Use 'sqrt_normalized' or 'log_normalized'")
        print(f"   Reason: Large cluster dominance (ratio > 10) - need to balance exploration")
        recommended = 'sqrt_normalized'
    elif size_ratio > 5:
        print(f"📊 RECOMMENDATION: Use 'balanced' method")
        print(f"   Reason: Moderate cluster dominance - balanced approach works well")
        recommended = 'balanced'
    else:
        print(f"📊 RECOMMENDATION: Use 'proportional' method")
        print(f"   Reason: Clusters are relatively balanced - proportional is natural")
        recommended = 'proportional'
    
    print(f"\nRecommended belief vector: {belief_methods[recommended]}")
    return recommended

if __name__ == "__main__":
    print("=== CALCULATING OPTIMAL BELIEFS FROM CLUSTER SIZES ===")
    
    # Calculate beliefs using different methods
    belief_methods = calculate_cluster_size_beliefs()
    
    # Analyze and compare methods
    analyze_methods(belief_methods)
    
    # Get recommendation
    recommended_method = recommend_method(belief_methods)
    
    # Save results
    save_optimal_beliefs(belief_methods)
    
    print(f"\n✅ Optimal belief calculation complete!")
    print(f"💡 Recommended method: '{recommended_method}'")