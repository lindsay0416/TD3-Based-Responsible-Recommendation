import pickle
import json
import numpy as np

def load_data():
    """
    Load all necessary data files
    """
    # Load item embeddings
    with open('embeddings/item_embedding.pkl', 'rb') as f:
        item_embeddings = pickle.load(f)
    print(f"Loaded item embeddings: {len(item_embeddings)} items")
    
    # Load cluster information
    with open('embeddings/cluster_matrix_manifest_K5_topk5.json', 'r') as f:
        cluster_data = json.load(f)
    cluster_to_items = cluster_data['cluster_to_items']
    print(f"Loaded cluster data: {len(cluster_to_items)} clusters")
    
    # Load item token mapping
    with open('embeddings/item_token_map.json', 'r') as f:
        item_token_map = json.load(f)
    print(f"Loaded item token map: {len(item_token_map)} mappings")
    
    return item_embeddings, cluster_to_items, item_token_map

def calculate_cluster_embeddings(item_embeddings, cluster_to_items, item_token_map):
    """
    Calculate cluster embeddings by averaging item embeddings within each cluster
    Uses item_token_map to convert item IDs to tokens for embedding lookup
    """
    cluster_embeddings = {}
    
    # Get embedding dimension from first item
    first_embedding = next(iter(item_embeddings.values()))
    embedding_dim = len(first_embedding) if isinstance(first_embedding, list) else first_embedding.shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    
    for cluster_id, item_list in cluster_to_items.items():
        print(f"\nProcessing cluster {cluster_id} with {len(item_list)} items")
        
        cluster_item_embeddings = []
        missing_items = []
        found_items = []
        
        for item_id in item_list:
            # Use item_token_map to get the token number
            if item_id in item_token_map:
                token = item_token_map[item_id]
                
                # Check if token exists in embeddings (convert to string)
                if str(token) in item_embeddings:
                    embedding = item_embeddings[str(token)]
                    # Convert to numpy array if it's a list
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    cluster_item_embeddings.append(embedding)
                    found_items.append(item_id)
                else:
                    missing_items.append(f"{item_id}->token:{token}")
            else:
                missing_items.append(f"{item_id}->no_token")
        
        if missing_items:
            print(f"  Warning: {len(missing_items)} items not found")
            if len(missing_items) <= 5:
                print(f"  Missing: {missing_items}")
            else:
                print(f"  Sample missing: {missing_items[:3]}")
        
        if cluster_item_embeddings:
            # Calculate average embedding for this cluster
            cluster_embedding = np.mean(cluster_item_embeddings, axis=0)
            cluster_embeddings[cluster_id] = cluster_embedding
            
            print(f"  ✅ Successfully calculated embedding for cluster {cluster_id}")
            print(f"  Used {len(cluster_item_embeddings)}/{len(item_list)} items ({len(cluster_item_embeddings)/len(item_list)*100:.1f}%)")
            print(f"  Embedding shape: {cluster_embedding.shape}")
            print(f"  Embedding norm: {np.linalg.norm(cluster_embedding):.4f}")
        else:
            print(f"  ❌ Error: No valid embeddings found for cluster {cluster_id}")
    
    return cluster_embeddings

def save_cluster_embeddings(cluster_embeddings):
    """
    Save cluster embeddings to processed_data folder
    """
    if not cluster_embeddings:
        print("No cluster embeddings to save!")
        return
    
    # Save as pickle for fastest loading
    with open('processed_data/cluster_embeddings.pkl', 'wb') as f:
        pickle.dump(cluster_embeddings, f)
    
    # Also save as numpy array for easy access
    cluster_array = np.array([cluster_embeddings[str(i)] for i in range(len(cluster_embeddings))])
    np.save('processed_data/cluster_embeddings.npy', cluster_array)
    
    # Save sample for inspection
    sample_data = {}
    for cluster_id, embedding in cluster_embeddings.items():
        sample_data[cluster_id] = {
            'embedding_shape': embedding.shape,
            'embedding_norm': float(np.linalg.norm(embedding)),
            'first_5_values': embedding[:5].tolist(),
            'last_5_values': embedding[-5:].tolist()
        }
    
    with open('processed_data/cluster_embeddings_info.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"\nSaved cluster embeddings:")
    print(f"  - cluster_embeddings.pkl (dictionary format)")
    print(f"  - cluster_embeddings.npy (array format: shape {cluster_array.shape})")
    print(f"  - cluster_embeddings_info.json (metadata)")

def analyze_cluster_embeddings(cluster_embeddings):
    """
    Analyze the calculated cluster embeddings
    """
    if not cluster_embeddings:
        print("No cluster embeddings to analyze!")
        return
        
    print(f"\nCluster Embedding Analysis:")
    
    # Convert to array for analysis
    embeddings_array = np.array([cluster_embeddings[str(i)] for i in range(len(cluster_embeddings))])
    
    print(f"Shape: {embeddings_array.shape}")
    print(f"Data type: {embeddings_array.dtype}")
    
    # Calculate pairwise similarities
    print(f"\nPairwise cosine similarities between clusters:")
    for i in range(len(cluster_embeddings)):
        for j in range(i+1, len(cluster_embeddings)):
            sim = np.dot(embeddings_array[i], embeddings_array[j]) / (
                np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
            )
            print(f"  Cluster {i} - Cluster {j}: {sim:.4f}")
    
    # Embedding statistics
    print(f"\nEmbedding norms:")
    for i in range(len(cluster_embeddings)):
        norm = np.linalg.norm(embeddings_array[i])
        print(f"  Cluster {i}: {norm:.4f}")

if __name__ == "__main__":
    # Load all data
    item_embeddings, cluster_to_items, item_token_map = load_data()
    
    # Calculate cluster embeddings
    cluster_embeddings = calculate_cluster_embeddings(item_embeddings, cluster_to_items, item_token_map)
    
    if cluster_embeddings:
        # Save the results
        save_cluster_embeddings(cluster_embeddings)
        
        # Analyze the results
        analyze_cluster_embeddings(cluster_embeddings)
    else:
        print("Failed to calculate any cluster embeddings!")