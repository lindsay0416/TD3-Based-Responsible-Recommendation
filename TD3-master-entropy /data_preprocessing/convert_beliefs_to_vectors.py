# import pickle
# import numpy as np
# import json

# def load_and_convert_beliefs():
#     """
#     Load user beliefs and convert them to proper numpy vectors for RL operations
#     """
#     # Load the processed beliefs
#     with open('processed_data/user_avg_beliefs.pkl', 'rb') as f:
#         user_beliefs = pickle.load(f)
    
#     print(f"Loaded beliefs for {len(user_beliefs)} users")
    
#     # Convert to numpy arrays for efficient math operations
#     user_belief_vectors = {}
    
#     for userid, belief_list in user_beliefs.items():
#         # Convert to numpy array with shape (5,)
#         belief_vector = np.array(belief_list, dtype=np.float32)
        
#         # Verify it's 5-dimensional
#         assert belief_vector.shape == (5,), f"User {userid} has wrong belief dimension: {belief_vector.shape}"
        
#         # Verify it's normalized (sum ≈ 1.0)
#         belief_sum = np.sum(belief_vector)
#         if not np.isclose(belief_sum, 1.0, atol=1e-6):
#             print(f"Warning: User {userid} beliefs don't sum to 1.0: {belief_sum}")
        
#         user_belief_vectors[userid] = belief_vector
    
#     return user_belief_vectors

# def save_belief_vectors(user_belief_vectors):
#     """
#     Save the vectorized beliefs for fast loading in RL environment
#     """
#     # Save as pickle for fastest loading
#     with open('processed_data/user_belief_vectors.pkl', 'wb') as f:
#         pickle.dump(user_belief_vectors, f)
    
#     print(f"Saved belief vectors for {len(user_belief_vectors)} users")
    
#     # Also create a sample for inspection
#     sample_users = list(user_belief_vectors.keys())[:5]
#     sample_data = {uid: user_belief_vectors[uid].tolist() for uid in sample_users}
    
#     with open('processed_data/sample_belief_vectors.json', 'w') as f:
#         json.dump(sample_data, f, indent=2)
    
#     return user_belief_vectors

# def analyze_belief_vectors(user_belief_vectors):
#     """
#     Analyze the belief vectors for RL environment design
#     """
#     all_vectors = np.array(list(user_belief_vectors.values()))
    
#     print(f"\nBelief Vector Analysis:")
#     print(f"Shape: {all_vectors.shape}")  # Should be (50000, 5)
#     print(f"Data type: {all_vectors.dtype}")
#     print(f"Mean per cluster: {np.mean(all_vectors, axis=0)}")
#     print(f"Std per cluster: {np.std(all_vectors, axis=0)}")
#     print(f"Min values per cluster: {np.min(all_vectors, axis=0)}")
#     print(f"Max values per cluster: {np.max(all_vectors, axis=0)}")
    
#     # Check for users with strong preferences (high max belief)
#     max_beliefs = np.max(all_vectors, axis=1)
#     strong_preference_users = np.sum(max_beliefs > 0.8)
#     print(f"Users with strong cluster preference (>0.8): {strong_preference_users}")
    
#     # Check for balanced users (similar beliefs across clusters)
#     belief_variance = np.var(all_vectors, axis=1)
#     balanced_users = np.sum(belief_variance < 0.01)
#     print(f"Users with balanced preferences (variance <0.01): {balanced_users}")

# if __name__ == "__main__":
#     # Load and convert beliefs to vectors
#     user_belief_vectors = load_and_convert_beliefs()
    
#     # Save the vectorized format
#     save_belief_vectors(user_belief_vectors)
    
#     # Analyze the vectors
#     analyze_belief_vectors(user_belief_vectors)
    
#     # Show sample vectors
#     print(f"\nSample belief vectors:")
#     for i, (userid, vector) in enumerate(list(user_belief_vectors.items())[:3]):
#         print(f"User {userid}: {vector}")
#         print(f"  - Dominant cluster: {np.argmax(vector)}")
#         print(f"  - Max belief: {np.max(vector):.3f}")