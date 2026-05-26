import pickle
import os

def read_pkl_file(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            return None
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded: {file_path}")
        print(f"Data type: {type(data)}")
        
        # Show basic info about the data
        if isinstance(data, (list, tuple)):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item: {data[0]}")
        elif isinstance(data, dict):
            print(f"Keys: {len(data)} items")
            print(f"Sample keys: {list(data.keys())[:5]}")
        else:
            print(f"Data: {data}")
        
        return data
        
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your .pkl file path
    file_path = "embeddings/user_embedding.pkl"
    
    data = read_pkl_file(file_path)
    
    # You can now work with the loaded data
    if data is not None:
        print("\n--- Additional Analysis ---")
        print("First few user embeddings:")
        count = 0
        for user_id, embedding in data.items():
            if count >= 3:
                break
            print(f"User ID: {user_id} (type: {type(user_id)})")
            print(f"Embedding type: {type(embedding)}")
            if hasattr(embedding, 'shape'):
                print(f"Embedding shape: {embedding.shape}")
            if hasattr(embedding, '__len__'):
                print(f"Embedding length: {len(embedding)}")
            print(f"Embedding values: {embedding}")
            print("---")
            count += 1