import pickle

def check_embedding_keys():
    """
    Check what keys are actually used in the item embeddings
    """
    with open('embeddings/item_embedding.pkl', 'rb') as f:
        item_embeddings = pickle.load(f)
    
    print(f"Item embeddings type: {type(item_embeddings)}")
    print(f"Number of items: {len(item_embeddings)}")
    
    # Check first 20 keys
    keys = list(item_embeddings.keys())
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(keys[:20]):
        print(f"  {i}: {key} (type: {type(key)})")
    
    print(f"\nLast 10 keys:")
    for i, key in enumerate(keys[-10:]):
        print(f"  {len(keys)-10+i}: {key} (type: {type(key)})")
    
    # Check if keys are sequential numbers
    if all(isinstance(k, int) for k in keys[:10]):
        print(f"\nKeys appear to be integers")
        print(f"Min key: {min(keys)}")
        print(f"Max key: {max(keys)}")
        print(f"Are keys sequential? {keys == list(range(min(keys), max(keys)+1))}")

if __name__ == "__main__":
    check_embedding_keys()