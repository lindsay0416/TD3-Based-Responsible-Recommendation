#!/usr/bin/env python3
"""
Convert cluster_matrix_manifest JSON file to PKL format
"""

import json
import pickle
import sys
from pathlib import Path


def convert_json_to_pkl(json_path: str, pkl_path: str = None):
    """
    Convert a JSON file to PKL format
    
    Args:
        json_path: Path to input JSON file
        pkl_path: Path to output PKL file (optional, will use same name with .pkl extension if not provided)
    """
    json_file = Path(json_path)
    
    if not json_file.exists():
        print(f"❌ Error: File not found: {json_path}")
        return False
    
    # Auto-generate pkl path if not provided
    if pkl_path is None:
        pkl_path = json_file.with_suffix('.pkl')
    else:
        pkl_path = Path(pkl_path)
    
    print(f"📖 Reading JSON file: {json_file}")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded data with {len(data)} keys")
        if 'num_clusters' in data:
            print(f"   - num_clusters: {data['num_clusters']}")
        if 'num_nodes' in data:
            print(f"   - num_nodes: {data['num_nodes']}")
        if 'cluster_sizes' in data:
            print(f"   - cluster_sizes: {data['cluster_sizes']}")
        
        print(f"\n💾 Writing PKL file: {pkl_path}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify the file was created
        if pkl_path.exists():
            size_kb = pkl_path.stat().st_size / 1024
            print(f"✅ Successfully created PKL file ({size_kb:.1f} KB)")
            
            # Quick verification
            with open(pkl_path, 'rb') as f:
                verify_data = pickle.load(f)
            
            if verify_data == data:
                print("✅ Verification passed: Data integrity confirmed")
            else:
                print("⚠️  Warning: Verification failed - data may be corrupted")
                return False
            
            return True
        else:
            print("❌ Error: PKL file was not created")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Default path for books dataset
    default_json = "embeddings/books/K_means/cluster_matrix_manifest_K5.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        pkl_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        json_path = default_json
        pkl_path = None
    
    print("=" * 60)
    print("JSON to PKL Converter")
    print("=" * 60)
    
    success = convert_json_to_pkl(json_path, pkl_path)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)
