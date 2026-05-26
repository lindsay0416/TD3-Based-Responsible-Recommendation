#!/usr/bin/env python3
"""
Setup script for Recommendation RL System.
Validates data files and prepares environment for training.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        print(f"{description}: {filepath} ({size:.1f} MB)")
        return True
    else:
        print(f"{description}: {filepath} (NOT FOUND)")
        return False


def validate_data_files() -> bool:
    """Validate all required data files exist."""
    print("Validating data files...")
    print("-" * 50)
    
    required_files = [
        ("processed_data/user_average_beliefs.json", "User Beliefs"),
        ("embeddings/user_embedding.pkl", "User Embeddings"),
        ("embeddings/item_embedding.pkl", "Item Embeddings"),
        ("processed_data/cluster_embeddings.pkl", "Cluster Embeddings"),
        ("processed_data/natural_belief_target.pkl", "Natural Belief Target"),
        ("embeddings/recommendations.pkl", "Pre-trained Recommendations"),
        ("embeddings/item_token_map.json", "Item Token Map"),
        ("embeddings/user_token_map.json", "User Token Map"),
        ("embeddings/cluster_matrix_manifest_K5_topk5.json", "Cluster Manifest")
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def validate_data_integrity() -> bool:
    """Validate data integrity and compatibility."""
    print("\nValidating data integrity...")
    print("-" * 50)
    
    try:
        # Load and check user beliefs
        with open("processed_data/user_average_beliefs.json", 'r') as f:
            user_beliefs = json.load(f)
        
        print(f"User beliefs loaded: {len(user_beliefs)} users")
        
        # Check belief structure
        sample_user = list(user_beliefs.keys())[0]
        sample_data = user_beliefs[sample_user]
        
        if 'beliefs' in sample_data and 'pp1_distance' in sample_data:
            belief_dim = len(sample_data['beliefs'])
            print(f"Belief dimension: {belief_dim}")
        else:
            print("Invalid user belief structure")
            return False
        
        # Load and check embeddings
        with open("embeddings/user_embedding.pkl", 'rb') as f:
            user_embeddings = pickle.load(f)
        
        print(f"User embeddings loaded: {len(user_embeddings)} users")
        
        # Check embedding dimension
        sample_embedding = next(iter(user_embeddings.values()))
        embedding_dim = len(sample_embedding)
        print(f"User embedding dimension: {embedding_dim}")
        
        # Validate state dimension
        expected_state_dim = belief_dim + embedding_dim
        print(f"Expected state dimension: {expected_state_dim} (beliefs: {belief_dim} + embedding: {embedding_dim})")
        
        return True
        
    except Exception as e:
        print(f"Data validation error: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    print("-" * 50)
    
    directories = ["results", "models", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking dependencies...")
    print("-" * 50)
    
    required_packages = [
        "torch", "numpy", "sklearn", "pandas", "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            print(f"{package} (NOT INSTALLED)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main setup function."""
    print("="*60)
    print("RECOMMENDATION RL SYSTEM - SETUP")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nSetup failed: Missing dependencies")
        sys.exit(1)
    
    # Validate data files
    if not validate_data_files():
        print("\nSetup failed: Missing data files")
        print("\nPlease run the data preprocessing scripts first:")
        print("1. python data_preprocessing/calculate_user_average_beliefs.py")
        print("2. python data_preprocessing/calculate_cluster_embeddings_fixed.py")
        print("3. Ensure all embedding files are in the embeddings/ folder")
        sys.exit(1)
    
    # Validate data integrity
    if not validate_data_integrity():
        print("\nSetup failed: Data integrity issues")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the training with:")
    print("python run_recommendation_rl.py")
    print("\nOr validate configuration first:")
    print("python run_recommendation_rl.py --validate-only")


if __name__ == "__main__":
    main()