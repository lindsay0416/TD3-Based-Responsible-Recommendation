#!/usr/bin/env python3
"""
Calculate percentage of distance improvement compared to initial belief distance.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_natural_target(dataset: str) -> np.ndarray:
    """Load natural belief target for a dataset."""
    natural_beliefs_path = Path(f"processed_data/{dataset}/Topk5/natural_belief_target.json")
    
    if not natural_beliefs_path.exists():
        print(f"⚠️  Natural target not found for {dataset}, using uniform [0.2, 0.2, 0.2, 0.2, 0.2]")
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    with open(natural_beliefs_path, 'r') as f:
        data = json.load(f)
    
    return np.array(data['natural_belief_target'])


def calculate_initial_distance(dataset: str, natural_target: np.ndarray) -> float:
    """Calculate average initial distance from user beliefs to natural target."""
    user_beliefs_path = Path(f"processed_data/{dataset}/Topk5/user_average_beliefs.json")
    
    if not user_beliefs_path.exists():
        print(f"⚠️  User beliefs not found for {dataset}")
        return None
    
    with open(user_beliefs_path, 'r') as f:
        user_data = json.load(f)
    
    # Calculate distance for each user
    distances = []
    for user_id, data in user_data.items():
        beliefs = np.array(data['beliefs'])
        # Mean absolute distance
        distance = np.mean(np.abs(beliefs - natural_target))
        distances.append(distance)
    
    # Return average initial distance
    return np.mean(distances)


def extract_distance_improvement(json_file: Path) -> float:
    """Extract avg_distance_improvement mean value from evaluation_results.json."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'metrics' in data and 'avg_distance_improvement' in data['metrics']:
        return data['metrics']['avg_distance_improvement']['mean']
    
    return None


def main():
    """Calculate improvement percentage for all models and datasets."""
    base_path = Path("RL_evaluation_metrics")
    
    # Find all evaluation_results.json files
    json_files = list(base_path.rglob("evaluation_results.json"))
    
    if not json_files:
        print("No evaluation_results.json files found!")
        return
    
    print(f"Found {len(json_files)} evaluation_results.json files\n")
    
    # Calculate initial distances for each dataset
    datasets = ['books', 'movies', 'news']
    initial_distances = {}
    
    print("Calculating initial distances...")
    for dataset in datasets:
        natural_target = load_natural_target(dataset)
        initial_dist = calculate_initial_distance(dataset, natural_target)
        if initial_dist is not None:
            initial_distances[dataset] = initial_dist
            print(f"  {dataset}: {initial_dist:.6f}")
    
    print()
    
    # Collect improvement data
    results = []
    
    for json_file in sorted(json_files):
        # Parse path to get dataset and model
        parts = json_file.parts
        
        try:
            dataset_idx = parts.index("RL_evaluation_metrics") + 1
            dataset = parts[dataset_idx]
            topk = parts[dataset_idx + 1]
            model = parts[dataset_idx + 2]
        except (IndexError, ValueError):
            print(f"⚠️  Skipping {json_file} - unexpected path structure")
            continue
        
        # Extract distance improvement
        improvement = extract_distance_improvement(json_file)
        
        if improvement is None or dataset not in initial_distances:
            print(f"⚠️  Missing data for {dataset}/{model}")
            continue
        
        # Calculate percentage
        initial_dist = initial_distances[dataset]
        improvement_percentage = (improvement / initial_dist) * 100
        final_distance = initial_dist - improvement
        
        results.append({
            'dataset': dataset,
            'model': model,
            'initial_distance': initial_dist,
            'improvement': improvement,
            'final_distance': final_distance,
            'improvement_percentage': improvement_percentage
        })
        
        print(f"✓ {dataset}/{model}: {improvement_percentage:.2f}% improvement")
    
    if not results:
        print("\nNo data extracted!")
        return
    
    # Save to CSV
    output_file = base_path / "distance_improvement_percentage.csv"
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['dataset', 'model', 'initial_distance', 'improvement', 
                      'final_distance', 'improvement_percentage']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results:
            writer.writerow({
                'dataset': row['dataset'],
                'model': row['model'],
                'initial_distance': f"{row['initial_distance']:.6f}",
                'improvement': f"{row['improvement']:.6f}",
                'final_distance': f"{row['final_distance']:.6f}",
                'improvement_percentage': f"{row['improvement_percentage']:.2f}"
            })
    
    print(f"\n✓ CSV saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*90)
    print("DISTANCE IMPROVEMENT PERCENTAGE SUMMARY")
    print("="*90)
    print(f"{'Dataset':<10} {'Model':<12} {'Initial':<12} {'Improvement':<12} {'Final':<12} {'%':<10}")
    print("-"*90)
    
    for row in results:
        print(f"{row['dataset']:<10} {row['model']:<12} "
              f"{row['initial_distance']:<12.6f} {row['improvement']:<12.6f} "
              f"{row['final_distance']:<12.6f} {row['improvement_percentage']:<10.2f}")
    
    # Print average by dataset
    print("\n" + "="*90)
    print("AVERAGE IMPROVEMENT BY DATASET")
    print("="*90)
    
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if dataset_results:
            avg_pct = np.mean([r['improvement_percentage'] for r in dataset_results])
            print(f"{dataset:<10} Average improvement: {avg_pct:.2f}%")


if __name__ == "__main__":
    main()
