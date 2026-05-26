#!/usr/bin/env python3
"""
Create a comparison table with datasets as separate sections.
Format: model names as RS_ModelName and RL_ModelName, unified metric names.
"""

import csv
from pathlib import Path
from typing import Dict, List


def load_rl_metrics(csv_file: Path) -> Dict:
    """Load RL metrics from evaluation_mean_values.csv."""
    rl_data = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['dataset'], row['model'])
            rl_data[key] = {
                'entropy': row['RL_entropy'],
                'ndcg': row['RL_ndcg'],
                'precision': row['RL_precision'],
                'recall': row['RL_recall'],
                'acceptance_rate': row['RL_acceptance_rate'],
            }
    
    return rl_data


def load_rs_metrics(csv_file: Path) -> Dict:
    """Load RS metrics from recbole_metrics_with_entropy.csv."""
    rs_data = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['dataset'], row['model'])
            rs_data[key] = {
                'entropy': row['RS_entropy'],
                'ndcg': row['RS_ndcg@10'],
                'precision': row['RS_precision@10'],
                'recall': row['RS_recall@10'],
            }
    
    return rs_data


def main():
    """Create comparison table with unified format."""
    base_path = Path("RL_evaluation_metrics")
    
    # Load data
    rl_file = base_path / "evaluation_mean_values.csv"
    rs_file = base_path / "recbole_metrics_with_entropy.csv"
    
    if not rl_file.exists() or not rs_file.exists():
        print("❌ Required files not found!")
        return
    
    print("Loading metrics...")
    rl_data = load_rl_metrics(rl_file)
    rs_data = load_rs_metrics(rs_file)
    
    # Get all datasets and models
    datasets = sorted(set(key[0] for key in rl_data.keys()))
    models = sorted(set(key[1] for key in rl_data.keys()))
    
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    
    # Create comparison data
    all_rows = []
    
    for dataset in datasets:
        # Add dataset header row
        all_rows.append({
            'dataset': f'=== {dataset.upper()} ===',
            'model': '',
            'entropy': '',
            'ndcg': '',
            'precision': '',
            'recall': '',
            'acceptance_rate': '',
        })
        
        # Add RS models
        for model in models:
            key = (dataset, model)
            if key in rs_data:
                row = {
                    'dataset': dataset,
                    'model': model,  # RS model uses original name
                    'entropy': rs_data[key]['entropy'],
                    'ndcg': rs_data[key]['ndcg'],
                    'precision': rs_data[key]['precision'],
                    'recall': rs_data[key]['recall'],
                    'acceptance_rate': '',  # RS doesn't have acceptance rate
                }
                all_rows.append(row)
        
        # Add RL models
        for model in models:
            key = (dataset, model)
            if key in rl_data:
                row = {
                    'dataset': dataset,
                    'model': f'RL_{model}',  # RL model prefixed with RL_
                    'entropy': rl_data[key]['entropy'],
                    'ndcg': rl_data[key]['ndcg'],
                    'precision': rl_data[key]['precision'],
                    'recall': rl_data[key]['recall'],
                    'acceptance_rate': rl_data[key]['acceptance_rate'],
                }
                all_rows.append(row)
        
        # Add empty row between datasets
        all_rows.append({
            'dataset': '',
            'model': '',
            'entropy': '',
            'ndcg': '',
            'precision': '',
            'recall': '',
            'acceptance_rate': '',
        })
    
    # Write to CSV
    output_file = base_path / "comparison_table.csv"
    
    csv_columns = ['dataset', 'model', 'entropy', 'ndcg', 'precision', 'recall', 'acceptance_rate']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n✓ Comparison table saved to: {output_file}")
    print(f"  Total rows: {len(all_rows)}")
    
    # Print preview
    print("\n" + "="*100)
    print("PREVIEW (first 20 rows)")
    print("="*100)
    print(f"{'Dataset':<15} {'Model':<15} {'Entropy':<12} {'NDCG':<10} {'Precision':<12} {'Recall':<10} {'Accept Rate':<12}")
    print("-"*100)
    
    for i, row in enumerate(all_rows[:20]):
        dataset = row['dataset'][:14] if row['dataset'] else ''
        model = row['model'][:14] if row['model'] else ''
        entropy = str(row['entropy'])[:10] if row['entropy'] else ''
        ndcg = str(row['ndcg'])[:8] if row['ndcg'] else ''
        precision = str(row['precision'])[:10] if row['precision'] else ''
        recall = str(row['recall'])[:8] if row['recall'] else ''
        accept = str(row['acceptance_rate'])[:10] if row['acceptance_rate'] else ''
        
        print(f"{dataset:<15} {model:<15} {entropy:<12} {ndcg:<10} {precision:<12} {recall:<10} {accept:<12}")


if __name__ == "__main__":
    main()
