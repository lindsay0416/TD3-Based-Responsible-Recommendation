#!/usr/bin/env python3
"""
Create a horizontal comparison table with datasets as column groups.
Format similar to the reference table with datasets (Books, Movies, News) as column headers.
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
                'entropy': float(row['RL_entropy']),
                'ndcg': float(row['RL_ndcg']),
                'precision': float(row['RL_precision']),
                'recall': float(row['RL_recall']),
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
                'entropy': float(row['RS_entropy']) if row['RS_entropy'] else 0.0,
                'ndcg': float(row['RS_ndcg@10']) if row['RS_ndcg@10'] else 0.0,
                'precision': float(row['RS_precision@10']) if row['RS_precision@10'] else 0.0,
                'recall': float(row['RS_recall@10']) if row['RS_recall@10'] else 0.0,
            }
    
    return rs_data


def main():
    """Create horizontal comparison table."""
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
    
    # Define datasets and models
    datasets = ['books', 'movies', 'news']
    dataset_labels = ['Books', 'Movies', 'News']
    models = ['ENMF', 'LightGCN', 'NCL', 'NGCF', 'SGL']
    metrics = ['recall', 'precision', 'ndcg', 'entropy']
    
    # Create rows
    all_rows = []
    
    # Add header row with dataset names
    header_row = {'model': 'Datasets'}
    for dataset_label in dataset_labels:
        header_row[f'{dataset_label}_recall'] = dataset_label
        header_row[f'{dataset_label}_precision'] = ''
        header_row[f'{dataset_label}_ndcg'] = ''
        header_row[f'{dataset_label}_entropy'] = ''
    all_rows.append(header_row)
    
    # Add metric names row
    metric_row = {'model': 'Metrics'}
    for dataset_label in dataset_labels:
        metric_row[f'{dataset_label}_recall'] = 'Recall'
        metric_row[f'{dataset_label}_precision'] = 'Precision'
        metric_row[f'{dataset_label}_ndcg'] = 'NDCG'
        metric_row[f'{dataset_label}_entropy'] = 'Entropy'
    all_rows.append(metric_row)
    
    # Add RS models
    for model in models:
        row = {'model': model}
        for i, dataset in enumerate(datasets):
            key = (dataset, model)
            if key in rs_data:
                row[f'{dataset_labels[i]}_recall'] = f"{rs_data[key]['recall']:.4f}"
                row[f'{dataset_labels[i]}_precision'] = f"{rs_data[key]['precision']:.4f}"
                row[f'{dataset_labels[i]}_ndcg'] = f"{rs_data[key]['ndcg']:.4f}"
                row[f'{dataset_labels[i]}_entropy'] = f"{rs_data[key]['entropy']:.4f}"
            else:
                row[f'{dataset_labels[i]}_recall'] = ''
                row[f'{dataset_labels[i]}_precision'] = ''
                row[f'{dataset_labels[i]}_ndcg'] = ''
                row[f'{dataset_labels[i]}_entropy'] = ''
        all_rows.append(row)
    
    # Add RL models
    for model in models:
        row = {'model': f'RL_{model}'}
        for i, dataset in enumerate(datasets):
            key = (dataset, model)
            if key in rl_data:
                row[f'{dataset_labels[i]}_recall'] = f"{rl_data[key]['recall']:.4f}"
                row[f'{dataset_labels[i]}_precision'] = f"{rl_data[key]['precision']:.4f}"
                row[f'{dataset_labels[i]}_ndcg'] = f"{rl_data[key]['ndcg']:.4f}"
                row[f'{dataset_labels[i]}_entropy'] = f"{rl_data[key]['entropy']:.4f}"
            else:
                row[f'{dataset_labels[i]}_recall'] = ''
                row[f'{dataset_labels[i]}_precision'] = ''
                row[f'{dataset_labels[i]}_ndcg'] = ''
                row[f'{dataset_labels[i]}_entropy'] = ''
        all_rows.append(row)
    
    # Write to CSV
    output_file = base_path / "horizontal_comparison.csv"
    
    # Define column order
    csv_columns = ['model']
    for dataset_label in dataset_labels:
        csv_columns.extend([
            f'{dataset_label}_recall',
            f'{dataset_label}_precision',
            f'{dataset_label}_ndcg',
            f'{dataset_label}_entropy'
        ])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n✓ Horizontal comparison table saved to: {output_file}")
    print(f"  Total rows: {len(all_rows)}")
    print(f"  Format: Datasets as column groups (Books, Movies, News)")
    print(f"  Metrics: Recall, Precision, NDCG, Entropy for each dataset")
    
    # Print preview
    print("\n" + "="*120)
    print("PREVIEW")
    print("="*120)
    print(f"{'Model':<15} {'Books':<50} {'Movies':<50}")
    print(f"{'':15} {'R':>10} {'P':>10} {'N':>10} {'E':>10} {'R':>10} {'P':>10} {'N':>10} {'E':>10}")
    print("-"*120)
    
    for row in all_rows[2:7]:  # Show first 5 data rows
        model = row['model'][:14]
        b_r = row.get('Books_recall', '')[:8]
        b_p = row.get('Books_precision', '')[:8]
        b_n = row.get('Books_ndcg', '')[:8]
        b_e = row.get('Books_entropy', '')[:8]
        m_r = row.get('Movies_recall', '')[:8]
        m_p = row.get('Movies_precision', '')[:8]
        m_n = row.get('Movies_ndcg', '')[:8]
        m_e = row.get('Movies_entropy', '')[:8]
        
        print(f"{model:<15} {b_r:>10} {b_p:>10} {b_n:>10} {b_e:>10} {m_r:>10} {m_p:>10} {m_n:>10} {m_e:>10}")


if __name__ == "__main__":
    main()
