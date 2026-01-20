#!/usr/bin/env python3
"""
Analyze belief changes for all 20 episodes compared to initial beliefs
Saves simplified results to file
"""

import json
import numpy as np
from pathlib import Path

# Output file
OUTPUT_FILE = "belief_analysis_episodes_vs_initial.txt"

# Load initial user beliefs (before training)
print("Loading initial user beliefs...")
with open("processed_data/news/Topk5/user_average_beliefs.json", "r") as f:
    initial_data = json.load(f)

# Extract initial beliefs
initial_beliefs = []
user_ids = list(initial_data.keys())
for user_id in user_ids:
    initial_beliefs.append(initial_data[user_id]['beliefs'])

initial_beliefs = np.array(initial_beliefs)
avg_initial = np.mean(initial_beliefs, axis=0)

print(f"Loaded {len(user_ids)} users")

# Load natural belief target
try:
    with open("processed_data/news/Topk5/natural_belief_target.json", "r") as f:
        target_data = json.load(f)
    natural_target = np.array(target_data['natural_belief_target'])
    has_target = True
except FileNotFoundError:
    print("Warning: Natural belief target not found")
    natural_target = None
    has_target = False

# Process all 20 episodes
print("\nProcessing episodes 1-20...")
results_dir = Path("results")
episode_results = {}

for ep in range(1, 51):
    ep_file = results_dir / f"user_beliefs_episode_{ep}.json"
    if ep_file.exists():
        with open(ep_file, "r") as f:
            ep_data = json.load(f)
        
        # Extract beliefs for this episode
        ep_beliefs = []
        for user_id in user_ids:
            if user_id in ep_data['user_beliefs']:
                ep_beliefs.append(ep_data['user_beliefs'][user_id]['beliefs'])
            else:
                # If user not in episode, use initial beliefs
                ep_beliefs.append(initial_data[user_id]['beliefs'])
        
        ep_beliefs = np.array(ep_beliefs)
        avg_ep = np.mean(ep_beliefs, axis=0)
        
        # Calculate changes
        belief_change = avg_ep - avg_initial
        percent_change = np.where(avg_initial != 0, (belief_change / avg_initial) * 100, 0)
        
        # Calculate distance from target
        if has_target:
            gap_from_target = natural_target - avg_ep
            distance_from_target = np.mean(np.abs(gap_from_target))
        else:
            gap_from_target = None
            distance_from_target = None
        
        episode_results[ep] = {
            'avg_beliefs': avg_ep,
            'belief_change': belief_change,
            'percent_change': percent_change,
            'gap_from_target': gap_from_target,
            'distance_from_target': distance_from_target
        }
        
        if ep % 5 == 0:
            print(f"  Processed episode {ep}")

print(f"\nProcessed {len(episode_results)} episodes")

# Save results to file
print(f"\nSaving results to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    f.write("="*100 + "\n")
    f.write("BELIEF CHANGE ANALYSIS: ALL EPISODES vs INITIAL\n")
    f.write("="*100 + "\n")
    
    f.write(f"\nTotal users analyzed: {len(user_ids)}\n")
    f.write(f"Number of clusters: {len(avg_initial)}\n")
    
    f.write("\n" + "-"*100 + "\n")
    f.write("INITIAL AVERAGE BELIEFS (Baseline)\n")
    f.write("-"*100 + "\n")
    for i in range(len(avg_initial)):
        f.write(f"  Cluster {i}: {avg_initial[i]:.17f}\n")
    
    if has_target:
        f.write("\n" + "-"*100 + "\n")
        f.write("NATURAL BELIEF TARGET\n")
        f.write("-"*100 + "\n")
        for i in range(len(natural_target)):
            f.write(f"  Cluster {i}: {natural_target[i]:.17f}\n")
    
    # Write simplified results for each episode
    for ep in sorted(episode_results.keys()):
        data = episode_results[ep]
        
        f.write("\n" + "="*100 + "\n")
        f.write(f"EPISODE {ep} vs INITIAL\n")
        f.write("="*100 + "\n")
        
        f.write(f"{'Cluster':<10} {'Initial':<25} {'Episode '+str(ep):<25} {'Change':<25} {'% Change':<15}\n")
        f.write("-"*120 + "\n")
        
        for i in range(len(avg_initial)):
            if avg_initial[i] == 0:
                pct_str = "N/A (0 base)"
            else:
                pct_str = f"{data['percent_change'][i]:.2f}%"
            
            f.write(f"Cluster {i:<3} {avg_initial[i]:<25.17f} {data['avg_beliefs'][i]:<25.17f} "
                  f"{data['belief_change'][i]:<+25.17f} {pct_str:<15}\n")
        
        f.write("-"*120 + "\n")
        f.write(f"{'TOTAL':<10} {np.sum(avg_initial):<25.17f} {np.sum(data['avg_beliefs']):<25.17f} "
              f"{np.sum(data['belief_change']):<+25.17f}\n")
        
        if has_target:
            f.write(f"Average distance from target: {data['distance_from_target']:.17f}\n")
    
    # Summary: Distance from target progression
    if has_target:
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY: DISTANCE FROM TARGET PROGRESSION\n")
        f.write("="*100 + "\n")
        for ep in sorted(episode_results.keys()):
            dist = episode_results[ep]['distance_from_target']
            f.write(f"Episode {ep:>2}: {dist:.17f}\n")
    
    # Summary: Cluster-wise progression
    f.write("\n" + "="*120 + "\n")
    f.write("SUMMARY: CLUSTER-WISE BELIEF PROGRESSION\n")
    f.write("="*120 + "\n")
    
    # Header
    f.write(f"{'Cluster':<10}")
    f.write(f"{'Initial':<22}")
    for ep in sorted(episode_results.keys()):
        f.write(f"{'Ep'+str(ep):<22}")
    if has_target:
        f.write(f"{'Target':<22}")
    f.write("\n")
    f.write("-"*120 + "\n")
    
    # Data rows
    for i in range(len(avg_initial)):
        f.write(f"Cluster {i:<3}")
        f.write(f"{avg_initial[i]:<22.17f}")
        for ep in sorted(episode_results.keys()):
            f.write(f"{episode_results[ep]['avg_beliefs'][i]:<22.17f}")
        if has_target:
            f.write(f"{natural_target[i]:<22.17f}")
        f.write("\n")
    
    # Total row
    f.write("-"*120 + "\n")
    f.write(f"{'TOTAL':<10}")
    f.write(f"{np.sum(avg_initial):<22.17f}")
    for ep in sorted(episode_results.keys()):
        f.write(f"{np.sum(episode_results[ep]['avg_beliefs']):<22.17f}")
    if has_target:
        f.write(f"{np.sum(natural_target):<22.17f}")
    f.write("\n")
    
    f.write("\n" + "="*120 + "\n")

print(f"Results saved to {OUTPUT_FILE}")
print(f"\nSummary:")
print(f"  - Analyzed {len(episode_results)} episodes")
print(f"  - Each episode compared with initial beliefs")
if has_target and episode_results:
    first_ep = min(episode_results.keys())
    last_ep = max(episode_results.keys())
    print(f"  - Distance from target: Episode {first_ep} = {episode_results[first_ep]['distance_from_target']:.17f}, Episode {last_ep} = {episode_results[last_ep]['distance_from_target']:.17f}")
else:
    print("  - Distance from target: N/A (natural belief target not found)")
