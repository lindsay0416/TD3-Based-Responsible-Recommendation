#!/usr/bin/env python3
"""
termination counts
"""
import json
from pathlib import Path

def quick_check():
    stats_file = Path("results/training_all_episodes.json")
    
    if not stats_file.exists():
        print("No training data")
        print("Start Training: python run_recommendation_rl.py")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("="*60)
    print("TERMINATION REWARD")
    print("="*60)
    
    for episode in stats:
        episode_num = episode['episode']
        total_terminations = episode.get('total_termination_bonuses', 0)
        rounds = episode['rounds']
        
        print(f"\nEpisode {episode_num}:")
        print(f"  total terminations: {total_terminations:,} users")
        print(f"  total users counts:   {len(rounds) * 50000:,}")
        print(f"  achievement Rate:     {total_terminations / (len(rounds) * 50000) * 100:.2f}%")
        
        print(f"\n  First 5 rounds:")
        for i in range(min(5, len(rounds))):
            r = rounds[i]
            print(f"    Round {r['round']:2d}: {r.get('termination_bonuses', 0):,} users")
        
        if len(rounds) > 5:
            print(f"\n  Last 5 rounds:")
            for i in range(max(0, len(rounds)-5), len(rounds)):
                r = rounds[i]
                print(f"    Round {r['round']:2d}: {r.get('termination_bonuses', 0):,} users")
    
    print("\n" + "="*60)
    print("Use `python analyze_termination_stats.py` view detailed analysis")
    print("="*60)

if __name__ == "__main__":
    quick_check()
