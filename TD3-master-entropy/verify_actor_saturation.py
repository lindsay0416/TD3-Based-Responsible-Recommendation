#!/usr/bin/env python3
"""
Single-actor saturation check. Use after partial smoke (e.g., training only
reached episode_10) to verify the fix is working without waiting for full
50-episode completion.

Usage:
    # auto-find latest movies/ENMF actor (final or episode_*):
    python verify_actor_saturation.py

    # explicit path:
    python verify_actor_saturation.py results_table3_retrain/movies/ENMF/models/td3_recommendation_20260525_120000_episode_10_actor

PASS criterion (post FIX 2026-05-24 with max_action=0.5):
    per-dim std (across 20 random input states) > 0.05
    action norm well below saturation ceiling sqrt(action_dim)*max_action = sqrt(64)*0.5 = 4.0

Exit codes:
    0 PASS
    1 FAIL (saturated)
    2 PARTIAL (some variation but borderline)
"""
import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    # Matches TD3.py FIX 2026-05-24 (A): 3-layer 256-hidden Fujimoto baseline.
    # Default max_action 0.5 matches config_template.yaml FIX 2026-05-24 (B).
    def __init__(self, state_dim=69, action_dim=64, max_action=0.5):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


def find_latest_actor():
    # Prefer final_actor, fall back to highest episode_N actor.
    finals = sorted(
        glob.glob("results_table3_retrain/movies/ENMF/models/td3_recommendation_*_final_actor"),
        key=os.path.getmtime, reverse=True,
    )
    if finals:
        return finals[0]
    episodes = sorted(
        glob.glob("results_table3_retrain/movies/ENMF/models/td3_recommendation_*_episode_*_actor"),
        key=os.path.getmtime, reverse=True,
    )
    return episodes[0] if episodes else None


def check_actor(path):
    print(f"checkpoint: {path}")
    actor = Actor()
    state_dict = torch.load(path, map_location="cpu")
    actor.load_state_dict(state_dict)
    actor.eval()

    torch.manual_seed(42)
    states = torch.randn(20, 69) * 0.5
    with torch.no_grad():
        actions = actor(states).numpy()

    norms = np.linalg.norm(actions, axis=1)
    per_dim_std = actions.std(axis=0)
    sat_ceiling = np.sqrt(64) * actor.max_action  # 4.0 with max_action=0.5

    print(f"  action_dim          : {actions.shape[1]}")
    print(f"  saturation ceiling  : {sat_ceiling:.4f}  (sqrt(action_dim) * max_action)")
    print(f"  mean action norm    : {norms.mean():.4f}")
    print(f"  min/max action norm : {norms.min():.4f} / {norms.max():.4f}")
    print(f"  per-dim std (across 20 input states):")
    print(f"      mean = {per_dim_std.mean():.6f}")
    print(f"      max  = {per_dim_std.max():.6f}")
    print(f"      min  = {per_dim_std.min():.6f}")
    print()

    if per_dim_std.mean() > 0.05 and norms.mean() < 0.9 * sat_ceiling:
        print("PASS: actor responds to input state, not saturated.")
        return 0
    elif per_dim_std.mean() > 0.01:
        print("PARTIAL: actor varies with input but is borderline. May still produce useful results.")
        return 2
    else:
        print("FAIL: actor is saturated, outputs are nearly constant across input states.")
        print("      The fix did not resolve the collapse. Send actor + train log back for deeper diagnosis.")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_actor()
        if path is None:
            print("No actor checkpoint found under results_table3_retrain/movies/ENMF/models/")
            print("Pass an explicit checkpoint path or run smoke train first.")
            sys.exit(1)
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        sys.exit(1)
    sys.exit(check_actor(path))
