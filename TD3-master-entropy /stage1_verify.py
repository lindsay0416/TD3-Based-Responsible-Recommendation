#!/usr/bin/env python3
"""
Stage 1 verify: after `DATASETS="movies" BACKBONES="ENMF LightGCN NCL NGCF SGL" ...`
completes, run this from sweep root. Validates BOTH fixes:

  (a) actor not saturated (bug #1 fix verified)  -> per-dim std across input states > 0.05
  (b) ferl eval entropy std > 0.05               -> downstream pipeline produces real
                                                    variance, not degenerate constants
  (c) no mid-train JSON crash (bug #2 fix verified)  -> grep log for known crash signature

Exit 0 if all 5 movies cells PASS all three checks. Exit 1 otherwise.
"""
import sys
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    # Matches TD3.py FIX 2026-05-24 (A): 3-layer 256-hidden Fujimoto baseline.
    # Note: max_action default updated to 0.5 to match config.yaml FIX 2026-05-24 (B).
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


def check_actor(path):
    actor = Actor()
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    torch.manual_seed(42)
    states = torch.randn(20, 69) * 0.5
    with torch.no_grad():
        actions = actor(states).numpy()
    return float(np.linalg.norm(actions, axis=1).mean()), float(actions.std(axis=0).mean())


def check_ferl(path):
    with open(path) as f:
        d = json.load(f)
    m = d.get('metrics', d)
    ent = m.get('avg_entropy', {})
    return float(ent.get('mean', float('nan'))), float(ent.get('std', float('nan')))


def check_log(path):
    if not os.path.exists(path):
        return False, "log file not found"
    with open(path, errors='ignore') as f:
        content = f.read()
    if 'JSONDecodeError' in content or 'Expecting' in content and 'delimiter' in content:
        return False, "JSON parse crash detected in log"
    if 'Error during training' in content:
        return False, "training error detected in log"
    return True, "clean"


def main():
    backbones = ["ENMF", "LightGCN", "NCL", "NGCF", "SGL"]
    overall_pass = True
    print(f"{'cell':<25} {'actor_norm':>10} {'actor_std':>10} {'ferl_ent_mean':>14} {'ferl_ent_std':>13} {'log':>8} {'verdict':>10}")
    print("-" * 100)
    for bb in backbones:
        cell = f"movies/{bb}"
        # find latest final_actor
        actor_glob = f"results_table3_retrain/movies/{bb}/models/td3_recommendation_*_final_actor"
        actors = sorted(glob.glob(actor_glob), key=os.path.getmtime, reverse=True)
        actor_path = actors[0] if actors else None
        ferl_path = f"RL_evaluation_metrics_k10/movies/K_means/{bb}/ferl_results.json"
        log_path = f"logs/train_movies_{bb}.log"

        if actor_path is None:
            print(f"{cell:<25} {'--':>10} {'--':>10} {'--':>14} {'--':>13} {'--':>8} {'FAIL':>10}  (no final_actor -- train crashed)")
            overall_pass = False
            continue

        norm, std_act = check_actor(actor_path)
        if not os.path.exists(ferl_path):
            print(f"{cell:<25} {norm:>10.4f} {std_act:>10.6f} {'--':>14} {'--':>13} {'--':>8} {'FAIL':>10}  (no ferl JSON -- eval crashed)")
            overall_pass = False
            continue

        ent_mean, ent_std = check_ferl(ferl_path)
        log_ok, log_msg = check_log(log_path)

        # FIX 2026-05-24: max_action 1.0->0.5 changes saturation ceiling sqrt(action_dim)*max_action
        # from 8.0 to 4.0. Healthy actor norm should be well below the new ceiling.
        actor_pass = std_act > 0.05 and norm < 3.6  # not saturated (ceiling now 4.0)
        ferl_pass = ent_std > 0.05
        log_pass = log_ok

        verdict = "PASS" if (actor_pass and ferl_pass and log_pass) else "FAIL"
        if verdict == "FAIL":
            overall_pass = False
            reasons = []
            if not actor_pass: reasons.append(f"actor saturated (norm={norm:.2f}, std={std_act:.4f})")
            if not ferl_pass: reasons.append(f"ferl entropy degenerate (std={ent_std:.4f})")
            if not log_pass: reasons.append(f"log: {log_msg}")
            print(f"{cell:<25} {norm:>10.4f} {std_act:>10.6f} {ent_mean:>14.4f} {ent_std:>13.6f} {('ok' if log_ok else 'BAD'):>8} {verdict:>10}  " + "; ".join(reasons))
        else:
            print(f"{cell:<25} {norm:>10.4f} {std_act:>10.6f} {ent_mean:>14.4f} {ent_std:>13.6f} {'ok':>8} {verdict:>10}")

    print()
    if overall_pass:
        print("STAGE 1 PASS -- all 5 movies cells healthy. Safe to launch stage 2 (books + news, ~30-40h).")
        return 0
    else:
        print("STAGE 1 FAIL -- do NOT launch stage 2. Send actor checkpoint + train log of any FAIL cell back for diagnosis.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
