#!/usr/bin/env python3
"""
Table 3 rerun: re-evaluate FERL with k=10 (fixed) AND evaluate base recommender
through the same simulator under matched protocol.

Two changes vs evaluate_rl_model_new.py:
  1. --k_eval N  (default 10):  metric is computed on slate[:k_eval], not on
                                 the full slate (which was the source of the
                                 mislabelled "Recall@10").
  2. --agent_mode {ferl, base_only}:
       ferl       -> TD3 actor produces virtual_item -> env.get_recommendations
                     gives RL list merged with the pretrained list (existing
                     flow, unchanged).
       base_only  -> skip TD3 entirely; slate is the pretrained recommender's
                     pre-computed top-N list (env.pretrained_recommendations),
                     fed directly into the simulator's accept/reject loop.

Usage (run from TD3-master-entropy/ root, same as the original evaluate_rl_model_new.py):

    # FERL with the k=10 fix:
    python test_set/evaluate_rl_model_rerun.py \
        --agent_mode ferl --k_eval 10 \
        --output_path RL_evaluation_metrics_k10/${dataset}/Topk5/${backbone}/ferl_results.json

    # Base recommender in simulator (no RL):
    python test_set/evaluate_rl_model_rerun.py \
        --agent_mode base_only --k_eval 10 \
        --output_path RL_evaluation_metrics_k10/${dataset}/Topk5/${backbone}/base_only_results.json

The dataset / backbone are picked up from config.yaml (existing convention).
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
import torch
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TD3.load() in the IJCAI codebase calls torch.load() without map_location,
# which fails when loading a CUDA-saved checkpoint on a CPU-only host. Patch
# torch.load globally so it always picks the right device. On a 4090 host
# (CUDA available) this is a no-op.
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_orig_torch_load = torch.load
def _torch_load_with_map(*args, **kwargs):
    kwargs.setdefault('map_location', _DEVICE)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_with_map

from TD3 import TD3
from recommendation_environment import RecommendationEnvironment


# --------------------------------------------------------------------------
# Config / data loading (cloned from evaluate_rl_model_new.py with no changes)
# --------------------------------------------------------------------------
def load_config(dataset_override=None, rs_model_override=None, topk_override=None,
                clustering_override=None):
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if "experiment_params" not in config:
        config["experiment_params"] = {}

    ep = config["experiment_params"]
    old_dataset = str(ep.get("dataset", "news"))
    old_rs_model = str(ep.get("rs_model", "LightGCN"))
    old_clustering = str(ep.get("clustering", "K_means"))
    old_topk = str(ep.get("topk", 5))

    new_dataset = dataset_override if dataset_override else old_dataset
    new_rs_model = rs_model_override if rs_model_override else old_rs_model
    new_clustering = clustering_override if clustering_override else old_clustering
    new_topk = str(topk_override) if topk_override is not None else old_topk

    ep["dataset"] = new_dataset
    ep["rs_model"] = new_rs_model
    ep["clustering"] = new_clustering
    try:
        ep["topk"] = int(new_topk)
    except (TypeError, ValueError):
        ep["topk"] = new_topk

    return resolve_data_paths(
        config,
        new_dataset=new_dataset, old_dataset=old_dataset,
        new_rs_model=new_rs_model, old_rs_model=old_rs_model,
        new_clustering=new_clustering, old_clustering=old_clustering,
        new_topk=new_topk, old_topk=old_topk,
    )


def resolve_data_paths(config, new_dataset=None, old_dataset=None,
                       new_rs_model=None, old_rs_model=None,
                       new_clustering=None, old_clustering=None,
                       new_topk=None, old_topk=None):
    if "data_paths" not in config:
        return config

    # Fallbacks if called the old way (no overrides).
    if new_dataset is None:
        ep = config.get("experiment_params", {})
        new_dataset = str(ep.get("dataset", "news"))
        new_rs_model = str(ep.get("rs_model", "LightGCN"))
        new_clustering = str(ep.get("clustering", "K_means"))
        new_topk = str(ep.get("topk", 5))
        old_dataset = new_dataset
        old_rs_model = new_rs_model
        old_clustering = new_clustering
        old_topk = new_topk

    for k, v in config["data_paths"].items():
        if not isinstance(v, str):
            continue
        # Pass 1: placeholder substitution (if the yaml uses {dataset} etc).
        v2 = (v.replace("{topk}", str(new_topk))
               .replace("{rs_model}", new_rs_model)
               .replace("{dataset}", new_dataset)
               .replace("{clustering}", new_clustering))
        # Pass 2: if the yaml is hardcoded (no placeholders), fall back to
        # substring substitution of the OLD experiment_params values. This is
        # how sweeps work when config.yaml has paths like
        # "embeddings/books/K_means/LightGCN/..." rather than
        # "embeddings/{dataset}/{clustering}/{rs_model}/...".
        if v2 == v:
            if new_dataset != old_dataset and old_dataset:
                v2 = v2.replace(f"/{old_dataset}/", f"/{new_dataset}/")
            if new_rs_model != old_rs_model and old_rs_model:
                v2 = v2.replace(f"/{old_rs_model}/", f"/{new_rs_model}/")
            if new_clustering != old_clustering and old_clustering:
                v2 = v2.replace(f"/{old_clustering}/", f"/{new_clustering}/")
        config["data_paths"][k] = v2
    return config


def find_latest_model():
    import glob
    # Search in both models/ and results/*/models/ directories
    search_patterns = [
        "models/td3_recommendation_*_final_actor",
        "results/*/models/td3_recommendation_*_final_actor",
        "results/*/*/models/td3_recommendation_*_final_actor",
    ]
    for pattern in search_patterns:
        final_models = glob.glob(pattern)
        if final_models:
            final_models.sort(reverse=True)
            return final_models[0].replace("_actor", "")
    
    # Try episode checkpoints
    episode_patterns = [
        "models/td3_recommendation_*_episode_*_actor",
        "results/*/models/td3_recommendation_*_episode_*_actor",
        "results/*/*/models/td3_recommendation_*_episode_*_actor",
    ]
    for pattern in episode_patterns:
        episode_models = glob.glob(pattern)
        if episode_models:
            episode_models.sort(reverse=True)
            return episode_models[0].replace("_actor", "")
    return None


def load_test_users(config):
    dataset = config["experiment_params"]["dataset"]
    path = f"processed_data/{dataset}/unique_test_users.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["unique_user_ids"]


def load_test_set(config):
    dataset = config["experiment_params"]["dataset"]
    topk = config["experiment_params"]["topk"]
    test_set_path = f"embeddings/{dataset}/Topk{topk}/test_set.csv"

    user_token_map_path = f"embeddings/{dataset}/Topk{topk}/user_token_map.json"
    item_token_map_path = f"embeddings/{dataset}/Topk{topk}/item_token_map.json"

    with open(user_token_map_path, "r") as f:
        user_token_map = json.load(f)
    with open(item_token_map_path, "r") as f:
        item_token_map = json.load(f)

    token_to_user = {str(token): user_id for user_id, token in user_token_map.items()}
    token_to_item = {str(token): item_id for item_id, token in item_token_map.items()}

    test_set = defaultdict(list)
    with open(test_set_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_token = str(row["user_id"])
            item_token = str(row["item_id"])
            if user_token in token_to_user and item_token in token_to_item:
                test_set[token_to_user[user_token]].append(token_to_item[item_token])
    return test_set


# --------------------------------------------------------------------------
# Metrics (cloned)
# --------------------------------------------------------------------------
def calculate_entropy(cluster_counts, num_clusters=5):
    if not cluster_counts:
        return 0.0
    total = sum(cluster_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for k in range(num_clusters):
        p = cluster_counts.get(k, 0) / total
        if p > 0:
            entropy -= p * np.log(p)
    return float(entropy)


def precision_recall_at_k(recommended_items, relevant_set, k):
    topk = recommended_items[:k]
    if not topk:
        return 0.0, 0.0
    hits = sum(1 for it in topk if it in relevant_set)
    precision = hits / len(topk)
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    return float(precision), float(recall)


def ndcg_at_k_binary(recommended_items, relevant_set, k):
    topk = recommended_items[:k]
    if not topk or not relevant_set:
        return 0.0
    dcg = 0.0
    for i, it in enumerate(topk):
        if it in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_abs_distance(beliefs, target):
    return float(np.mean(np.abs(np.asarray(beliefs) - np.asarray(target))))


def get_natural_target(env):
    # Try common attribute names without forcing a specific structure.
    for attr in ("natural_target", "target_belief", "natural_belief_target"):
        if hasattr(env, attr):
            t = getattr(env, attr)
            if t is not None:
                return np.asarray(t)
    return None


# --------------------------------------------------------------------------
# Base-only slate provider (replaces env.get_recommendations for that mode)
# --------------------------------------------------------------------------
def build_base_only_provider(env):
    """
    Returns a function: get_slate(user_id) -> List[item_id]
    Uses env.pretrained_recommendations (the base recommender's pre-computed
    top-N list) directly, no RL.
    """
    token_to_item = {str(token): item_id for item_id, token in env.item_token_map.items()}
    slate_size = env.config['recommendation'].get('total_recommendations', 100)

    def get_slate(user_id):
        numeric_token = str(env.user_token_map.get(user_id, -1))
        pretrained_token_recs = env.pretrained_recommendations.get(numeric_token, [])
        items = []
        for token in pretrained_token_recs:
            tok_str = str(token)
            if tok_str in token_to_item:
                items.append(token_to_item[tok_str])
            if len(items) >= slate_size:
                break
        return items

    return get_slate


# --------------------------------------------------------------------------
# Evaluation loop (modified: k_eval + agent_mode)
# --------------------------------------------------------------------------
def evaluate_model(env, td3_agent, test_user_ids, test_set_ground_truth,
                   num_rounds, batch_size, k_eval, agent_mode):
    valid_test_users = [u for u in test_user_ids if u in env.users]
    print(f"Valid test users: {len(valid_test_users)} / {len(test_user_ids)}")
    if not valid_test_users:
        return None

    natural_target = get_natural_target(env)
    base_only_get_slate = build_base_only_provider(env) if agent_mode == "base_only" else None

    users_entropy, users_precision, users_recall, users_ndcg = [], [], [], []
    users_acceptance_rate, users_distance_improvement = [], []

    num_batches = (len(valid_test_users) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating ({agent_mode})"):
        batch_users = valid_test_users[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        for user_id in batch_users:
            if user_id in env.initial_user_beliefs:
                env.users[user_id]["beliefs"] = env.initial_user_beliefs[user_id].copy()

            base_test_set = set(test_set_ground_truth.get(user_id, []))

            if natural_target is not None:
                initial_distance = mean_abs_distance(env.users[user_id]["beliefs"], natural_target)
            else:
                initial_distance = 0.0

            step_entropy_list, step_precision_list = [], []
            step_recall_list, step_ndcg_list = [], []
            accepted_count, shown_count = 0, 0

            for _ in range(num_rounds):
                # ---- get slate (mode-dependent) ----
                if agent_mode == "ferl":
                    state = env.get_user_state(user_id)
                    with torch.no_grad():
                        action = td3_agent.select_action(state)
                    recommended_items = env.get_recommendations(user_id, action)
                else:  # base_only
                    recommended_items = base_only_get_slate(user_id)

                if not recommended_items:
                    continue

                # ---- simulator step ----
                accepted_items, rejected_items = env.simulate_user_interaction(user_id, recommended_items)
                relevant_set = base_test_set.union(set(accepted_items))

                # ---- entropy on full slate (unchanged) ----
                cluster_counts = Counter()
                for item_id in recommended_items:
                    item_id_str = str(item_id)
                    if hasattr(env, "item_id_normalized") and env.item_id_normalized:
                        normalized_id = env.item_id_normalized.get(item_id_str, item_id_str)
                    else:
                        normalized_id = item_id_str
                    if hasattr(env, "item_to_cluster") and env.item_to_cluster:
                        cluster = env.item_to_cluster.get(normalized_id, None)
                        if cluster is not None:
                            cluster_counts[cluster] += 1
                step_entropy_list.append(calculate_entropy(cluster_counts, num_clusters=5))

                # ---- ranking metrics (THIS is the k_eval fix) ----
                p, r = precision_recall_at_k(recommended_items, relevant_set, k_eval)
                n = ndcg_at_k_binary(recommended_items, relevant_set, k_eval)
                step_precision_list.append(p)
                step_recall_list.append(r)
                step_ndcg_list.append(n)

                accepted_count += len(accepted_items)
                shown_count += len(recommended_items)

                env.update_user_beliefs_step(user_id, accepted_items, rejected_items, recommended_items)

            users_entropy.append(float(np.mean(step_entropy_list)) if step_entropy_list else 0.0)
            users_precision.append(float(np.mean(step_precision_list)) if step_precision_list else 0.0)
            users_recall.append(float(np.mean(step_recall_list)) if step_recall_list else 0.0)
            users_ndcg.append(float(np.mean(step_ndcg_list)) if step_ndcg_list else 0.0)
            users_acceptance_rate.append(float(accepted_count / shown_count) if shown_count > 0 else 0.0)

            if natural_target is not None:
                final_distance = mean_abs_distance(env.users[user_id]["beliefs"], natural_target)
            else:
                final_distance = 0.0
            users_distance_improvement.append(float(initial_distance - final_distance))

    def stat(arr):
        a = np.asarray(arr) if arr else np.zeros(1)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    return {
        "num_test_users": len(users_recall),
        "num_rounds_per_user": num_rounds,
        "k_eval": k_eval,
        "agent_mode": agent_mode,
        "metrics": {
            "avg_entropy": stat(users_entropy),
            "avg_acceptance_rate": stat(users_acceptance_rate),
            "precision": stat(users_precision),
            "recall": stat(users_recall),
            "ndcg": stat(users_ndcg),
            "avg_distance_improvement": stat(users_distance_improvement),
        },
    }


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_mode", choices=["ferl", "base_only"], default="ferl",
                   help="ferl: TD3 + base list merged (existing flow). base_only: pretrained list only.")
    p.add_argument("--k_eval", type=int, default=10,
                   help="Top-k for Recall/NDCG/Precision (default 10).")
    p.add_argument("--num_rounds", type=int, default=50,
                   help="Rounds per user (default 50; original main used 1000 -- use whichever matches your headline run).")
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--output_path", type=str, default="test_set/results/rerun_results.json")
    p.add_argument("--model_path", type=str, default=None,
                   help="Override TD3 model path. Default: auto-detect latest in models/. Ignored for base_only.")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override config.yaml experiment_params.dataset (e.g. news|books|movies). "
                        "Useful for sweep without editing config.yaml between runs.")
    p.add_argument("--rs_model", type=str, default=None,
                   help="Override config.yaml experiment_params.rs_model (e.g. LightGCN|NCL|NGCF|ENMF|SGL).")
    p.add_argument("--topk", type=int, default=None,
                   help="Override config.yaml experiment_params.topk (default 5).")
    args = p.parse_args()

    print("=" * 60)
    print(f"Table 3 rerun: agent_mode={args.agent_mode}, k_eval={args.k_eval}, num_rounds={args.num_rounds}")
    print("=" * 60)

    config = load_config(
        dataset_override=args.dataset,
        rs_model_override=args.rs_model,
        topk_override=args.topk,
    )
    print(f"experiment_params: dataset={config['experiment_params'].get('dataset')}, "
          f"rs_model={config['experiment_params'].get('rs_model')}, "
          f"topk={config['experiment_params'].get('topk')}")
    test_user_ids = load_test_users(config)
    test_set_ground_truth = load_test_set(config)
    print(f"Test users: {len(test_user_ids)}, test_set users: {len(test_set_ground_truth)}")

    env = RecommendationEnvironment(config)

    td3_agent = None
    if args.agent_mode == "ferl":
        state_dim = config["environment"]["state_dim"]
        action_dim = config["environment"]["action_dim"]
        max_action = config["environment"]["max_action"]
        td3_agent = TD3(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action,
            discount=config["td3"]["discount"], tau=config["td3"]["tau"],
            policy_noise=config["td3"]["policy_noise"], noise_clip=config["td3"]["noise_clip"],
            policy_freq=config["td3"]["policy_freq"],
        )
        model_path = args.model_path or find_latest_model()
        if model_path is None:
            print("ERROR: no model found; specify --model_path or place a checkpoint in models/.")
            sys.exit(1)
        print(f"Loading TD3 model: {model_path}")
        td3_agent.load(model_path)

    results = evaluate_model(
        env=env,
        td3_agent=td3_agent,
        test_user_ids=test_user_ids,
        test_set_ground_truth=test_set_ground_truth,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        k_eval=args.k_eval,
        agent_mode=args.agent_mode,
    )

    if results is None:
        print("Evaluation failed.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_path}")
    print("=" * 60)
    print(f"Test users evaluated: {results['num_test_users']}")
    print(f"Rounds per user: {results['num_rounds_per_user']}")
    print(f"k_eval: {results['k_eval']}  agent_mode: {results['agent_mode']}")
    print()
    for key in ["avg_entropy", "avg_acceptance_rate", "precision", "recall", "ndcg", "avg_distance_improvement"]:
        m = results["metrics"][key]
        print(f"{key}: mean={m['mean']:.6f}, std={m['std']:.6f}, range=[{m['min']:.6f}, {m['max']:.6f}]")


if __name__ == "__main__":
    main()
