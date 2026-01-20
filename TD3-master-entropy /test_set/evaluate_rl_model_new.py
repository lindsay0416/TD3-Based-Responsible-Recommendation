#!/usr/bin/env python3

"""
Evaluate trained RL model on test set users.

Metrics (returned in JSON, aggregated over users):
1) avg_entropy: per-step entropy of user's recommendation list (diversity across 5 clusters),
                averaged over steps for each user, then averaged over users
2) avg_acceptance_rate: per-user acceptance rate (accepted / shown), averaged over users
3) precision: per-step Precision@K where K = len(Rec_u(t)), averaged over steps for each user, then averaged over users
4) recall: per-step Recall@K where K = len(Rec_u(t)), averaged over steps for each user, then averaged over users
5) ndcg: per-step NDCG@K (binary relevance) where K = len(Rec_u(t)), averaged over steps for each user, then averaged over users
6) avg_distance_improvement: per-user improvement in mean-absolute-distance to natural target
                             (initial_distance - final_distance), averaged over users

YOUR requested step relevance definition (binary, step-local):
- Rec_u(t) = env.get_recommendations(user, action)   # RS list + RL list merged in env
- accepted_t = items accepted in CURRENT step t only
- Rel_u(t) = test_set_ground_truth[user] ∪ accepted_t
- K = len(Rec_u(t))  (actual list length)

Input:
- processed_data/{dataset}/unique_test_users.json
- embeddings/{dataset}/Topk{topk}/test_set.csv
- Trained TD3 model (auto-detected)

Output:
- test_set/results/evaluation_results.json
"""

import os
import sys
import json
import numpy as np
import torch
import csv
from collections import Counter, defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TD3 import TD3
from recommendation_environment import RecommendationEnvironment


def find_latest_model():
    """Find the latest final model in models directory."""
    import glob

    final_models = glob.glob("models/td3_recommendation_*_final_actor")
    if final_models:
        final_models.sort(reverse=True)
        return final_models[0].replace("_actor", "")

    episode_models = glob.glob("models/td3_recommendation_*_episode_*_actor")
    if episode_models:
        episode_models.sort(reverse=True)
        return episode_models[0].replace("_actor", "")

    return None


def load_config():
    """Load configuration."""
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return resolve_data_paths(config)


def resolve_data_paths(config: dict) -> dict:
    """Resolve template placeholders in data_paths using experiment_params."""
    if "experiment_params" not in config:
        return config

    topk = config["experiment_params"].get("topk", 5)
    rs_model = config["experiment_params"].get("rs_model", "NCL")
    dataset = config["experiment_params"].get("dataset", "news")

    if "data_paths" in config:
        resolved_paths = {}
        for key, path_template in config["data_paths"].items():
            if isinstance(path_template, str):
                resolved_paths[key] = path_template.format(
                    topk=topk, rs_model=rs_model, dataset=dataset
                )
            else:
                resolved_paths[key] = path_template
        config["data_paths"] = resolved_paths

    return config


def load_test_users(config):
    """Load unique test user IDs."""
    dataset = config["experiment_params"]["dataset"]
    test_users_path = f"processed_data/{dataset}/unique_test_users.json"
    with open(test_users_path, "r") as f:
        data = json.load(f)
    return data["unique_user_ids"]


def load_test_set(config):
    """
    Load test set ground truth from test_set.csv.

    Returns:
        dict: {user_id: [item_ids]} mapping from user tokens to item tokens
    """
    dataset = config["experiment_params"]["dataset"]
    topk = config["experiment_params"]["topk"]
    test_set_path = f"embeddings/{dataset}/Topk{topk}/test_set.csv"

    print(f"Loading test set from: {test_set_path}")

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

            user_id = token_to_user.get(user_token)
            item_id = token_to_item.get(item_token)

            if user_id and item_id:
                test_set[user_id].append(item_id)

    print(f"Loaded test set: {len(test_set)} users, {sum(len(v) for v in test_set.values())} interactions")
    return dict(test_set)


# -----------------------------
# Metrics helpers
# -----------------------------
def calculate_entropy(cluster_counts, num_clusters=5):
    """Entropy of cluster distribution over a recommendation list."""
    total = sum(cluster_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for i in range(num_clusters):
        p = cluster_counts.get(i, 0) / total
        if p > 0:
            entropy += -p * np.log(p)
    return float(entropy)


def dcg_at_k_binary(recommended_items, relevant_set, k):
    """DCG@k for a single ranked list with binary relevance."""
    dcg = 0.0
    # rank positions are 1-based in the log denominator (standard)
    for r, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(r + 1)
    return float(dcg)


def idcg_at_k_binary(num_relevant, k):
    """Ideal DCG@k for binary relevance."""
    ideal_hits = min(num_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    return float(sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1)))


def ndcg_at_k_binary(recommended_items, relevant_set, k):
    """NDCG@k = DCG@k / IDCG@k."""
    idcg = idcg_at_k_binary(len(relevant_set), k)
    if idcg == 0.0:
        return 0.0
    return float(dcg_at_k_binary(recommended_items, relevant_set, k) / idcg)


def precision_recall_at_k(recommended_items, relevant_set, k):
    """Precision@k and Recall@k for a single ranked list."""
    if k <= 0:
        return 0.0, 0.0
    topk = recommended_items[:k]
    if not topk:
        return 0.0, 0.0

    hits = sum(1 for it in topk if it in relevant_set)
    precision = hits / len(topk)
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    return float(precision), float(recall)


def get_natural_target(env):
    """Get natural belief target vector from env."""
    if hasattr(env, "natural_belief_target") and env.natural_belief_target is not None:
        return np.array(env.natural_belief_target, dtype=np.float32)

    if hasattr(env, "optimal_beliefs_tensor") and env.optimal_beliefs_tensor is not None:
        try:
            return env.optimal_beliefs_tensor.detach().cpu().numpy().astype(np.float32)
        except Exception:
            return None

    return None


def mean_abs_distance(beliefs, target):
    """Mean absolute distance between belief vector and target vector."""
    b = np.array(beliefs, dtype=np.float32)
    t = np.array(target, dtype=np.float32)
    if b.shape != t.shape:
        return 0.0
    return float(np.mean(np.abs(b - t)))


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(env, td3_agent, test_user_ids, test_set_ground_truth, num_rounds=1000, batch_size=1000):
    """
    Per user:
      - run num_rounds steps
      - each step:
          Rec_u(t) = env.get_recommendations(user, action)
          accepted_t = env.simulate_user_interaction(...)
          Rel_u(t) = test_set[u] ∪ accepted_t
          K = len(Rec_u(t))
          compute entropy/precision/recall/ndcg for that step
      - per-user metrics are step-averages (except acceptance rate = accepted/shown)
      - distance improvement is initial_distance - final_distance (per user)
    Then aggregate across users (mean/std/min/max).
    """
    valid_test_users = [u for u in test_user_ids if u in env.users]
    print(f"Valid test users in environment: {len(valid_test_users)} / {len(test_user_ids)}")
    if not valid_test_users:
        print("No valid test users found!")
        return None

    natural_target = get_natural_target(env)
    if natural_target is None:
        print("Warning: No natural belief target found in environment; distance improvement will be 0.")
    else:
        print(f"Natural target found, shape={natural_target.shape}")

    # Per-user results (each is a scalar per user)
    users_entropy = []
    users_precision = []
    users_recall = []
    users_ndcg = []
    users_acceptance_rate = []
    users_distance_improvement = []

    num_batches = (len(valid_test_users) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(valid_test_users))
        batch_users = valid_test_users[start_idx:end_idx]

        for user_id in batch_users:
            # Reset beliefs to initial
            if user_id in env.initial_user_beliefs:
                env.users[user_id]["beliefs"] = env.initial_user_beliefs[user_id].copy()

            base_test_set = set(test_set_ground_truth.get(user_id, []))

            # Distance before
            if natural_target is not None:
                initial_distance = mean_abs_distance(env.users[user_id]["beliefs"], natural_target)
            else:
                initial_distance = 0.0

            # Per-step accumulators (for this user only)
            step_entropy_list = []
            step_precision_list = []
            step_recall_list = []
            step_ndcg_list = []

            accepted_count = 0
            shown_count = 0

            for _ in range(num_rounds):
                state = env.get_user_state(user_id)

                with torch.no_grad():
                    action = td3_agent.select_action(state)

                # Rec_u(t): RS + RL merged inside env
                recommended_items = env.get_recommendations(user_id, action)
                if not recommended_items:
                    continue

                # Simulate acceptance for CURRENT step
                accepted_items, rejected_items = env.simulate_user_interaction(user_id, recommended_items)

                # Rel_u(t) = test_set[u] ∪ accepted_t  (CURRENT step only!)
                relevant_set = base_test_set.union(set(accepted_items))

                # K = actual list length
                k = len(recommended_items)

                # ---- Entropy per step (on Rec list) ----
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

                # ---- Precision/Recall/NDCG per step ----
                p, r = precision_recall_at_k(recommended_items, relevant_set, k)
                n = ndcg_at_k_binary(recommended_items, relevant_set, k)

                step_precision_list.append(p)
                step_recall_list.append(r)
                step_ndcg_list.append(n)

                # ---- Acceptance rate (per user) ----
                accepted_count += len(accepted_items)
                shown_count += len(recommended_items)

                # Update beliefs
                env.update_user_beliefs_step(user_id, accepted_items, rejected_items, recommended_items)

            # Per-user averages
            user_entropy = float(np.mean(step_entropy_list)) if step_entropy_list else 0.0
            user_precision = float(np.mean(step_precision_list)) if step_precision_list else 0.0
            user_recall = float(np.mean(step_recall_list)) if step_recall_list else 0.0
            user_ndcg = float(np.mean(step_ndcg_list)) if step_ndcg_list else 0.0
            user_accept = float(accepted_count / shown_count) if shown_count > 0 else 0.0

            # Distance after
            if natural_target is not None:
                final_distance = mean_abs_distance(env.users[user_id]["beliefs"], natural_target)
            else:
                final_distance = 0.0
            user_dist_impr = float(initial_distance - final_distance)

            users_entropy.append(user_entropy)
            users_precision.append(user_precision)
            users_recall.append(user_recall)
            users_ndcg.append(user_ndcg)
            users_acceptance_rate.append(user_accept)
            users_distance_improvement.append(user_dist_impr)

    def summary(arr):
        if not arr:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        a = np.array(arr, dtype=np.float32)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    results = {
        "num_test_users": len(valid_test_users),
        "num_rounds_per_user": num_rounds,
        "K_eval": "len(Rec_u(t)) per step",
        "metrics": {
            "avg_entropy": {
                **summary(users_entropy),
                "description": "Per-user average of per-step entropy over Rec_u(t) cluster distribution; then aggregated over users.",
            },
            "avg_acceptance_rate": {
                **summary(users_acceptance_rate),
                "description": "Per-user acceptance rate = total_accepted / total_shown over all steps; then aggregated over users.",
            },
            "precision": {
                **summary(users_precision),
                "description": "Per-user average of per-step Precision@K with K=len(Rec_u(t)), Rel_u(t)=test_set[u] ∪ accepted_t; then aggregated over users.",
            },
            "recall": {
                **summary(users_recall),
                "description": "Per-user average of per-step Recall@K with K=len(Rec_u(t)), Rel_u(t)=test_set[u] ∪ accepted_t; then aggregated over users.",
            },
            "ndcg": {
                **summary(users_ndcg),
                "description": "Per-user average of per-step NDCG@K (binary) with K=len(Rec_u(t)), Rel_u(t)=test_set[u] ∪ accepted_t; then aggregated over users.",
            },
            "avg_distance_improvement": {
                **summary(users_distance_improvement),
                "description": "Per-user improvement in mean absolute distance to natural target: (initial_distance - final_distance); then aggregated over users.",
            },
        },
    }

    return results


def main():
    print("=" * 60)
    print("Evaluating RL Model on Test Set (RS+RL, step-local Rel)")
    print("=" * 60)

    config = load_config()

    test_user_ids = load_test_users(config)
    print(f"Total test users: {len(test_user_ids)}")

    test_set_ground_truth = load_test_set(config)
    print(f"Test set ground truth loaded: {len(test_set_ground_truth)} users")

    print("\nInitializing environment...")
    env = RecommendationEnvironment(config)

    state_dim = config["environment"]["state_dim"]
    action_dim = config["environment"]["action_dim"]
    max_action = config["environment"]["max_action"]

    td3_agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=config["td3"]["discount"],
        tau=config["td3"]["tau"],
        policy_noise=config["td3"]["policy_noise"],
        noise_clip=config["td3"]["noise_clip"],
        policy_freq=config["td3"]["policy_freq"],
    )

    model_path = find_latest_model()
    if model_path is None:
        print("Error: No model found in models/ directory!")
        return

    print(f"\nLoading model from: {model_path}")
    try:
        td3_agent.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nIf you see 'CUDA device but torch.cuda.is_available() is False', fix TD3.load by adding map_location.")
        print("In TD3.py, change torch.load(...) -> torch.load(..., map_location=device)")
        return

    print("\nStarting evaluation...")
    results = evaluate_model(
        env=env,
        td3_agent=td3_agent,
        test_user_ids=test_user_ids,
        test_set_ground_truth=test_set_ground_truth,
        num_rounds=50,
        batch_size=1000,
    )

    if results is None:
        print("Evaluation failed!")
        return

    os.makedirs("test_set/results", exist_ok=True)
    output_path = "test_set/results/evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Test users evaluated: {results['num_test_users']}")
    print(f"Rounds per user: {results['num_rounds_per_user']}")
    print(f"K_eval: {results['K_eval']}")
    print()

    for key in ["avg_entropy", "avg_acceptance_rate", "precision", "recall", "ndcg", "avg_distance_improvement"]:
        m = results["metrics"][key]
        print(f"{key}: mean={m['mean']:.6f}, std={m['std']:.6f}, range=[{m['min']:.6f}, {m['max']:.6f}]")


if __name__ == "__main__":
    main()
