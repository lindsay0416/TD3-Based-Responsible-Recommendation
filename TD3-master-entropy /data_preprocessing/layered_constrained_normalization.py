#!/usr/bin/env python3
"""
Layered constrained normalization: Ensure user beliefs don't exceed reasonable proportions of target beliefs
Solve logical contradictions where user beliefs exceed optimal beliefs
"""

import json
import numpy as np
from calculate_belief_central_distances import BeliefCentralDistanceCalculator

def constrained_belief_scaling(beliefs, optimal_beliefs, target_scale=10.0):
    """
    Constrained belief scaling: Scale user beliefs to reasonable numerical range while satisfying constraints
    
    Core idea:
    1. Maintain user's relative preference ordering
    2. Scale values to reasonable range (not forcing sum to 1)
    3. Ensure not exceeding optimal constraints
    4. Make acceptance rate function meaningful
    
    Args:
        beliefs: numpy array or list of belief values (user's current beliefs)
        optimal_beliefs: numpy array of optimal belief distribution (upper bound constraints)
        target_scale: target scaling factor, scale strongest preference to this proportion
        
    Returns:
        numpy array of scaled beliefs (all <= optimal, meaningful range)
    """
    beliefs = np.array(beliefs, dtype=float)
    optimal_beliefs = np.array(optimal_beliefs, dtype=float)
    
    if len(beliefs) == 0:
        return beliefs
    
    # Handle all-zero case
    if beliefs.sum() == 0:
        # If user has no preferences, return small uniform values
        return np.ones_like(beliefs) * 0.01
    
    # 1. Find user's strongest preference
    max_belief = beliefs.max()
    max_idx = np.argmax(beliefs)
    
    # 2. Calculate scaling factor, ensure strongest preference doesn't exceed optimal constraint
    max_allowed = optimal_beliefs[max_idx]
    
    # Goal: scale strongest preference to target_scale% of optimal, but not exceeding optimal
    target_value = min(max_allowed * (target_scale / 100.0), max_allowed)
    
    if max_belief > 0:
        scale_factor = target_value / max_belief
    else:
        scale_factor = 1.0
    
    # 3. Apply scaling
    scaled_beliefs = beliefs * scale_factor
    
    # 4. Ensure all values don't exceed corresponding optimal constraints
    final_beliefs = np.minimum(scaled_beliefs, optimal_beliefs)
    
    return final_beliefs

def apply_layered_normalization_to_users():
    """
    Apply layered constrained normalization to all users
    """
    print("Loading user beliefs from original backup...")
    
    # Load data from original backup
    try:
        with open('processed_data/user_average_beliefs_original.json', 'r') as f:
            user_data = json.load(f)
        print("Loaded from original backup (pre-normalization)")
    except FileNotFoundError:
        print("Original backup not found, using current file")
        with open('processed_data/user_average_beliefs.json', 'r') as f:
            user_data = json.load(f)
    
    # Load optimal beliefs
    with open('processed_data/natural_belief_target.json', 'r') as f:
        target_data = json.load(f)
    optimal_beliefs = np.array(target_data['natural_belief_target'])
    
    print(f"Loaded {len(user_data)} users")
    print(f"Optimal beliefs: {optimal_beliefs}")
    
    # Show sample original data
    sample_user = list(user_data.keys())[0]
    original_beliefs = user_data[sample_user]['beliefs']
    print(f"\nSample original beliefs ({sample_user}): {original_beliefs}")
    print(f"Original sum: {sum(original_beliefs):.6f}")
    
    # Apply constrained belief scaling
    print("\nApplying constrained belief scaling...")
    print("Method: scale user preferences to 80% of optimal maximum, ensuring all <= optimal")
    
    updated_count = 0
    belief_calculator = BeliefCentralDistanceCalculator()
    belief_calculator.load_optimal_beliefs()
    
    for user_id, user_info in user_data.items():
        original_beliefs = np.array(user_info['beliefs'])
        
        # Apply constrained belief scaling
        constrained_beliefs = constrained_belief_scaling(original_beliefs, optimal_beliefs, target_scale=80.0)
        
        # Update beliefs
        user_info['beliefs'] = constrained_beliefs.tolist()
        
        # Note: cluster_distances are fixed target distances, should not be modified!
        # Only recalculate PP1 distance (user belief center to target center distance)
        pp1_distance = belief_calculator.calculate_pp1_distance(constrained_beliefs.tolist())
        user_info['pp1_distance'] = pp1_distance
        
        updated_count += 1
        
        if updated_count % 10000 == 0:
            print(f"  Processed {updated_count} users...")
    
    print(f"Updated {updated_count} users with layered constrained normalization")
    
    # Show sample processed data
    new_beliefs = user_data[sample_user]['beliefs']
    print(f"\nSample constrained beliefs ({sample_user}): {new_beliefs}")
    print(f"New sum: {sum(new_beliefs):.6f}")
    print(f"New range: [{min(new_beliefs):.3f}, {max(new_beliefs):.3f}]")
    
    # Verify constrained scaling effect
    print(f"\nConstrained scaling verification for {sample_user}:")
    for i, (user_belief, optimal_belief) in enumerate(zip(new_beliefs, optimal_beliefs)):
        constraint_ok = user_belief <= optimal_belief + 1e-6
        ratio = (user_belief / optimal_belief * 100) if optimal_belief > 0 else 0
        print(f"  Cluster {i}: {user_belief:.6f} <= {optimal_belief:.6f} ({'✅' if constraint_ok else '❌'}) ({ratio:.1f}% of optimal)")
    print(f"  Sum: {sum(new_beliefs):.6f} (not required to be 1.0)")
    
    # Calculate overall statistics
    all_beliefs = []
    all_sums = []
    
    for user_info in user_data.values():
        beliefs = user_info['beliefs']
        all_beliefs.extend(beliefs)
        all_sums.append(sum(beliefs))
    
    all_beliefs = np.array(all_beliefs)
    all_sums = np.array(all_sums)
    
    print(f"\nOverall statistics:")
    print(f"  Belief values - Mean: {all_beliefs.mean():.3f}, Std: {all_beliefs.std():.3f}")
    print(f"  Belief values - Min: {all_beliefs.min():.3f}, Max: {all_beliefs.max():.3f}")
    print(f"  Sum constraint - Mean: {all_sums.mean():.6f}, Std: {all_sums.std():.6f}")
    # Count constraint satisfaction
    constraint_violations = 0
    total_beliefs = 0
    belief_ranges = []
    
    for user_info in user_data.values():
        beliefs = user_info['beliefs']
        for i, belief in enumerate(beliefs):
            total_beliefs += 1
            if belief > optimal_beliefs[i] + 1e-6:
                constraint_violations += 1
            if belief > 0:
                belief_ranges.append(belief)
    
    belief_ranges = np.array(belief_ranges)
    print(f"  Constraint violations: {constraint_violations} out of {total_beliefs} beliefs")
    print(f"  Violation rate: {constraint_violations/total_beliefs*100:.2f}%")
    print(f"  Non-zero belief range: [{belief_ranges.min():.6f}, {belief_ranges.max():.6f}]")
    
    # Save updated data
    print("\nSaving updated user beliefs...")
    
    # Backup current file
    import shutil
    try:
        shutil.copy('processed_data/user_average_beliefs.json', 'processed_data/user_average_beliefs_before_layered.json')
        print("Current file backed up as user_average_beliefs_before_layered.json")
    except:
        pass
    
    # Save updated file
    with open('processed_data/user_average_beliefs.json', 'w') as f:
        json.dump(user_data, f, indent=2)
    
    print("Updated user beliefs saved!")
    
    # Show comparison and acceptance rate analysis for first few users
    print("\nComparison for first 3 users:")
    for i, user_id in enumerate(list(user_data.keys())[:3]):
        beliefs = user_data[user_id]['beliefs']
        belief_sum = sum(beliefs)
        max_belief = max(beliefs)
        max_cluster = beliefs.index(max_belief)
        
        print(f"  {user_id}: {[f'{b:.3f}' for b in beliefs]} (sum={belief_sum:.3f})")
        print(f"    Strongest preference: {max_belief:.3f} for cluster {max_cluster}")
        
        # Show acceptance rate range
        min_rate = 0.0 * max_belief + 0.1  # least similar
        max_rate = 1.0 * max_belief + 0.1  # most similar
        print(f"    Acceptance rate range: [{min_rate:.3f}, {max_rate:.3f}]")
    
    return user_data

if __name__ == "__main__":
    # Test single user normalization
    print("=== Testing single user normalization ===")
    
    # Test case: extreme preference user
    test_beliefs = [0.0, 0.0, 0.0, 0.0, 0.0006257822277847309]
    optimal = [0.69958181, 0.07083116, 0.07386304, 0.07219028, 0.08353372]
    
    print(f"Original beliefs: {test_beliefs}")
    print(f"Sum: {sum(test_beliefs):.6f}")
    
    result = constrained_belief_scaling(test_beliefs, optimal, target_scale=80.0)
    
    print(f"Normalized beliefs: {result}")
    print(f"Sum: {sum(result):.6f}")
    
    # Verify constrained belief scaling
    print("Constrained belief scaling verification:")
    for i, (belief, opt_belief) in enumerate(zip(result, optimal)):
        constraint_ok = belief <= opt_belief + 1e-6
        print(f"  Cluster {i}: {belief:.6f} <= {opt_belief:.6f} ({'✅' if constraint_ok else '❌'})")
    print(f"  Sum: {sum(result):.6f} (not required to be 1.0)")
    
    # Show acceptance rate function effects
    print("\nAcceptance rate function analysis:")
    print("Formula: (1 - current_distance/max_distance) * user_belief + 0.1")
    for i, belief in enumerate(result):
        if belief > 0:
            # Assume distance factor of 0.5 (medium similarity)
            distance_factor = 0.5
            acceptance_rate = distance_factor * belief + 0.1
            print(f"  Cluster {i}: {distance_factor} * {belief:.3f} + 0.1 = {acceptance_rate:.3f}")
        else:
            print(f"  Cluster {i}: No preference, rate = 0.1")
    
    print("\n" + "="*50)
    
    # Apply to all users
    updated_data = apply_layered_normalization_to_users()
    print("\nConstrained belief scaling completed successfully!")