#!/usr/bin/env python3
"""
Recommendation Environment for RL Training
Simulates user interactions with recommendation system using TD3.
"""

import numpy as np
import json
import pickle
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import random
# Removed BeliefCentralDistanceCalculator import - using step-based rewards now


class RecommendationEnvironment:
    """
    Custom environment for recommendation system RL training.
    Manages user states, item recommendations, and belief updates.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config['experiment']['device'] != "cpu" 
            else "cpu"
        )
        
        # GPU tensors for batch processing
        self.item_embeddings_tensor = None
        self.user_embeddings_tensor = None
        self.enhanced_embeddings_tensor = None
        
        # GPU tensors for reward calculation
        self.optimal_beliefs_tensor = None
        self.user_beliefs_tensor = None
        self.previous_beliefs_tensor = None
        
        # Initialize tracking variables first
        self.current_episode = 0
        self.current_round = 0
        self.user_interactions = {}  # Track user interactions per episode
        self.previous_distances = {}  # Track previous episode distances for reward calculation
        self.initial_user_beliefs = {}  # Store initial beliefs for episode reset
        self.user_accepted_counts = {}  # Track accepted items count per cluster for belief calculation
        self.initial_user_counts = {}  # Store initial counts for episode reset
        
        # Note: Using step-based reward system, no need for separate belief calculator
        
        # Initialize storage for previous beliefs (for reward calculation)
        self.previous_beliefs = {}
        
        # Load data
        self._load_data()
        
        # Load natural belief target for constraints
        self._load_natural_belief_target()
        
        # Initialize GPU tensors for batch processing
        self._initialize_gpu_tensors()
        
        print(f"Environment initialized with {len(self.users)} users")
        print(f"Item embeddings: {len(self.item_embeddings)} items")
        print(f"Cluster embeddings: {len(self.cluster_embeddings)} clusters")
        print(f"Enhanced item embeddings: {len(self.enhanced_item_embeddings)} items")
        print(f"Pretrained recommendations: {len(self.pretrained_recommendations)} users (using numeric tokens)")
        print(f"User token mapping: {len(self.user_token_map)} users")
        print(f"Device: {self.device}")
        
        # Sample mapping verification
        sample_user = list(self.users.keys())[0]
        sample_item = list(self.enhanced_item_embeddings.keys())[0]
        user_token = self.user_token_map.get(sample_user, 'N/A')
        item_token = self.item_token_map.get(sample_item, 'N/A')
        print(f"Mapping verification: User {sample_user} → Token {user_token}, Item {sample_item} → Token {item_token}")
    
    def _load_data(self):
        """Load all required data files."""
        # Load user beliefs and embeddings
        with open(self.config['data_paths']['user_beliefs'], 'r') as f:
            user_data = json.load(f)
        
        # Extract current user beliefs (will be updated during training)
        self.users = {}
        self.user_id_mapping = {}  # Map between different user ID formats
        
        for user_id, data in user_data.items():
            beliefs = np.array(data['beliefs'], dtype=np.float32)
            self.users[user_id] = {
                'beliefs': beliefs.copy(),
                'pp1_distance': data['pp1_distance'],
                'cluster_distances': np.array(data['cluster_distances'], dtype=np.float32)
            }
            # Store initial beliefs for episode reset
            self.initial_user_beliefs[user_id] = beliefs.copy()
            
            # Initialize accepted counts based on initial beliefs
            # Note: Will be properly initialized after cluster_sizes are loaded
            # For now, use placeholder values
            self.user_accepted_counts[user_id] = {
                'cluster_counts': np.zeros(5, dtype=np.float32),
                'total_count': 0
            }
            self.initial_user_counts[user_id] = {
                'cluster_counts': np.zeros(5, dtype=np.float32),
                'total_count': 0
            }
            
            # Create mapping from U-format to numeric format
            # U100 -> 100, U1000 -> 1000, etc.
            if user_id.startswith('U'):
                numeric_id = user_id[1:]  # Remove 'U' prefix
                self.user_id_mapping[user_id] = numeric_id
        
        # Load user embeddings
        with open(self.config['data_paths']['user_embeddings'], 'rb') as f:
            self.user_embeddings = pickle.load(f)
        
        # Load item embeddings
        with open(self.config['data_paths']['item_embeddings'], 'rb') as f:
            self.item_embeddings = pickle.load(f)
        
        # Load cluster embeddings
        with open(self.config['data_paths']['cluster_embeddings'], 'rb') as f:
            self.cluster_embeddings = pickle.load(f)
        
        # Load pre-trained recommendations
        with open(self.config['data_paths']['recommendations'], 'rb') as f:
            recommendations_list = pickle.load(f)
        
        # Convert list to dictionary for faster lookup
        # Note: recommendations_list uses numeric tokens as user_id, need to map to actual user_ids
        self.pretrained_recommendations = {}
        for item in recommendations_list:
            numeric_token = str(item['user_id'])  # Ensure it's a string
            recommended_items = item['recommended_items'].tolist() if hasattr(item['recommended_items'], 'tolist') else item['recommended_items']
            # Store using numeric token as key for now, will map during lookup
            # Note: recommended_items are in numeric token format, will be converted during get_recommendations
            self.pretrained_recommendations[numeric_token] = recommended_items
        
        # Load item token mapping
        with open(self.config['data_paths']['item_token_map'], 'r') as f:
            self.item_token_map = json.load(f)
        
        # Load user token mapping
        with open(self.config['data_paths']['user_token_map'], 'r') as f:
            self.user_token_map = json.load(f)
        
        # Create reverse mapping: numeric_token -> user_id
        self.token_to_user_map = {str(token): user_id for user_id, token in self.user_token_map.items()}
    
    def _load_natural_belief_target(self):
        """Load natural belief target for constraining user belief updates."""
        try:
            natural_beliefs_path = self.config['data_paths']['natural_beliefs']
            
            # Support both JSON and PKL formats
            if natural_beliefs_path.endswith('.json'):
                with open(natural_beliefs_path, 'r') as f:
                    target_data = json.load(f)
                self.natural_belief_target = np.array(target_data['natural_belief_target'], dtype=np.float32)
            else:  # PKL format
                with open(natural_beliefs_path, 'rb') as f:
                    target_data = pickle.load(f)
                self.natural_belief_target = np.array(target_data['natural_belief_target'], dtype=np.float32)
            
            print(f"Loaded natural belief target: {self.natural_belief_target}")
            
        except Exception as e:
            print(f"Warning: Could not load natural belief target: {e}")
            # Fallback to uniform distribution
            self.natural_belief_target = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
            print(f"Using fallback uniform target: {self.natural_belief_target}")
        
        # Create enhanced item embeddings (item + cluster)
        self._create_enhanced_item_embeddings()
        
        # Initialize user interaction tracking
        self._reset_user_interactions()
        
        # Load cluster distance data for intelligent acceptance probability
        self._load_cluster_distance_data()
        
        # Pre-load cluster assignment mapping for efficiency
        self._load_cluster_assignments()
        
        # Initialize user accepted counts based on initial beliefs and total_items_in_dataset
        self._initialize_user_counts_from_beliefs()
    
    def _create_enhanced_item_embeddings(self):
        """Create enhanced item embeddings by combining item + cluster embeddings."""
        self.enhanced_item_embeddings = {}
        
        # Load cluster assignments (using balanced clusters if configured)
        cluster_file = self.config['data_paths'].get('cluster_assignments', 'embeddings/cluster_matrix_manifest_K5_topk5.json')
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)
        
        # Create item to cluster mapping
        item_to_cluster = {}
        for cluster_id, items in cluster_data['cluster_to_items'].items():
            for item in items:
                item_to_cluster[item] = cluster_id
        
        # Combine embeddings with normalization
        for item_id, token in self.item_token_map.items():
            # Convert token to string for lookup
            token_str = str(token)
            if token_str in self.item_embeddings:
                item_emb = self.item_embeddings[token_str]
                
                # Get cluster embedding
                cluster_id = item_to_cluster.get(item_id, '0')  # Default to cluster 0
                cluster_emb = self.cluster_embeddings[cluster_id]
                
                # Normalize both embeddings before combining
                item_emb_norm = item_emb / (np.linalg.norm(item_emb) + 1e-8)
                cluster_emb_norm = cluster_emb / (np.linalg.norm(cluster_emb) + 1e-8)
                
                # Add normalized embeddings (element-wise addition)
                combined_emb = item_emb_norm + cluster_emb_norm
                
                # Normalize the combined embedding
                enhanced_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)
                
                self.enhanced_item_embeddings[item_id] = enhanced_emb
        
        print(f"Created {len(self.enhanced_item_embeddings)} enhanced item embeddings")
    
    def _initialize_gpu_tensors(self):
        """Initialize GPU tensors for batch processing."""
        print("Initializing GPU tensors...")
        
        # Convert enhanced item embeddings to GPU tensor
        if self.enhanced_item_embeddings:
            item_ids = list(self.enhanced_item_embeddings.keys())
            # Enhanced embeddings are now already normalized 64D vectors (item + cluster, normalized)
            item_embeddings_list = [self.enhanced_item_embeddings[item_id] for item_id in item_ids]
            
            # Create tensor (already normalized during creation, no need to normalize again)
            self.enhanced_embeddings_tensor_normalized = torch.FloatTensor(item_embeddings_list).to(self.device)
            self.item_ids_list = item_ids  # Keep mapping for results
            
            print(f"Item embeddings tensor (pre-normalized): {self.enhanced_embeddings_tensor_normalized.shape} on {self.device}")
            print(f"Enhanced embeddings: normalized(item_norm + cluster_norm) ready for cosine similarity")
        
        # Convert user embeddings to GPU tensor
        if self.user_embeddings:
            # Create ordered list of user embeddings
            user_ids = list(self.users.keys())
            user_embeddings_list = []
            
            for user_id in user_ids:
                token = self.user_token_map.get(user_id, -1)
                token_str = str(token) if token != -1 else None
                user_emb = self.user_embeddings.get(token_str, np.zeros(64, dtype=np.float32)) if token_str else np.zeros(64, dtype=np.float32)
                user_embeddings_list.append(user_emb)
            
            self.user_embeddings_tensor = torch.FloatTensor(user_embeddings_list).to(self.device)
            self.user_ids_list = user_ids  # Keep mapping for results
            
            print(f"User embeddings tensor: {self.user_embeddings_tensor.shape} on {self.device}")
        
        # Initialize reward calculation tensors
        self._initialize_reward_tensors()
        
        print("GPU tensors initialized successfully!")
    
    def _initialize_reward_tensors(self):
        """Initialize GPU tensors for reward calculation."""
        print("Initializing reward calculation tensors...")
        
        # Load optimal beliefs tensor
        self.optimal_beliefs_tensor = torch.FloatTensor(self.natural_belief_target).to(self.device)
        
        # Initialize user beliefs tensor (will be updated during training)
        user_ids = list(self.users.keys())
        user_beliefs_list = [self.users[user_id]['beliefs'] for user_id in user_ids]
        self.user_beliefs_tensor = torch.FloatTensor(user_beliefs_list).to(self.device)
        
        # Initialize previous beliefs tensor (copy of current beliefs initially)
        self.previous_beliefs_tensor = self.user_beliefs_tensor.clone()
        
        # Store user ID to tensor index mapping for efficient updates
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        
        # Reward calculation constants (from config)
        self.termination_threshold = self.config['reward'].get('termination_threshold', 0.25)
        self.termination_reward = self.config['reward'].get('termination_reward', 5.0)
        
        # Entropy reward configuration
        self.entropy_reward_weight = self.config['reward'].get('entropy_reward_weight', 1.0)
        self.num_clusters = 5
        
        print(f"Optimal beliefs tensor: {self.optimal_beliefs_tensor.shape} on {self.device}")
        print(f"User beliefs tensor: {self.user_beliefs_tensor.shape} on {self.device}")
        print(f"Termination threshold: {self.termination_threshold}")
        print(f"Entropy reward weight: {self.entropy_reward_weight}")
    
    def calculate_entropy_reward(self, recommended_items: List[str]) -> float:
        """
        Calculate entropy reward based on cluster distribution of recommended items
        
        Args:
            recommended_items: List of recommended item IDs
            
        Returns:
            entropy_reward: Reward based on normalized entropy (between 0 and entropy_reward_weight)
        """
        if not recommended_items or not hasattr(self, 'item_to_cluster') or self.item_to_cluster is None:
            return 0.0
        
        
        cluster_ids = []
        for item_id in recommended_items:
            item_id_str = str(item_id)
            
            if hasattr(self, 'item_id_normalized') and self.item_id_normalized is not None:
                normalized_id = self.item_id_normalized.get(item_id_str, item_id_str)
            else:
                normalized_id = item_id_str
            cluster_id = self.item_to_cluster.get(normalized_id, None)
            
            if cluster_id is not None and 0 <= cluster_id < self.num_clusters:
                cluster_ids.append(cluster_id)
        
        if not cluster_ids:
            return 0.0
        
        cluster_counts = Counter(cluster_ids)
        total = len(cluster_ids)
        
        probabilities = np.array([cluster_counts.get(i, 0) / total for i in range(self.num_clusters)])
        
        # Calculate entropy: H(X) = -Σ p(x) log(p(x))
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy += -p * np.log(p)
        
        return entropy * self.entropy_reward_weight
    

    def calculate_step_reward_gpu(self, user_id: str, new_beliefs: np.ndarray, recommended_items: List[str] = None) -> float:
        """
        Calculate step-based reward using GPU acceleration.
        
        Args:
            user_id: User identifier
            new_beliefs: Updated user beliefs after interaction
            recommended_items: List of recommended items (for calculating entropy reward)
            
        Returns:
            Total reward (improvement reward + termination bonus + entropy reward)
        """
        if user_id not in self.user_id_to_index:
            return 0.0
        
        user_idx = self.user_id_to_index[user_id]
        
        # Convert new beliefs to tensor
        new_beliefs_tensor = torch.FloatTensor(new_beliefs).to(self.device)
        
        # Get current and previous beliefs for this user
        current_beliefs = self.user_beliefs_tensor[user_idx]
        previous_beliefs = self.previous_beliefs_tensor[user_idx]
        
        # Calculate distances using GPU
        with torch.no_grad():
            # Part (a): Improvement-based reward
            current_distances = torch.abs(self.optimal_beliefs_tensor - new_beliefs_tensor)
            previous_distances = torch.abs(self.optimal_beliefs_tensor - previous_beliefs)
            
            # Check for improvements (current_distance < previous_distance)
            improvements = current_distances < previous_distances
            
            # Calculate max distance for normalization (theoretical max is 1.0 per cluster)
            max_distance = 1.0
            
            # Calculate cluster rewards only for improved clusters
            cluster_rewards = torch.zeros(5, device=self.device)
            
            # For improved clusters: (max - current) / max + (previous - current)
            improved_mask = improvements
            if improved_mask.any():
                # Absolute performance component: (max - current) / max
                abs_performance = (max_distance - current_distances[improved_mask]) / max_distance
                
                # Improvement bonus: (previous - current)
                improvement_bonus = previous_distances[improved_mask] - current_distances[improved_mask]
                
                # Combined reward for improved clusters
                cluster_rewards[improved_mask] = abs_performance + improvement_bonus
            
            # Sum all cluster rewards
            improvement_reward = cluster_rewards.sum().item()
            
            # Part (b): Termination bonus
            avg_distance = current_distances.mean().item()
            termination_reward = self.termination_reward if avg_distance < self.termination_threshold else 0.0
            
            # Part (c): Entropy reward (diversity bonus)
            entropy_reward = 0.0
            if recommended_items:
                entropy_reward = self.calculate_entropy_reward(recommended_items)
            
            # Total reward
            total_reward = improvement_reward + termination_reward + entropy_reward
        
        # Update tensors with new beliefs
        self.previous_beliefs_tensor[user_idx] = current_beliefs.clone()
        self.user_beliefs_tensor[user_idx] = new_beliefs_tensor
        
        return total_reward
    
    def batch_calculate_step_rewards_gpu(self, user_ids: List[str], new_beliefs_batch: torch.Tensor) -> torch.Tensor:
        """
        Calculate step-based rewards for a batch of users using GPU acceleration.
        
        Args:
            user_ids: List of user identifiers
            new_beliefs_batch: [batch_size, 5] tensor of updated beliefs
            
        Returns:
            [batch_size] tensor of rewards
        """
        batch_size = len(user_ids)
        user_indices = [self.user_id_to_index[user_id] for user_id in user_ids if user_id in self.user_id_to_index]
        
        if not user_indices:
            return torch.zeros(batch_size, device=self.device)
        
        # Convert to tensor indices
        user_indices_tensor = torch.LongTensor(user_indices).to(self.device)
        
        with torch.no_grad():
            # Get current and previous beliefs for batch users
            current_beliefs_batch = self.user_beliefs_tensor[user_indices_tensor]  # [batch_size, 5]
            previous_beliefs_batch = self.previous_beliefs_tensor[user_indices_tensor]  # [batch_size, 5]
            
            # Expand optimal beliefs for batch computation
            optimal_expanded = self.optimal_beliefs_tensor.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 5]
            
            # Calculate distances
            current_distances = torch.abs(optimal_expanded - new_beliefs_batch)  # [batch_size, 5]
            previous_distances = torch.abs(optimal_expanded - previous_beliefs_batch)  # [batch_size, 5]
            
            # Check for improvements
            improvements = current_distances < previous_distances  # [batch_size, 5]
            
            # Calculate cluster rewards
            max_distance = 1.0
            
            # Absolute performance component: (max - current) / max
            abs_performance = (max_distance - current_distances) / max_distance
            
            # Improvement bonus: (previous - current)
            improvement_bonus = previous_distances - current_distances
            
            # Combined reward for improved clusters only
            cluster_rewards = (abs_performance + improvement_bonus) * improvements.float()
            
            # Sum cluster rewards for each user
            improvement_rewards = cluster_rewards.sum(dim=1)  # [batch_size]
            
            # Calculate termination bonuses
            avg_distances = current_distances.mean(dim=1)  # [batch_size]
            termination_rewards = (avg_distances < self.termination_threshold).float() * self.termination_reward
            
            # Total rewards
            total_rewards = improvement_rewards + termination_rewards
            
            # Update tensors
            self.previous_beliefs_tensor[user_indices_tensor] = current_beliefs_batch
            self.user_beliefs_tensor[user_indices_tensor] = new_beliefs_batch
        
        return total_rewards
    
    def update_user_beliefs_step(self, user_id: str, accepted_items: List[str], rejected_items: List[str], recommended_items: List[str] = None) -> Tuple[np.ndarray, float]:
        """
        Update user beliefs after a single step and calculate reward.
        Uses count-based distribution: belief[i] = count[i] / total_count
        
        Args:
            user_id: User identifier
            accepted_items: List of accepted item IDs
            rejected_items: List of rejected item IDs
            recommended_items: List of recommended items (for calculating entropy reward)
            
        Returns:
            Tuple of (new_beliefs, step_reward)
        """
        if not accepted_items:
            # No accepted items, no belief update, no reward
            return self.users[user_id]['beliefs'].copy(), 0.0
        
        # Update counts for each accepted item
        for item_id in accepted_items:
            # Get item's cluster assignment
            if hasattr(self, 'item_to_cluster') and self.item_to_cluster:
                # Normalize item ID for lookup
                item_cluster = None
                item_id_str = str(item_id)
                
                # Try different formats to find the item
                if hasattr(self, 'item_id_normalized') and self.item_id_normalized:
                    normalized_id = self.item_id_normalized.get(item_id_str, item_id_str)
                    item_cluster = self.item_to_cluster.get(normalized_id, None)
                else:
                    # Fallback to manual format checking
                    item_cluster = self.item_to_cluster.get(item_id_str, None)
                    if item_cluster is None and not item_id_str.startswith('N'):
                        item_with_n = f"N{item_id_str}"
                        item_cluster = self.item_to_cluster.get(item_with_n, None)
                    if item_cluster is None and item_id_str.startswith('N'):
                        item_without_n = item_id_str[1:]
                        item_cluster = self.item_to_cluster.get(item_without_n, None)
                
                if item_cluster is not None and 0 <= item_cluster < 5:
                    # Increment count for this cluster
                    self.user_accepted_counts[user_id]['cluster_counts'][item_cluster] += 1
        
        # Calculate new beliefs: belief[i] = cluster_accepted_count[i] / total_items_in_dataset
        # This represents the absolute coverage of items in the dataset
        # Sum of beliefs ≤ 1.0 (equals 1.0 only if user accepted all items in dataset)
        counts = self.user_accepted_counts[user_id]['cluster_counts']
        
        if hasattr(self, 'total_items_in_dataset'):
            # Correct formula: belief[i] = accepted_count[i] / total_items_in_dataset
            new_beliefs = counts / self.total_items_in_dataset
        else:
            # Fallback to old method if total_items_in_dataset not loaded
            total = self.user_accepted_counts[user_id]['total_count']
            if total > 0:
                new_beliefs = counts / total
            else:
                new_beliefs = np.zeros(5, dtype=np.float32)
        
        # Update user beliefs in CPU storage
        self.users[user_id]['beliefs'] = new_beliefs
        
        # Calculate step reward using GPU (includes entropy reward)
        step_reward = self.calculate_step_reward_gpu(user_id, new_beliefs, recommended_items)
        
        return new_beliefs, step_reward
    
    def batch_update_user_beliefs_gpu(self, user_ids: List[str], accepted_items_batch: List[List[str]], 
                                      rejected_items_batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch update user beliefs using GPU acceleration (vectorized).
        
        Args:
            user_ids: List of user identifiers
            accepted_items_batch: List of accepted item lists for each user
            rejected_items_batch: List of rejected item lists for each user
            
        Returns:
            Tuple of (new_beliefs_tensor [batch_size, 5], rewards_tensor [batch_size])
        """
        batch_size = len(user_ids)
        
        # Get current beliefs for all users
        current_beliefs_list = []
        user_indices = []
        
        for user_id in user_ids:
            if user_id in self.user_id_to_index:
                user_indices.append(self.user_id_to_index[user_id])
                current_beliefs_list.append(self.users[user_id]['beliefs'])
            else:
                # Fallback for missing users
                user_indices.append(-1)
                current_beliefs_list.append(np.zeros(5, dtype=np.float32))
        
        # Convert to GPU tensor
        current_beliefs_tensor = torch.FloatTensor(current_beliefs_list).to(self.device)  # [batch_size, 5]
        
        # Calculate belief updates for each user (vectorized where possible)
        belief_updates_list = []
        
        for i, (user_id, accepted_items) in enumerate(zip(user_ids, accepted_items_batch)):
            if not accepted_items:
                # No updates for this user
                belief_updates_list.append(np.zeros(5, dtype=np.float32))
                continue
            
            # Vectorized cluster lookup
            belief_update = np.zeros(5, dtype=np.float32)
            current_belief = current_beliefs_list[i]
            
            # Batch process accepted items
            for item_id in accepted_items:
                # Fast normalized lookup
                item_id_str = str(item_id)
                normalized_id = self.item_id_normalized.get(item_id_str, item_id_str)
                item_cluster = self.item_to_cluster.get(normalized_id, None)
                
                if item_cluster is not None and 0 <= item_cluster < 5:
                    # Adaptive learning rate based on distance to target
                    distance_to_target = max(0, self.natural_belief_target[item_cluster] - current_belief[item_cluster])
                    
                    if distance_to_target > 0.3:
                        increment = 0.02
                    elif distance_to_target > 0.1:
                        increment = 0.015
                    else:
                        increment = 0.01
                    
                    belief_update[item_cluster] += increment
            
            belief_updates_list.append(belief_update)
        
        # Convert updates to GPU tensor
        belief_updates_tensor = torch.FloatTensor(belief_updates_list).to(self.device)  # [batch_size, 5]
        
        # Apply updates (vectorized)
        new_beliefs_tensor = current_beliefs_tensor + belief_updates_tensor
        
        # Normalize (vectorized): ensure non-negative and sum <= 1
        new_beliefs_tensor = torch.clamp(new_beliefs_tensor, min=0.0)
        
        # Normalize rows where sum > 1
        row_sums = new_beliefs_tensor.sum(dim=1, keepdim=True)  # [batch_size, 1]
        needs_normalization = row_sums > 1.0
        new_beliefs_tensor = torch.where(
            needs_normalization,
            new_beliefs_tensor / row_sums,
            new_beliefs_tensor
        )
        
        # Apply belief constraints (vectorized)
        new_beliefs_tensor = self._apply_belief_constraints_batch_gpu(new_beliefs_tensor)
        
        # Calculate rewards using existing batch method
        rewards_tensor = self.batch_calculate_step_rewards_gpu(user_ids, new_beliefs_tensor)
        
        # Update CPU storage
        new_beliefs_np = new_beliefs_tensor.cpu().numpy()
        for i, user_id in enumerate(user_ids):
            if user_id in self.users:
                self.users[user_id]['beliefs'] = new_beliefs_np[i]
        
        return new_beliefs_tensor, rewards_tensor
    
    def _apply_belief_constraints_batch_gpu(self, beliefs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply belief constraints to batch of beliefs using GPU (vectorized).
        
        Args:
            beliefs_tensor: [batch_size, 5] tensor of beliefs
            
        Returns:
            [batch_size, 5] tensor of constrained beliefs
        """
        # Convert natural belief target to tensor
        target_tensor = self.optimal_beliefs_tensor.unsqueeze(0)  # [1, 5]
        
        # Calculate how much each belief exceeds target
        excess = torch.clamp(beliefs_tensor - target_tensor, min=0.0)  # [batch_size, 5]
        
        # Reduce excess by 50% (soft constraint)
        constrained_beliefs = beliefs_tensor - 0.5 * excess
        
        # Ensure non-negative
        constrained_beliefs = torch.clamp(constrained_beliefs, min=0.0)
        
        # Re-normalize if needed
        row_sums = constrained_beliefs.sum(dim=1, keepdim=True)
        needs_normalization = row_sums > 1.0
        constrained_beliefs = torch.where(
            needs_normalization,
            constrained_beliefs / row_sums,
            constrained_beliefs
        )
        
        return constrained_beliefs
    
    def _load_cluster_distance_data(self):
        """Load cluster distance data for intelligent acceptance probability."""
        try:
            import torch
            
            # Load cluster distances
            self.cluster_distances = torch.load('embeddings/cluster_distances_K5_topk5.pt')
            print(f"Loaded cluster distances: {self.cluster_distances.shape}")
            
            # Load cluster centers (optional, for reference)
            try:
                with open('embeddings/cluster_centers_K5_topk5.json', 'r') as f:
                    self.cluster_centers = json.load(f)
                print(f"Loaded cluster centers for {len(self.cluster_centers)} clusters")
            except FileNotFoundError:
                print("Cluster centers file not found, using distances only")
                self.cluster_centers = None
            
            # Create item index mapping from item_token_map
            self.item_to_index = {}
            for item_id, token in self.item_token_map.items():
                # Assuming the cluster_distances tensor is indexed by token
                self.item_to_index[item_id] = int(token)
            
            print(f"Created item-to-index mapping for {len(self.item_to_index)} items")
            
        except Exception as e:
            print(f"Warning: Could not load cluster distance data: {e}")
            print("Falling back to fixed acceptance rate")
            self.cluster_distances = None
            self.cluster_centers = None
            self.item_to_index = None
    
    def _load_cluster_assignments(self):
        """Pre-load cluster assignments for efficient lookup with GPU tensors."""
        try:
            # Load cluster assignment mapping once at initialization (using balanced clusters if configured)
            cluster_file = self.config['data_paths'].get('cluster_assignments', 'embeddings/cluster_matrix_manifest_K5_topk5.json')
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            # Load cluster sizes for belief calculation
            # belief[i] = cluster_accepted_count[i] / total_items_in_dataset
            self.cluster_sizes = np.array([
                int(cluster_data['cluster_sizes']['0']),
                int(cluster_data['cluster_sizes']['1']),
                int(cluster_data['cluster_sizes']['2']),
                int(cluster_data['cluster_sizes']['3']),
                int(cluster_data['cluster_sizes']['4'])
            ], dtype=np.float32)
            self.total_items_in_dataset = np.sum(self.cluster_sizes)
            print(f"Loaded cluster sizes: {self.cluster_sizes}")
            print(f"Total items in dataset: {self.total_items_in_dataset}")
            
            # Create efficient item-to-cluster mapping (dict for backward compatibility)
            self.item_to_cluster = {}
            for cluster_id, items in cluster_data['cluster_to_items'].items():
                cluster_int = int(cluster_id)
                for item in items:
                    self.item_to_cluster[item] = cluster_int
            
            # Create normalized item ID mapping for tensor operations
            # This handles all possible ID formats (N123, 123, etc.)
            self.item_id_normalized = {}
            for item_id in self.item_to_cluster.keys():
                # Store multiple formats for fast lookup
                self.item_id_normalized[item_id] = item_id
                if item_id.startswith('N'):
                    without_n = item_id[1:]
                    self.item_id_normalized[without_n] = item_id
                else:
                    with_n = f"N{item_id}"
                    self.item_id_normalized[with_n] = item_id
            
            print(f"Loaded cluster assignments for {len(self.item_to_cluster)} items across {len(cluster_data['cluster_to_items'])} clusters")
            
            # Verify cluster distribution
            cluster_counts = {}
            for item, cluster in self.item_to_cluster.items():
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            
            print("Cluster distribution:")
            for cluster_id in sorted(cluster_counts.keys()):
                print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} items")
            
            # Sample verification: show some item-to-cluster mappings
            sample_items = list(self.item_to_cluster.items())[:3]
            print(f"Mapping examples: {sample_items}")
            
        except Exception as e:
            print(f"Warning: Could not load cluster assignments: {e}")
            self.item_to_cluster = None
            self.item_id_normalized = None
    
    def _initialize_user_counts_from_beliefs(self):
        """Initialize user accepted counts based on initial beliefs and total_items_in_dataset."""
        if not hasattr(self, 'total_items_in_dataset'):
            print("Warning: total_items_in_dataset not loaded, cannot initialize counts")
            return
        
        print("Initializing user accepted counts from initial beliefs...")
        
        for user_id in self.users.keys():
            initial_beliefs = self.initial_user_beliefs[user_id]
            
            # Reverse calculate counts from beliefs
            # belief[i] = count[i] / total_items_in_dataset
            # => count[i] = belief[i] * total_items_in_dataset
            initial_counts = initial_beliefs * self.total_items_in_dataset
            
            self.user_accepted_counts[user_id] = {
                'cluster_counts': initial_counts.copy(),
            }
            self.initial_user_counts[user_id] = {
                'cluster_counts': initial_counts.copy(),
            }
        
        print(f"Initialized counts for {len(self.users)} users")
    
    def _reset_user_interactions(self):
        """Reset user interaction tracking for new episode."""
        self.user_interactions = {
            user_id: {
                'recommended_items': [],
                'accepted_items': [],
                'rejected_items': [],
                'round_recommended': [],  # Track current round only
                'round_accepted': [],     # Track current round only
                'round_rejected': []      # Track current round only
            }
            for user_id in self.users.keys()
        }
    
    def reset_round_interactions(self):
        """Reset round-level interaction tracking."""
        for user_id in self.user_interactions:
            self.user_interactions[user_id]['round_recommended'] = []
            self.user_interactions[user_id]['round_accepted'] = []
            self.user_interactions[user_id]['round_rejected'] = []
    
    def get_user_state(self, user_id: str) -> np.ndarray:
        """
        Get current state for a user.
        State = user beliefs (5D) + user embedding (64D) = 69D
        """
        user_beliefs = self.users[user_id]['beliefs']
        
        # Get user embedding using token
        token = self.user_token_map.get(user_id, -1)
        token_str = str(token) if token != -1 else None
        user_embedding = self.user_embeddings.get(token_str, np.zeros(64, dtype=np.float32)) if token_str else np.zeros(64, dtype=np.float32)
        
        # Concatenate beliefs and embedding
        state = np.concatenate([user_beliefs, user_embedding])
        return state.astype(np.float32)
    
    def get_batch_user_states(self, user_ids: List[str]) -> torch.Tensor:
        """
        Get batch of user states for GPU processing.
        Returns: [batch_size, 69] tensor on GPU
        """
        batch_states = []
        
        for user_id in user_ids:
            user_beliefs = self.users[user_id]['beliefs']
            
            # Find user index in pre-loaded embeddings
            try:
                user_idx = self.user_ids_list.index(user_id)
                user_embedding = self.user_embeddings_tensor[user_idx].cpu().numpy()
            except (ValueError, AttributeError):
                # Fallback to original method using token
                token = self.user_token_map.get(user_id, -1)
                token_str = str(token) if token != -1 else None
                user_embedding = self.user_embeddings.get(token_str, np.zeros(64, dtype=np.float32)) if token_str else np.zeros(64, dtype=np.float32)
            
            # Concatenate beliefs and embedding
            state = np.concatenate([user_beliefs, user_embedding])
            batch_states.append(state)
        
        # Convert to GPU tensor
        batch_tensor = torch.FloatTensor(batch_states).to(self.device)
        return batch_tensor
    
    def get_recommendations(self, user_id: str, virtual_item: np.ndarray) -> List[str]:
        """
        Generate recommendations for a user based on virtual item representation.
        
        Args:
            user_id: User identifier
            virtual_item: 64D virtual item representation from TD3
            
        Returns:
            List of recommended item IDs
        """
        # Get RL recommendations using cosine similarity
        rl_recommendations = self._get_rl_recommendations(virtual_item)
        
        # Get pre-trained recommendations using token mapping
        # Use original user_id for token mapping (user_token_map expects string format like "U100")
        numeric_token = str(self.user_token_map.get(user_id, -1))  # Get numeric token, default to -1 if not found
        pretrained_token_recs = self.pretrained_recommendations.get(numeric_token, [])
        
        # Convert pretrained recommendations from tokens to item IDs
        pretrained_recs = []
        if pretrained_token_recs:
            # Create reverse token mapping: token -> item_id
            token_to_item = {str(token): item_id for item_id, token in self.item_token_map.items()}
            
            # Sample verification for token conversion
            if hasattr(self, '_token_conversion_shown'):
                pass  # Already shown
            else:
                self._token_conversion_shown = True
                print(f"Token conversion example (user {user_id}):")
                for i, token in enumerate(pretrained_token_recs[:3]):
                    token_str = str(token)
                    if token_str in token_to_item:
                        item_id = token_to_item[token_str]
                        print(f"  Token {token} → Item {item_id}")
                    else:
                        print(f"  Token {token} → No corresponding Item")
            
            for token in pretrained_token_recs:
                token_str = str(token)
                if token_str in token_to_item:
                    pretrained_recs.append(token_to_item[token_str])
                # Skip tokens that don't have corresponding item IDs
        
        # Combine recommendations (RL + pre-trained)
        combined_recs = rl_recommendations + pretrained_recs
        
        # Remove duplicates while preserving order
        seen = set()
        final_recs = []
        for item in combined_recs:
            if item not in seen and len(final_recs) < self.config['recommendation']['total_recommendations']:
                seen.add(item)
                final_recs.append(item)
        

        
        return final_recs
    
    def _get_rl_recommendations(self, virtual_item: np.ndarray) -> List[str]:
        """Get top-k items most similar to virtual item representation."""
        if len(self.enhanced_item_embeddings) == 0:
            print("Warning: No enhanced item embeddings available, returning empty list")
            return []
        
        try:
            # Convert to numpy arrays for vectorized computation
            item_ids = list(self.enhanced_item_embeddings.keys())
            # Enhanced embeddings are already normalized (item_norm + cluster_norm, then normalized)
            item_embeddings = np.array([
                self.enhanced_item_embeddings[item_id]  # Use full enhanced embedding (64D)
                for item_id in item_ids
            ])
            
            # Normalize virtual item for cosine similarity
            virtual_item_norm = virtual_item / (np.linalg.norm(virtual_item) + 1e-8)
            
            # Enhanced embeddings are already normalized, so direct dot product gives cosine similarity
            similarities = np.dot(item_embeddings, virtual_item_norm)
            
            # Get top-k indices
            top_k = min(self.config['recommendation']['rl_list_size'], len(item_ids))
            if top_k <= 0:
                return []
                
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            # Return top item IDs
            top_items = [item_ids[i] for i in top_indices]
            return top_items
            
        except Exception as e:
            print(f"Error in _get_rl_recommendations: {e}")
            return []
    
    def _get_batch_rl_recommendations(self, virtual_items: torch.Tensor) -> List[List[str]]:
        """
        GPU-accelerated batch recommendation generation with pre-normalized embeddings.
        Args:
            virtual_items: [batch_size, 64] tensor on GPU
        Returns:
            List of recommendation lists for each user
        """
        if not hasattr(self, 'enhanced_embeddings_tensor_normalized') or self.enhanced_embeddings_tensor_normalized is None:
            print("Warning: Normalized GPU tensors not initialized, falling back to CPU")
            return [self._get_rl_recommendations(vi.cpu().numpy()) for vi in virtual_items]
        
        try:
            with torch.no_grad():  # No gradients needed for inference
                # Normalize virtual items: [batch_size, 64]
                virtual_items_norm = torch.nn.functional.normalize(virtual_items, p=2, dim=1)
                
                # Use pre-normalized item embeddings (no repeated normalization!)
                # This saves significant computation time in each batch
                
                # Direct matrix multiplication for cosine similarity: [batch_size, num_items]
                # Since both tensors are L2 normalized, dot product = cosine similarity
                similarities = torch.mm(virtual_items_norm, self.enhanced_embeddings_tensor_normalized.t())
                
                # Get top-k for each user
                top_k = min(self.config['recommendation']['rl_list_size'], len(self.item_ids_list))
                if top_k <= 0:
                    return [[] for _ in range(virtual_items.shape[0])]
                
                # Get top-k indices for all users at once: [batch_size, top_k]
                _, top_indices = torch.topk(similarities, top_k, dim=1)
                
                # Convert to item IDs
                batch_recommendations = []
                for user_indices in top_indices:
                    user_recs = [self.item_ids_list[idx.item()] for idx in user_indices]
                    batch_recommendations.append(user_recs)
                
                # Apply forced Cluster 3 exploration
                batch_recommendations = self._apply_forced_cluster3_exploration(batch_recommendations)
                
                return batch_recommendations
                
        except Exception as e:
            print(f"Error in batch recommendations: {e}")
            # Fallback to CPU processing
            return [self._get_rl_recommendations(vi.cpu().numpy()) for vi in virtual_items]
    
    def _apply_forced_cluster3_exploration(self, batch_recommendations: List[List[str]]) -> List[List[str]]:
        """
        Force a percentage of recommendations to be from Cluster 3 to break cold start cycle.
        
        This is a critical intervention to address the Cluster 3 cold start problem:
        - Users have near-zero Cluster 3 beliefs (0.000046)
        - Actor learned to avoid Cluster 3 (negative cosine similarity)
        - Without forced exploration, Cluster 3 items are never recommended
        - Beliefs can't improve without recommendations
        
        Args:
            batch_recommendations: List of recommendation lists for each user
            
        Returns:
            Modified recommendation lists with forced Cluster 3 items
        """
        if not hasattr(self, 'item_to_cluster') or self.item_to_cluster is None:
            return batch_recommendations
        
        # Get forced exploration ratio from config (default 20%)
        forced_ratio = self.config['environment'].get('forced_cluster3_ratio', 0.20)
        
        if forced_ratio <= 0:
            return batch_recommendations  # No forced exploration
        
        # Get all Cluster 3 items
        cluster3_items = []
        for item_id, cluster_id in self.item_to_cluster.items():
            if cluster_id == 3:
                cluster3_items.append(item_id)
        
        if not cluster3_items:
            print("Warning: No Cluster 3 items found for forced exploration")
            return batch_recommendations
        
        # Apply forced exploration to each user's recommendations
        modified_recommendations = []
        for user_recs in batch_recommendations:
            if not user_recs:
                modified_recommendations.append(user_recs)
                continue
            
            # Calculate how many items to force
            num_forced = max(1, int(len(user_recs) * forced_ratio))
            num_keep = len(user_recs) - num_forced
            
            # Keep top items from actor (excluding Cluster 3)
            kept_items = []
            for item in user_recs:
                if len(kept_items) >= num_keep:
                    break
                # Check if item is NOT Cluster 3
                item_str = str(item)
                normalized_id = self.item_id_normalized.get(item_str, item_str) if hasattr(self, 'item_id_normalized') and self.item_id_normalized else item_str
                cluster_id = self.item_to_cluster.get(normalized_id, None)
                if cluster_id != 3:
                    kept_items.append(item)
            
            # If we couldn't get enough non-Cluster-3 items, just take what we have
            if len(kept_items) < num_keep:
                kept_items = user_recs[:num_keep]
            
            # Add random Cluster 3 items
            forced_items = random.sample(cluster3_items, min(num_forced, len(cluster3_items)))
            
            # Combine: kept items + forced Cluster 3 items
            final_recs = kept_items + forced_items
            
            # Shuffle to avoid always having Cluster 3 at the end
            random.shuffle(final_recs)
            
            modified_recommendations.append(final_recs)
        
        return modified_recommendations
    
    def simulate_user_interaction(self, user_id: str, recommended_items: List[str]) -> Tuple[List[str], List[str]]:
        """
        Simulate user interaction with recommended items using intelligent acceptance probability.
        
        Uses sigmoid transformation: acceptance_prob = (1 / (1 + distance)) * user_belief_for_cluster
        
        Args:
            user_id: User identifier
            recommended_items: List of recommended item IDs
            
        Returns:
            Tuple of (accepted_items, rejected_items)
        """
        accepted_items = []
        rejected_items = []
        
        for item_id in recommended_items:
            # Calculate intelligent acceptance probability for this user-item pair
            acceptance_prob = self._calculate_acceptance_probability(user_id, item_id)
            
            # Check if using threshold system
            use_threshold = self.config['environment'].get('use_threshold_system', False)
            
            if use_threshold:
                # Threshold system: acceptance_prob is either 1.0 or 0.0
                if acceptance_prob > 0.5:  # If above threshold (returned as 1.0)
                    accepted_items.append(item_id)
                else:  # If below threshold (returned as 0.0)
                    rejected_items.append(item_id)
            else:
                # Probability system: use random comparison
                if random.random() < acceptance_prob:
                    accepted_items.append(item_id)
                else:
                    rejected_items.append(item_id)
        
        # Track interactions (both episode-level and round-level)
        self.user_interactions[user_id]['recommended_items'].extend(recommended_items)
        self.user_interactions[user_id]['accepted_items'].extend(accepted_items)
        self.user_interactions[user_id]['rejected_items'].extend(rejected_items)
        
        # Track round-level interactions
        self.user_interactions[user_id]['round_recommended'].extend(recommended_items)
        self.user_interactions[user_id]['round_accepted'].extend(accepted_items)
        self.user_interactions[user_id]['round_rejected'].extend(rejected_items)
        
        return accepted_items, rejected_items
    
    def _calculate_acceptance_probability(self, user_id: str, item_id: str) -> float:
        """
        Calculate intelligent acceptance probability using Sigmoid activation.
        
        Formula: acceptance_prob = (1 - distance/max_distance) × sigmoid(user_belief × 10000) + 0.1
        
        This approach:
        - Keeps original user beliefs unchanged (no artificial inflation)
        - Uses Sigmoid to transform tiny beliefs (0.0006) to meaningful values (0.998)
        - Creates significant differentiation in acceptance rates (10% to 90%)
        - Maintains logical consistency (user belief never exceeds optimal)
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Acceptance probability between 0.1 and 1.0
        """
        # Check if intelligent acceptance is enabled
        if not self.config['environment'].get('use_intelligent_acceptance', False):
            return self.config['environment']['user_acceptance_rate']
        
        # Fallback to fixed rate if cluster data not available
        if self.cluster_distances is None or self.item_to_index is None:
            return self.config['environment']['user_acceptance_rate']
        
        try:
            # Get user beliefs (these are updated after each episode)
            user_beliefs = self.users[user_id]['beliefs']
            
            # Get item's cluster assignment using pre-loaded mapping
            if self.item_to_cluster is None:
                # Fallback: use average belief if no cluster mapping
                user_belief = np.mean(user_beliefs)
                distance = 1.0
                item_cluster = None
            else:
                # Efficient cluster lookup
                # Ensure item_id is string format for cluster lookup
                item_id_str = str(item_id)
                
                # Try multiple formats to find the item in cluster assignments
                item_cluster = None
                
                # Format 1: Try as-is (in case it already has 'N' prefix)
                item_cluster = self.item_to_cluster.get(item_id_str, None)
                
                # Format 2: Try with 'N' prefix (most common case)
                if item_cluster is None and not item_id_str.startswith('N'):
                    item_with_n = f"N{item_id_str}"
                    item_cluster = self.item_to_cluster.get(item_with_n, None)
                
                # Format 3: Try without 'N' prefix (if original had it)
                if item_cluster is None and item_id_str.startswith('N'):
                    item_without_n = item_id_str[1:]
                    item_cluster = self.item_to_cluster.get(item_without_n, None)
                
                if item_cluster is None:
                    # Still not found, use average belief
                    user_belief = np.mean(user_beliefs)
                    distance = 1.0
                    # Suppress warning for cleaner output - this is expected for some items
                    # print(f"Warning: Item {item_id} not found in cluster assignments")
                else:
                    # CRITICAL: Get user's belief for the CORRECT cluster
                    if 0 <= item_cluster < len(user_beliefs):
                        user_belief = user_beliefs[item_cluster]
                    else:
                        print(f"Warning: Invalid cluster {item_cluster} for item {item_id}")
                        user_belief = np.mean(user_beliefs)
                        distance = 1.0
                        item_cluster = None
                    
                    # Get item's distance to its cluster center
                    # Try multiple formats for token lookup
                    item_index = None
                    if item_cluster is not None:
                        # Try different formats to find the item in token mapping
                        if item_id in self.item_to_index:
                            item_index = self.item_to_index[item_id]
                        elif not item_id_str.startswith('N'):
                            # Try with 'N' prefix
                            item_with_n = f"N{item_id_str}"
                            if item_with_n in self.item_to_index:
                                item_index = self.item_to_index[item_with_n]
                        elif item_id_str.startswith('N'):
                            # Try without 'N' prefix
                            item_without_n = item_id_str[1:]
                            if item_without_n in self.item_to_index:
                                item_index = self.item_to_index[item_without_n]
                    
                    if item_index is not None:
                        
                        # Check if cluster_distances is 1D or 2D
                        if len(self.cluster_distances.shape) == 1:
                            # 1D tensor: use item index directly
                            if item_index < self.cluster_distances.shape[0]:
                                distance = float(self.cluster_distances[item_index])
                                # Handle negative distances (convert to positive)
                                if distance < 0:
                                    distance = abs(distance)
                            else:
                                distance = 1.0
                        else:
                            # 2D tensor: use item index and cluster index
                            if (item_index < self.cluster_distances.shape[0] and 
                                item_cluster < self.cluster_distances.shape[1]):
                                distance = float(self.cluster_distances[item_index, item_cluster])
                                # Handle negative distances
                                if distance < 0:
                                    distance = abs(distance)
                            else:
                                distance = 1.0
                    else:
                        distance = 1.0  # Default distance
            
            # Sigmoid activation parameters
            sigmoid_scale = self.config['environment'].get('sigmoid_scale', 10000)
            baseline_rate = self.config['environment'].get('baseline_acceptance_rate', 0.1)
            max_distance = self.config['environment'].get('max_cluster_distance', 5)
            
            # Calculate distance factor: (1 - distance/max_distance)
            distance_factor = max(0.0, 1.0 - (distance / max_distance))
            
            # Apply Sigmoid activation to user belief
            # sigmoid(x) = 1 / (1 + exp(-x))
            belief_activated = 1.0 / (1.0 + np.exp(-user_belief * sigmoid_scale))
            
            # Calculate final acceptance probability
            # Formula: (1 - distance/max_distance) × sigmoid(belief × scale) + baseline
            acceptance_prob = distance_factor * belief_activated + baseline_rate
            
            # Ensure probability is in valid range [0, 1]
            acceptance_prob = max(0.0, min(1.0, acceptance_prob))
            
            # Sample verification: print first few calculations for validation (disabled)
            # if hasattr(self, '_verification_count'):
            #     self._verification_count += 1
            # else:
            #     self._verification_count = 1
            # 
            # if self._verification_count <= 5:  # Print first 5 calculations
            #     print(f"Sigmoid verification #{self._verification_count}: {item_id} → Cluster {item_cluster} → Distance {distance:.1f}")
            #     print(f"  User belief: {user_belief:.6f} → Sigmoid: {belief_activated:.6f}")
            #     print(f"  Distance factor: {distance_factor:.3f} → Final prob: {acceptance_prob:.3f} ({acceptance_prob*100:.1f}%)")

            return acceptance_prob
            
        except Exception as e:
            print(f"Error calculating acceptance probability for user {user_id}, item {item_id}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to fixed rate
            return self.config['environment']['user_acceptance_rate']
    
    def step(self, user_id: str, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            user_id: User identifier
            action: Virtual item representation from TD3 (64D)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current state
        current_state = self.get_user_state(user_id)
        
        # Generate recommendations (this might be slow)
        recommended_items = self.get_recommendations(user_id, action)
        
        # Simulate user interaction
        accepted_items, rejected_items = self.simulate_user_interaction(user_id, recommended_items)
        
        # For now, next state is same as current state (beliefs update after full round)
        next_state = current_state
        
        # Reward will be calculated after full round
        reward = 0.0
        
        # Episode is never "done" for individual users
        done = False
        
        # Info dictionary
        info = {
            'user_id': user_id,
            'recommended_items': recommended_items,
            'accepted_items': accepted_items,
            'rejected_items': rejected_items,
            'acceptance_rate': len(accepted_items) / len(recommended_items) if recommended_items else 0
        }
        
        return next_state, reward, done, info
    
    def batch_step(self, user_ids: List[str], virtual_items: torch.Tensor) -> List[Dict]:
        """
        Execute batch step for multiple users using GPU acceleration.
        
        Args:
            user_ids: List of user IDs
            virtual_items: [batch_size, 64] tensor of virtual items on GPU
            
        Returns:
            List of info dictionaries for each user
        """
        batch_results = []
        
        # Get batch RL recommendations using GPU
        batch_rl_recs = self._get_batch_rl_recommendations(virtual_items)
        
        # Process each user (this part still needs individual processing for pre-trained recs)
        for i, user_id in enumerate(user_ids):
            rl_recommendations = batch_rl_recs[i]
            
            # Get pre-trained recommendations (individual lookup needed)
            # Use original user_id for token mapping (user_token_map expects string format like "U100")
            numeric_token = str(self.user_token_map.get(user_id, -1))  # Get numeric token, default to -1 if not found
            pretrained_recs = self.pretrained_recommendations.get(numeric_token, [])
            
            # Combine recommendations
            combined_recs = rl_recommendations + pretrained_recs
            
            # Remove duplicates while preserving order
            seen = set()
            final_recs = []
            for item in combined_recs:
                if item not in seen and len(final_recs) < self.config['recommendation']['total_recommendations']:
                    seen.add(item)
                    final_recs.append(item)
            
            # Simulate user interaction
            accepted_items, rejected_items = self.simulate_user_interaction(user_id, final_recs)
            
            # Step-based belief update and reward calculation (pass recommendation list for entropy reward)
            new_beliefs, step_reward = self.update_user_beliefs_step(user_id, accepted_items, rejected_items, final_recs)
            
            # Create info dictionary with step reward
            info = {
                'user_id': user_id,
                'recommended_items': final_recs,
                'accepted_items': accepted_items,
                'rejected_items': rejected_items,
                'acceptance_rate': len(accepted_items) / len(final_recs) if final_recs else 0,
                'step_reward': step_reward,
                'new_beliefs': new_beliefs.tolist(),
                'belief_updated': len(accepted_items) > 0
            }
            
            batch_results.append(info)
        
        return batch_results
    
    def update_user_beliefs_after_episode(self, active_users=None):
        """Update user beliefs based on interactions during the entire episode (50 rounds)."""
        import time
        start_time = time.time()
        
        print(f"Updating user beliefs after episode {self.current_episode}")
        
        # Only update beliefs for active users
        users_to_update = active_users if active_users else self.users.keys()
        
        # Pre-load cluster data once (optimization) - using balanced clusters if configured
        cluster_load_start = time.time()
        cluster_file = self.config['data_paths'].get('cluster_assignments', 'embeddings/cluster_matrix_manifest_K5_topk5.json')
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)
        
        # Create item to cluster mapping once
        item_to_cluster = {}
        for cluster_id, items in cluster_data['cluster_to_items'].items():
            for item in items:
                item_to_cluster[item] = int(cluster_id)
        
        cluster_load_time = time.time() - cluster_load_start
        print(f"  Cluster data loaded in {cluster_load_time:.3f}s")
        
        # Process users
        belief_calc_start = time.time()
        users_updated = 0
        
        for user_id in users_to_update:
            if user_id in self.user_interactions:
                accepted_items = self.user_interactions[user_id]['accepted_items']
                
                if accepted_items:
                    # Calculate new beliefs based on accepted items from all rounds
                    new_beliefs = self._calculate_new_beliefs_optimized(user_id, accepted_items, item_to_cluster)
                    self.users[user_id]['beliefs'] = new_beliefs
                    
                    # Store new beliefs, calculate distances in batch later
                    self.users[user_id]['beliefs'] = new_beliefs
                    users_updated += 1
        
        belief_calc_time = time.time() - belief_calc_start
        
        # Batch calculate distances for all updated users
        distance_calc_start = time.time()
        updated_user_ids = []
        updated_beliefs = []
        
        for user_id in users_to_update:
            if user_id in self.user_interactions and self.user_interactions[user_id]['accepted_items']:
                updated_user_ids.append(user_id)
                updated_beliefs.append(self.users[user_id]['beliefs'].tolist())
        
        # Batch distance calculation
        if updated_beliefs:
            for i, user_id in enumerate(updated_user_ids):
                beliefs = np.array(updated_beliefs[i])
                # Calculate simple distance to natural target
                pp1_distance = np.abs(beliefs - self.natural_belief_target).sum()
                cluster_distances = np.abs(beliefs - self.natural_belief_target)
                
                self.users[user_id]['pp1_distance'] = pp1_distance
                self.users[user_id]['cluster_distances'] = cluster_distances
        
        distance_calc_time = time.time() - distance_calc_start
        total_time = time.time() - start_time
        
        print(f"  Belief calculation: {belief_calc_time:.3f}s")
        print(f"  Distance calculation: {distance_calc_time:.3f}s")
        print(f"  Updated {users_updated}/{len(users_to_update)} users")
        print(f"User beliefs updated successfully in {total_time:.3f}s total")
    
    def _calculate_new_beliefs(self, user_id: str, accepted_items: List[str]) -> np.ndarray:
        """
        Calculate new user beliefs based on accepted items.
        This is a simplified version - you may want to implement more sophisticated belief update logic.
        """
        current_beliefs = self.users[user_id]['beliefs'].copy()
        
        if not accepted_items:
            return current_beliefs
        
        # Load cluster assignments (using balanced clusters if configured)
        cluster_file = self.config['data_paths'].get('cluster_assignments', 'embeddings/cluster_matrix_manifest_K5_topk5.json')
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)
        
        # Create item to cluster mapping
        item_to_cluster = {}
        for cluster_id, items in cluster_data['cluster_to_items'].items():
            for item in items:
                item_to_cluster[item] = int(cluster_id)
        
        # Count interactions per cluster
        cluster_interactions = np.zeros(5)
        mapped_items = 0
        
        for item_id in accepted_items:
            # Try direct mapping first
            if item_id in item_to_cluster:
                cluster_id = item_to_cluster[item_id]
                cluster_interactions[cluster_id] += 1
                mapped_items += 1
            else:
                # Try with 'N' prefix (for pre-trained recommendations)
                item_with_n = f"N{item_id}"
                if item_with_n in item_to_cluster:
                    cluster_id = item_to_cluster[item_with_n]
                    cluster_interactions[cluster_id] += 1
                    mapped_items += 1
                else:
                    # Default to cluster 0 if no mapping found
                    cluster_interactions[0] += 1
                    mapped_items += 1
        

        
        # Update beliefs (simple additive model with decay)
        learning_rate = 0.05  # Increased from 0.01 for more noticeable changes
        
        if mapped_items > 0:
            interaction_distribution = cluster_interactions / mapped_items
            # Weighted update: current beliefs + learning_rate * new evidence
            new_beliefs = (1 - learning_rate) * current_beliefs + learning_rate * interaction_distribution
        else:
            new_beliefs = current_beliefs
        
        # Apply natural belief target constraints
        new_beliefs = self._apply_belief_constraints(new_beliefs)
        
        return new_beliefs.astype(np.float32)
    
    def _calculate_new_beliefs_optimized(self, user_id: str, accepted_items: List[str], item_to_cluster: Dict[str, int]) -> np.ndarray:
        """
        Optimized version of belief calculation that reuses cluster mapping.
        """
        current_beliefs = self.users[user_id]['beliefs'].copy()
        
        if not accepted_items:
            return current_beliefs
        
        # Count interactions per cluster
        cluster_interactions = np.zeros(5)
        mapped_items = 0
        
        for item_id in accepted_items:
            # Try direct mapping first
            if item_id in item_to_cluster:
                cluster_id = item_to_cluster[item_id]
                cluster_interactions[cluster_id] += 1
                mapped_items += 1
            else:
                # Try with 'N' prefix (for pre-trained recommendations)
                item_with_n = f"N{item_id}"
                if item_with_n in item_to_cluster:
                    cluster_id = item_to_cluster[item_with_n]
                    cluster_interactions[cluster_id] += 1
                    mapped_items += 1
                else:
                    # Default to cluster 0 if no mapping found
                    cluster_interactions[0] += 1
                    mapped_items += 1
        
        # Update beliefs (simple additive model with decay)
        learning_rate = 0.05  # Increased from 0.01 for more noticeable changes
        
        if mapped_items > 0:
            interaction_distribution = cluster_interactions / mapped_items
            # Weighted update: current beliefs + learning_rate * new evidence
            new_beliefs = (1 - learning_rate) * current_beliefs + learning_rate * interaction_distribution
        else:
            new_beliefs = current_beliefs
        
        # Apply natural belief target constraints
        new_beliefs = self._apply_belief_constraints(new_beliefs)
        
        return new_beliefs.astype(np.float32)
    
    def _apply_belief_constraints(self, beliefs: np.ndarray) -> np.ndarray:
        """
        Apply natural belief target constraints to prevent beliefs from exceeding natural limits.
        
        Args:
            beliefs: User's belief vector to constrain
            
        Returns:
            Constrained belief vector
        """
        if not hasattr(self, 'natural_belief_target'):
            return beliefs
        
        # Ensure beliefs don't exceed natural belief target
        constrained_beliefs = np.minimum(beliefs, self.natural_belief_target)
        
        return constrained_beliefs
    
    def _calculate_distance_based_reward(self, current_beliefs, previous_beliefs):
        """
        Calculate reward based on distance improvement to optimal beliefs.
        
        Args:
            current_beliefs: Current user belief vector
            previous_beliefs: Previous user belief vector
            
        Returns:
            Reward value (0 if no improvement, 1/distance average if improved)
        """
        if not hasattr(self, 'natural_belief_target'):
            return 0.0
        
        optimal_beliefs = self.natural_belief_target
        improved_rewards = []
        
        # Check each cluster for improvement
        for i in range(len(current_beliefs)):
            # Calculate distances to optimal belief for this cluster
            current_distance = abs(optimal_beliefs[i] - current_beliefs[i])
            previous_distance = abs(optimal_beliefs[i] - previous_beliefs[i])
            
            # Only reward if distance decreased (improvement)
            if current_distance < previous_distance:
                # Avoid division by zero - add small epsilon
                epsilon = 1e-6
                cluster_reward = 1.0 / (current_distance + epsilon)
                improved_rewards.append(cluster_reward)
        
        # Average reward across improved clusters
        if len(improved_rewards) > 0:
            total_reward = sum(improved_rewards) / len(improved_rewards)
            return total_reward
        else:
            return 0.0  # No improvement, no reward
    
    def calculate_combined_belief_score(self, user_beliefs, target_engagement=1.0):
        """
        Calculate combined engagement + alignment score for belief evaluation.
        
        Args:
            user_beliefs: User's belief vector
            target_engagement: Target total belief magnitude (default 1.0)
        
        Returns:
            dict with engagement_score, alignment_score, combined_score
        """
        user_array = np.array(user_beliefs)
        
        # Load optimal beliefs (you may want to cache this)
        optimal_beliefs = np.array([0.69958181, 0.07083116, 0.07386304, 0.07219028, 0.08353372])
        
        # 1. Engagement Score (0 to 1)
        # How close is the user's total belief magnitude to target?
        user_engagement = user_array.sum()
        engagement_score = min(user_engagement / target_engagement, 1.0)
        
        # 2. Alignment Score (0 to 1) 
        # How well does the belief shape match optimal shape?
        if user_engagement > 1e-8:
            # Normalize both to unit vectors for shape comparison
            user_normalized = user_array / user_engagement
            optimal_normalized = optimal_beliefs / optimal_beliefs.sum()
            
            # Cosine similarity (0 to 1)
            alignment_score = np.dot(user_normalized, optimal_normalized) / (
                np.linalg.norm(user_normalized) * np.linalg.norm(optimal_normalized)
            )
            alignment_score = max(0.0, alignment_score)  # Ensure non-negative
        else:
            alignment_score = 0.0  # No beliefs = no alignment
        
        # 3. Combined Score
        # Both engagement AND alignment must be good for high score
        combined_score = engagement_score * alignment_score
        
        return {
            'engagement_score': engagement_score,
            'alignment_score': alignment_score, 
            'combined_score': combined_score,
            'user_engagement': user_engagement
        }

    def calculate_episode_rewards(self, active_users=None) -> Dict[str, float]:
        """
        Simplified progressive reward based on distance to optimal beliefs.
        Reward = 1/distance for improved clusters, averaged across improved clusters.
        Only rewards improvement (distance reduction), otherwise reward = 0.
        """
        rewards = {}
        
        # Only calculate rewards for active users
        users_to_reward = active_users if active_users else self.users.keys()
        
        # Get max reward cap from config
        max_reward = self.config['reward']['max_reward']
        
        for user_id in users_to_reward:
            current_beliefs = self.users[user_id]['beliefs']
            
            # Get previous beliefs (if available)
            if user_id in self.previous_beliefs:
                previous_beliefs = self.previous_beliefs[user_id]
                reward = self._calculate_distance_based_reward(current_beliefs, previous_beliefs)
                
                # Apply max reward cap
                reward = min(reward, max_reward)
            else:
                # First episode - no previous beliefs to compare, give small baseline reward
                reward = 0.1
            
            rewards[user_id] = reward
            
            # Store current beliefs for next episode comparison
            self.previous_beliefs[user_id] = current_beliefs.copy()
        
        return rewards
    
    def calculate_round_rewards(self, active_users=None) -> Dict[str, float]:
        """
        Calculate immediate rewards for users based on THIS ROUND's performance only.
        Called after each round to track immediate recommendation quality.
        """
        rewards = {}
        
        # Only calculate rewards for active users
        users_to_reward = active_users if active_users else self.users.keys()
        
        for user_id in users_to_reward:
            if user_id in self.user_interactions:
                # Get THIS ROUND's interactions only
                round_recommended = len(self.user_interactions[user_id]['round_recommended'])
                round_accepted = len(self.user_interactions[user_id]['round_accepted'])
                
                # Calculate acceptance-based reward for this round
                if round_recommended > 0:
                    acceptance_rate = round_accepted / round_recommended
                    
                    # Dynamic reward based on acceptance rate and randomness
                    base_reward = acceptance_rate * 10  # Scale down for more variation
                    
                    # Add some randomness to make rewards more dynamic
                    import random
                    randomness_factor = random.uniform(0.8, 1.2)  # ±20% variation
                    
                    # Add diversity bonus for this round only
                    diversity_bonus = self._calculate_round_diversity_bonus(user_id)
                    
                    reward = (base_reward + diversity_bonus) * randomness_factor
                    reward = min(reward, self.config['reward']['max_reward'])
                else:
                    reward = 0.0
            else:
                reward = 0.0
            
            rewards[user_id] = reward
        
        return rewards
    
    def _calculate_diversity_bonus(self, user_id: str) -> float:
        """Calculate diversity bonus based on cluster distribution of ALL accepted items."""
        if user_id not in self.user_interactions:
            return 0.0
        
        accepted_items = self.user_interactions[user_id]['accepted_items']
        if not accepted_items:
            return 0.0
        
        # Load cluster mapping (could be optimized by caching) - using balanced clusters if configured
        try:
            cluster_file = self.config['data_paths'].get('cluster_assignments', 'embeddings/cluster_matrix_manifest_K5_topk5.json')
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            item_to_cluster = {}
            for cluster_id, items in cluster_data['cluster_to_items'].items():
                for item in items:
                    item_to_cluster[item] = int(cluster_id)
            
            # Count clusters represented in accepted items
            clusters_used = set()
            for item_id in accepted_items:
                if item_id in item_to_cluster:
                    clusters_used.add(item_to_cluster[item_id])
                elif f"N{item_id}" in item_to_cluster:
                    clusters_used.add(item_to_cluster[f"N{item_id}"])
            
            # Diversity bonus: more clusters = higher bonus
            diversity_bonus = len(clusters_used) * 2.0  # 2 points per cluster
            return diversity_bonus
            
        except Exception:
            return 0.0
    
    def _calculate_round_diversity_bonus(self, user_id: str) -> float:
        """Calculate diversity bonus based on THIS ROUND's accepted items only."""
        if user_id not in self.user_interactions:
            return 0.0
        
        round_accepted = self.user_interactions[user_id]['round_accepted']
        if not round_accepted:
            return 0.0
        
        # Simple diversity calculation for this round
        # More accepted items = higher bonus, with some randomness
        import random
        base_bonus = len(round_accepted) * 0.5  # 0.5 points per accepted item
        random_bonus = random.uniform(0, 2.0)   # Random bonus 0-2 points
        
        return base_bonus + random_bonus
    
    def reset_episode(self, active_users=None):
        """Reset environment for new episode."""
        self.current_episode += 1
        self.current_round = 0
        
        # Reset user beliefs to initial state (only for active users if specified)
        users_to_reset = active_users if active_users else self.users.keys()
        
        for user_id in users_to_reset:
            if user_id in self.users:
                # Reset beliefs to initial values
                self.users[user_id]['beliefs'] = self.initial_user_beliefs[user_id].copy()
                
                # Reset counts to initial values (based on initial beliefs)
                if user_id in self.initial_user_counts:
                    self.user_accepted_counts[user_id]['cluster_counts'] = self.initial_user_counts[user_id]['cluster_counts'].copy()
                
                # Calculate simple distance to natural target
                initial_beliefs = self.initial_user_beliefs[user_id]
                pp1_distance = np.abs(initial_beliefs - self.natural_belief_target).sum()
                self.users[user_id]['pp1_distance'] = pp1_distance
                # NOTE: cluster_distances should NOT be updated here!
                # It represents the fixed embedding space distance, not belief distance
                # The original cluster_distances from initialization should be preserved
        
        # Reset interactions
        self._reset_user_interactions()
        print(f"Starting episode {self.current_episode}")
    
    def next_round(self):
        """Move to next round within current episode."""
        self.current_round += 1
        # Reset round-level interaction tracking
        self.reset_round_interactions()
        print(f"Episode {self.current_episode}, Round {self.current_round}")
    
    def update_beliefs_after_round(self, active_users=None):
        """
        Optional: Update beliefs incrementally after each round.
        This would make round rewards more dynamic.
        """
        # Only update if we want dynamic beliefs within episodes
        if not self.config.get('environment', {}).get('update_beliefs_per_round', False):
            return
        
        users_to_update = active_users if active_users else self.users.keys()
        
        for user_id in users_to_update:
            if user_id in self.user_interactions:
                accepted_items = self.user_interactions[user_id]['accepted_items']
                
                if accepted_items:
                    # Small incremental belief update (much smaller learning rate)
                    current_beliefs = self.users[user_id]['beliefs'].copy()
                    
                    # Very small learning rate for round-by-round updates
                    learning_rate = 0.001  # Much smaller than episode-level (0.01)
                    
                    # Simple cluster counting for this round's items
                    cluster_interactions = np.zeros(5)
                    for item_id in accepted_items[-10:]:  # Only last 10 items (this round)
                        # Simplified cluster assignment (could be optimized)
                        cluster_id = hash(item_id) % 5  # Simple hash-based assignment
                        cluster_interactions[cluster_id] += 1
                    
                    if np.sum(cluster_interactions) > 0:
                        interaction_distribution = cluster_interactions / np.sum(cluster_interactions)
                        new_beliefs = (1 - learning_rate) * current_beliefs + learning_rate * interaction_distribution
                        self.users[user_id]['beliefs'] = new_beliefs.astype(np.float32)
                        
                        # Update pp1_distance only (cluster_distances should remain fixed)
                        pp1_distance = np.abs(new_beliefs - self.natural_belief_target).sum()
                        self.users[user_id]['pp1_distance'] = pp1_distance
                        # NOTE: cluster_distances should NOT be updated!
                        # It represents the fixed embedding space distance
    
    def get_statistics(self) -> Dict:
        """Get current environment statistics."""
        total_interactions = sum(
            len(data['accepted_items']) + len(data['rejected_items'])
            for data in self.user_interactions.values()
        )
        
        total_accepted = sum(
            len(data['accepted_items'])
            for data in self.user_interactions.values()
        )
        
        avg_pp1_distance = np.mean([user['pp1_distance'] for user in self.users.values()])
        avg_cluster_distance = np.mean([np.mean(user['cluster_distances']) for user in self.users.values()])
        
        return {
            'episode': self.current_episode,
            'round': self.current_round,
            'total_interactions': total_interactions,
            'total_accepted': total_accepted,
            'acceptance_rate': total_accepted / total_interactions if total_interactions > 0 else 0,
            'avg_pp1_distance': avg_pp1_distance,
            'avg_cluster_distance': avg_cluster_distance
        }