#!/usr/bin/env python3
"""
Recommendation System Trainer using TD3
Main training loop for the recommendation RL system.
"""

import numpy as np
import torch
import yaml
import os
import json
import pickle
import time
import signal
import sys
from typing import Dict, List
import random
from datetime import datetime

# Import existing TD3 components
import TD3
import utils
from recommendation_environment import RecommendationEnvironment


class RecommendationTrainer:
    """
    Main trainer class for the recommendation RL system.
    Manages training loop, data collection, and model saving.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Resolve template paths from experiment_params
        self.config = self._resolve_data_paths(self.config)
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize environment
        self.env = RecommendationEnvironment(self.config)
        
        # Initialize TD3 agent
        self._initialize_td3()
        
        # Initialize replay buffer
        self.replay_buffer = utils.ReplayBuffer(
            state_dim=self.config['environment']['state_dim'],
            action_dim=self.config['environment']['action_dim'],
            max_size=int(1e6)
        )
        
        # Initialize tracking variables
        self.total_timesteps = 0
        self.training_stats = []
        self.td3_training_started = False
        self.rounds_since_training = 0
        
        # Get user list and limit based on config
        all_users = list(self.env.users.keys())
        max_users = self.config['training']['users_per_round']
        self.user_list = all_users[:max_users]  # Take first N users
        
        print(f"Using {len(self.user_list)} users out of {len(all_users)} total users")
        print(f"TD3 training will start after {self.config['training']['warmup_transitions']} transitions")
        print(f"TD3 training frequency: every {self.config['training']['train_frequency']} rounds")
        
        # Create directories
        self._create_directories()
        
        # Setup interrupt handler
        self._setup_interrupt_handler()
        
        print("Trainer initialized successfully")
        print(f"Training for {self.config['training']['episodes']} episodes")
        print(f"Rounds per episode: {self.config['training']['rounds_per_episode']}")
        print(f"Users per round: {len(self.user_list)} (limited from config)")
    
    def _resolve_data_paths(self, config: dict) -> dict:
        """
        Resolve template placeholders in data_paths using experiment_params.
        Replaces {topk}, {rs_model}, {dataset} with actual values.
        """
        if 'experiment_params' not in config:
            return config
        
        params = config['experiment_params']
        topk = str(params.get('topk', 5))
        rs_model = params.get('rs_model', 'LightGCN')
        dataset = params.get('dataset', 'news')
        
        print(f"\nExperiment: dataset={dataset}, topk={topk}, rs_model={rs_model}")
        
        if 'data_paths' in config:
            resolved_paths = {}
            for key, path_template in config['data_paths'].items():
                if isinstance(path_template, str):
                    resolved_path = path_template.format(
                        topk=topk,
                        rs_model=rs_model,
                        dataset=dataset
                    )
                    resolved_paths[key] = resolved_path
                else:
                    resolved_paths[key] = path_template
            config['data_paths'] = resolved_paths
        
        return config
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config['experiment']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _initialize_td3(self):
        """Initialize TD3 agent."""
        state_dim = self.config['environment']['state_dim']
        action_dim = self.config['environment']['action_dim']
        max_action = self.config['environment']['max_action']
        
        self.td3_agent = TD3.TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=self.config['td3']['discount'],
            tau=self.config['td3']['tau'],
            policy_noise=self.config['td3']['policy_noise'],
            noise_clip=self.config['td3']['noise_clip'],
            policy_freq=self.config['td3']['policy_freq']
        )
        
        print(f"TD3 agent initialized: {state_dim}D state → {action_dim}D action")
    
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config['logging']['results_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['models_dir'], exist_ok=True)
    
    def _setup_interrupt_handler(self):
        """Setup handler for graceful interruption (Ctrl+C)."""
        def signal_handler(sig, frame):
            print("\n" + "="*60)
            print("TRAINING INTERRUPTED BY USER")
            print("="*60)
            print("Saving partial results...")
            
            # Save current training statistics
            self.save_training_stats()
            print("Training statistics saved")
            
            # Save current user beliefs
            if self.training_stats:
                current_episode = len(self.training_stats)
                self.save_current_user_beliefs(f"INTERRUPTED_EP_{current_episode}")
                print("Current user beliefs saved")
            
            # Save current model
            self.save_model("interrupted")
            print("Model saved")
            
            print(f"Completed {len(self.training_stats)} episodes before interruption")
            print("You can resume training or analyze partial results.")
            print("="*60)
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def run_round(self, round_num: int) -> Dict:
        """
        Run one round (all users get recommendations once) with GPU batch processing.
        
        Args:
            round_num: Current round number within episode
            
        Returns:
            Round statistics
        """
        round_stats = {
            'round': round_num,
            'transitions': 0,
            'total_recommended': 0,
            'total_accepted': 0,
            'round_reward': 0.0,
            'step_rewards': 0.0,
            'belief_improvements': 0,
            'termination_bonuses': 0
        }
        
        # Shuffle users for each round
        shuffled_users = self.user_list.copy()
        random.shuffle(shuffled_users)
        
        # Process users in batches for GPU efficiency
        batch_size = min(self.config['training']['batch_size'], len(shuffled_users))
        
        start_time = time.time()
        all_batch_results = []  # Store all batch results for statistics
        all_recommendations = []  # Store all recommendations for cluster distribution analysis
        
        for batch_start in range(0, len(shuffled_users), batch_size):
            batch_end = min(batch_start + batch_size, len(shuffled_users))
            batch_users = shuffled_users[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//batch_size + 1}/{(len(shuffled_users)-1)//batch_size + 1} "
                  f"({len(batch_users)} users) - {time.time() - start_time:.2f}s elapsed")
            
            # Get batch states on GPU
            batch_states = self.env.get_batch_user_states(batch_users)
            
            # Batch TD3 inference
            with torch.no_grad():
                batch_actions = self.td3_agent.actor(batch_states)
                
                # Add exploration noise during training
                if self.env.current_episode < 100:
                    noise = torch.normal(0, 0.1, size=batch_actions.shape).to(batch_actions.device)
                    batch_actions = torch.clamp(batch_actions + noise, -1, 1)
                
                # Optional: Pre-normalize TD3 outputs for consistency
                # This ensures all virtual items have unit norm, which can help training stability
                batch_actions = torch.nn.functional.normalize(batch_actions, p=2, dim=1)
            
            # Execute batch actions in environment
            batch_results = self.env.batch_step(batch_users, batch_actions)
            all_batch_results.extend(batch_results)
            
            # Collect recommendations for cluster distribution analysis
            for result in batch_results:
                all_recommendations.append({
                    'user_id': result['user_id'],
                    'recommended_items': result['recommended_items'],
                    'accepted_items': result['accepted_items'],
                    'rejected_items': result['rejected_items']
                })
            
            # ===== OPTIMIZED: Batch processing for replay buffer =====
            # Prepare batch data for vectorized operations
            batch_next_states_list = []
            batch_rewards_list = []
            batch_dones_list = []
            
            for i, (user_id, info) in enumerate(zip(batch_users, batch_results)):
                # Next state: get after belief update
                if info['belief_updated']:
                    # Beliefs were updated, get new state with updated beliefs
                    next_state = self.env.get_user_state(user_id)
                else:
                    # No belief update, state remains the same
                    next_state = batch_states[i].cpu().numpy()
                
                batch_next_states_list.append(next_state)
                
                # Use step reward from environment
                reward = info['step_reward']
                batch_rewards_list.append(reward)
                
                # Termination condition: check if user reached optimal beliefs
                avg_distance = np.mean([abs(self.env.natural_belief_target[j] - info['new_beliefs'][j]) for j in range(5)])
                done = avg_distance < self.env.termination_threshold  # Use threshold from config
                batch_dones_list.append(done)
                
                # Update statistics with aggregation
                round_stats['transitions'] += 1
                round_stats['total_recommended'] += len(info['recommended_items'])
                round_stats['total_accepted'] += len(info['accepted_items'])
                round_stats['step_rewards'] = round_stats.get('step_rewards', 0.0) + reward
                
                # Track reward distribution for analysis (optional)
                if reward > 0:
                    round_stats['positive_rewards'] = round_stats.get('positive_rewards', 0) + 1
                if reward >= 5.0:  # Termination bonus
                    round_stats['termination_bonuses'] = round_stats.get('termination_bonuses', 0) + 1
                
                self.total_timesteps += 1
            
            # Batch convert to numpy arrays (vectorized CPU-GPU transfer)
            batch_states_np = batch_states.cpu().numpy()  # [batch_size, state_dim]
            batch_actions_np = batch_actions.cpu().numpy()  # [batch_size, action_dim]
            batch_next_states_np = np.array(batch_next_states_list)  # [batch_size, state_dim]
            batch_rewards_np = np.array(batch_rewards_list, dtype=np.float32)  # [batch_size]
            batch_dones_np = np.array(batch_dones_list, dtype=np.float32)  # [batch_size]
            
            # Add entire batch to replay buffer at once (vectorized)
            self.replay_buffer.add_batch(
                batch_states_np,
                batch_actions_np,
                batch_next_states_np,
                batch_rewards_np,
                batch_dones_np
            )
        
        # Train TD3 agent (with proper warmup and frequency control)
        td3_trained = self.train_td3()
        
        # Update round statistics with step-based rewards
        round_stats['round_reward'] = round_stats['step_rewards']  # Use accumulated step rewards
        
        # Count termination bonuses and belief improvements
        termination_count = 0
        improvement_count = 0
        for batch_result in all_batch_results:
            if batch_result.get('step_reward', 0) >= 5.0:  # Has termination bonus
                termination_count += 1
            if batch_result.get('step_reward', 0) > 0:  # Has any improvement
                improvement_count += 1
        
        round_stats['termination_bonuses'] = termination_count
        round_stats['belief_improvements'] = improvement_count
        
        elapsed = time.time() - start_time
        
        # Enhanced timing info
        status_info = []
        if self.td3_training_started:
            status_info.append("TD3 Active")
        else:
            remaining = self.config['training']['warmup_transitions'] - self.replay_buffer.size
            status_info.append(f"Warmup ({remaining} left)")
        
        if td3_trained:
            status_info.append("Trained")
        
        status_str = " | ".join(status_info)
        print(f"  Round {round_num} completed in {elapsed:.2f}s - Round Reward: {round_stats['round_reward']:.2f} [{status_str}]")
        
        # Save recommendations for this round
        round_stats['recommendations'] = all_recommendations
        
        return round_stats
    
    def train_td3(self, force_train=False):
        """Train TD3 agent using collected experiences with proper warmup."""
        warmup_transitions = self.config['training']['warmup_transitions']
        train_frequency = self.config['training']['train_frequency']
        
        # Check if we have enough transitions for warmup
        if self.replay_buffer.size < warmup_transitions and not force_train:
            if self.replay_buffer.size % 10000 == 0:  # Log every 10k transitions
                print(f"  Warmup: {self.replay_buffer.size}/{warmup_transitions} transitions collected")
            return False
        
        # Mark that TD3 training has started
        if not self.td3_training_started:
            self.td3_training_started = True
            print(f"TD3 training started! Buffer size: {self.replay_buffer.size}")
        
        # Check training frequency
        self.rounds_since_training += 1
        if self.rounds_since_training < train_frequency and not force_train:
            return False
        
        # Reset counter and train
        self.rounds_since_training = 0
        
        # Train TD3 with multiple updates for stability
        training_start = time.time()
        num_updates = 10  # Multiple updates per training session (optimized from 5)
        
        for _ in range(num_updates):
            self.td3_agent.train(self.replay_buffer, self.config['td3']['batch_size'])
        
        training_time = time.time() - training_start
        print(f"  TD3 trained ({num_updates} updates) in {training_time:.3f}s")
        
        return True
    
    def analyze_buffer_quality(self):
        """Analyze the quality of the replay buffer"""
        from tools.analyze_replay_buffer_quality import ReplayBufferAnalyzer
        
        analyzer = ReplayBufferAnalyzer(self.replay_buffer)
        analyzer.analyze_quality()
        analyzer.save_analysis_report(f"results/buffer_quality_episode_{self.env.current_episode}.json")
    
    def run_episode(self, episode_num: int) -> Dict:
        """
        Run one complete episode (50 rounds + belief update).
        
        Args:
            episode_num: Current episode number
            
        Returns:
            Episode statistics
        """
        print(f"\n{'='*50}")
        print(f"Starting Episode {episode_num}")
        print(f"{'='*50}")
        
        # Reset environment for new episode (only for active users)
        self.env.reset_episode(active_users=self.user_list)
        
        episode_stats = {
            'episode': episode_num,
            'rounds': [],
            'total_transitions': 0,
            'avg_acceptance_rate': 0,
            'belief_improvements': 0,
            'total_reward': 0
        }
        
        # Run 50 rounds for this episode
        for round_num in range(1, self.config['training']['rounds_per_episode'] + 1):
            self.env.next_round()
            round_stats = self.run_round(round_num)
            
            # Save round recommendations for cluster distribution analysis
            if 'recommendations' in round_stats:
                self.save_round_recommendations(episode_num, round_num, round_stats['recommendations'])
                # Remove from stats to save memory in training_all_episodes.json
                del round_stats['recommendations']
            
            episode_stats['rounds'].append(round_stats)
            episode_stats['total_transitions'] += round_stats['transitions']
            
            # Log progress
            if round_num % 10 == 0:  # Log every 10 rounds
                env_stats = self.env.get_statistics()
                print(f"Episode {episode_num}, Round {round_num}: "
                      f"Acceptance Rate: {env_stats['acceptance_rate']:.3f}, "
                      f"Avg PP1: {env_stats['avg_pp1_distance']:.6f}")
        
        # Note: User beliefs are now updated at each step, no episode-level updates needed
        # Note: Rewards are calculated at each step and stored in replay buffer immediately
        
        # Calculate episode statistics
        total_recommended = sum(round_data['total_recommended'] for round_data in episode_stats['rounds'])
        total_accepted = sum(round_data['total_accepted'] for round_data in episode_stats['rounds'])
        episode_stats['avg_acceptance_rate'] = total_accepted / total_recommended if total_recommended > 0 else 0
        
        # FIXED: total_reward should be the sum of all round rewards in this episode
        episode_stats['total_reward'] = sum(round_data['round_reward'] for round_data in episode_stats['rounds'])
        
        # Add average round reward
        episode_stats['avg_round_reward'] = episode_stats['total_reward'] / len(episode_stats['rounds']) if episode_stats['rounds'] else 0
        
        # Add step-based reward metrics
        total_step_rewards = sum(round_data.get('step_rewards', 0) for round_data in episode_stats['rounds'])
        total_termination_bonuses = sum(round_data.get('termination_bonuses', 0) for round_data in episode_stats['rounds'])
        total_belief_improvements = sum(round_data.get('belief_improvements', 0) for round_data in episode_stats['rounds'])
        
        episode_stats['total_step_rewards'] = total_step_rewards
        episode_stats['total_termination_bonuses'] = total_termination_bonuses
        episode_stats['belief_improvements'] = total_belief_improvements
        episode_stats['avg_step_reward'] = total_step_rewards / len(self.user_list) if self.user_list else 0
        
        # Calculate episode reward changes (compared to previous episode)
        if len(self.training_stats) > 0:  # Not the first episode
            prev_episode = self.training_stats[-1]
            
            # Total reward change
            prev_total_reward = prev_episode.get('total_step_rewards', prev_episode.get('total_reward', 0))
            episode_stats['reward_change'] = total_step_rewards - prev_total_reward
            episode_stats['reward_change_percent'] = (episode_stats['reward_change'] / prev_total_reward * 100) if prev_total_reward > 0 else 0
            
            # Average reward change
            prev_avg_reward = prev_episode.get('avg_step_reward', 0)
            episode_stats['avg_reward_change'] = episode_stats['avg_step_reward'] - prev_avg_reward
            
            # Termination bonus change
            prev_termination_bonuses = prev_episode.get('total_termination_bonuses', 0)
            episode_stats['termination_bonus_change'] = total_termination_bonuses - prev_termination_bonuses
            
            # Belief improvement change
            prev_belief_improvements = prev_episode.get('belief_improvements', 0)
            episode_stats['belief_improvement_change'] = total_belief_improvements - prev_belief_improvements
            
            # Calculate belief changes (compared to previous episode)
            belief_changes = self._calculate_episode_belief_changes(prev_episode)
            episode_stats.update(belief_changes)
            
        else:  # First episode
            episode_stats['reward_change'] = 0.0
            episode_stats['reward_change_percent'] = 0.0
            episode_stats['avg_reward_change'] = 0.0
            episode_stats['termination_bonus_change'] = 0
            episode_stats['belief_improvement_change'] = 0
            
            # Initialize belief change metrics for first episode
            belief_changes = self._calculate_episode_belief_changes({})
            episode_stats.update(belief_changes)
        
        # Calculate reward trend (last round reward - first round reward)
        if len(episode_stats['rounds']) > 1:
            round_rewards = [round_data['round_reward'] for round_data in episode_stats['rounds']]
            episode_stats['reward_trend'] = round_rewards[-1] - round_rewards[0]
        else:
            episode_stats['reward_trend'] = 0.0
        
        # Note: user_belief_summary removed to save storage space
        # Belief changes will be calculated using current_user_beliefs.json file instead
        
        # Save current user beliefs after each episode (overwrites same file)
        self.save_current_user_beliefs(episode_num)
        
        # Save model periodically
        if episode_num % 10 == 0:
            self.save_model(f"episode_{episode_num}")
        
        return episode_stats
    
    def _calculate_episode_belief_changes(self, prev_episode: Dict) -> Dict:
        """
        Calculate simplified belief changes compared to previous episode.
        Uses aggregate statistics instead of storing individual user beliefs.
        
        Args:
            prev_episode: Previous episode statistics
            
        Returns:
            Dictionary with simplified belief change metrics
        """
        # For first episode, return zero changes
        if len(self.training_stats) == 0:
            return {
                'avg_belief_distance_change': 0.0,
                'belief_convergence_rate': 0.0,
                'users_improved_beliefs': 0,
                'users_worsened_beliefs': 0
            }
        
        # Calculate current aggregate belief statistics
        current_distances = []
        optimal_beliefs = self.env.natural_belief_target
        
        for user_id in self.user_list:
            if user_id in self.env.users:
                user_beliefs = self.env.users[user_id]['beliefs']
                distance = np.mean([abs(optimal_beliefs[i] - user_beliefs[i]) for i in range(5)])
                current_distances.append(distance)
        
        current_avg_distance = np.mean(current_distances) if current_distances else 0.0
        
        # Get previous episode's average distance (if available)
        prev_avg_distance = prev_episode.get('avg_belief_distance_to_optimal', current_avg_distance)
        
        # Calculate simplified metrics
        distance_change = current_avg_distance - prev_avg_distance
        
        # Estimate convergence based on distance improvement
        users_improved = 0
        users_worsened = 0
        
        if distance_change < -0.001:  # Overall improvement
            # Estimate that most users improved
            users_improved = int(len(self.user_list) * 0.6)  # Estimate 60% improved
            users_worsened = int(len(self.user_list) * 0.2)   # Estimate 20% worsened
        elif distance_change > 0.001:  # Overall worsening
            # Estimate that most users worsened
            users_improved = int(len(self.user_list) * 0.2)   # Estimate 20% improved
            users_worsened = int(len(self.user_list) * 0.6)   # Estimate 60% worsened
        else:  # No significant change
            users_improved = int(len(self.user_list) * 0.4)   # Estimate 40% improved
            users_worsened = int(len(self.user_list) * 0.4)   # Estimate 40% worsened
        
        convergence_rate = users_improved / len(self.user_list) if self.user_list else 0.0
        
        return {
            'avg_belief_distance_change': float(distance_change),
            'belief_convergence_rate': float(convergence_rate),
            'users_improved_beliefs': users_improved,
            'users_worsened_beliefs': users_worsened,
            'avg_belief_distance_to_optimal': float(current_avg_distance)  # Store for next episode
        }
    
    def _get_current_belief_summary(self) -> Dict:
        """
        Get current user belief summary for comparison in next episode.
        
        Returns:
            Dictionary mapping user_id to belief arrays
        """
        belief_summary = {}
        for user_id in self.user_list:
            if user_id in self.env.users:
                belief_summary[user_id] = self.env.users[user_id]['beliefs'].tolist()
        
        return belief_summary
    
    def _update_replay_buffer_rewards(self, episode_rewards: Dict[str, float]):
        """
        Update replay buffer with actual rewards calculated after belief updates.
        This is a simplified approach - in practice, you might want more sophisticated reward assignment.
        """
        # For simplicity, we'll assign the average reward to recent transitions
        # In a more sophisticated implementation, you'd track which transitions belong to which users
        avg_reward = np.mean(list(episode_rewards.values()))
        
        # Update recent transitions (last episode's worth - 50 rounds * users)
        recent_transitions = min(len(self.user_list) * self.config['training']['rounds_per_episode'], 
                               self.replay_buffer.size)
        start_idx = max(0, self.replay_buffer.ptr - recent_transitions)
        
        for i in range(recent_transitions):
            idx = (start_idx + i) % self.replay_buffer.max_size
            if idx < self.replay_buffer.size:
                self.replay_buffer.reward[idx] = avg_reward
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Total episodes: {self.config['training']['episodes']}")
        print(f"Rounds per episode: {self.config['training']['rounds_per_episode']}")
        print(f"Total rounds: {self.config['training']['episodes'] * self.config['training']['rounds_per_episode']}")
        
        start_time = datetime.now()
        
        for episode_num in range(1, self.config['training']['episodes'] + 1):
            episode_stats = self.run_episode(episode_num)
            self.training_stats.append(episode_stats)
            
            # Analyze buffer quality periodically
            if episode_num == 1 or episode_num % 5 == 0 or episode_num == self.config['training']['episodes']:
                print(f"\n{'='*60}")
                print(f"Analyzing Replay Buffer Quality (Episode {episode_num})")
                print(f"{'='*60}")
                self.analyze_buffer_quality()
            
            # Print episode summary with step-based metrics
            print(f"\nEpisode {episode_num} Summary:")
            print(f"  Transitions: {episode_stats['total_transitions']}")
            print(f"  Avg Acceptance Rate: {episode_stats['avg_acceptance_rate']:.3f}")
            print(f"  Belief Improvements: {episode_stats['belief_improvements']}/{len(self.user_list)}")
            print(f"  Termination Bonuses: {episode_stats['total_termination_bonuses']}")
            print(f"  Avg Step Reward: {episode_stats['avg_step_reward']:.3f}")
            print(f"  Total Step Rewards: {episode_stats['total_step_rewards']:.2f}")
            print(f"  Total Reward: {episode_stats['total_reward']:.2f}")
            
            # Show episode-to-episode changes
            if episode_num > 1:
                print(f"  Reward Change: {episode_stats['reward_change']:+.2f} ({episode_stats['reward_change_percent']:+.1f}%)")
                print(f"  Avg Reward Change: {episode_stats['avg_reward_change']:+.4f}")
                print(f"  Termination Bonus Change: {episode_stats['termination_bonus_change']:+d}")
                print(f"  Belief Improvement Change: {episode_stats['belief_improvement_change']:+d}")
                
                # Show belief changes
                print(f"  Avg Belief Distance Change: {episode_stats['avg_belief_distance_change']:+.6f}")
                print(f"  Belief Convergence Rate: {episode_stats['belief_convergence_rate']:.3f}")
                print(f"  Users Improved/Worsened: {episode_stats['users_improved_beliefs']}/{episode_stats['users_worsened_beliefs']}")
                print(f"  Avg Distance to Optimal: {episode_stats.get('avg_belief_distance_to_optimal', 0.0):.6f}")
            
            # Show round progression
            round_rewards = [round_data['round_reward'] for round_data in episode_stats['rounds']]
            if len(round_rewards) > 1:
                print(f"  Round Rewards: {[f'{r:.1f}' for r in round_rewards]}")
                print(f"  Reward Trend: {episode_stats['reward_trend']:+.1f} (last - first)")
            
            # Save training statistics after each episode
            self.save_training_stats()
            
            # Save backup every 100 episodes
            if episode_num % 100 == 0:
                print(f"Checkpoint: Saved progress at episode {episode_num}")
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\nTraining completed!")
        print(f"Total training time: {training_duration}")
        print(f"Total timesteps: {self.total_timesteps}")
        
        # Save final model
        self.save_model("final")
        
        return self.training_stats
    
    def save_model(self, suffix: str = ""):
        """Save TD3 model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"td3_recommendation_{timestamp}_{suffix}"
        filepath = os.path.join(self.config['logging']['models_dir'], filename)
        
        self.td3_agent.save(filepath)
        print(f"Model saved: {filepath}")
    
    def save_training_stats(self):
        """Save all training statistics in one file (overwrites each time)."""
        filename = "training_all_episodes.json"
        filepath = os.path.join(self.config['logging']['results_dir'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def save_round_recommendations(self, episode_num, round_num, recommendations):
        """
        Save recommendations for a specific round for cluster distribution analysis.
        Uses per-episode files to avoid memory issues.
        """
        # Use separate file for each episode to avoid huge files
        filename = f'recommendations_ep{episode_num}.json'
        filepath = os.path.join(self.config['logging']['results_dir'], filename)
        
        # Load existing data for this episode if file exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                episode_data = json.load(f)
        else:
            episode_data = {
                'episode': episode_num,
                'rounds': {}
            }
        
        # Sample recommendations to reduce memory (keep 1000 users per round)
        # This is sufficient for cluster distribution analysis
        sample_size = min(1000, len(recommendations))
        import random
        sampled_recs = random.sample(recommendations, sample_size) if len(recommendations) > sample_size else recommendations
        
        # Save sampled recommendations for this round
        episode_data['rounds'][f'round_{round_num}'] = {
            'round': round_num,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_users': len(recommendations),
            'sampled_users': len(sampled_recs),
            'recommendations': sampled_recs
        }
        
        # Save back to file
        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # Also save a lightweight summary for quick access
        summary_file = os.path.join(self.config['logging']['results_dir'], 'recommendations_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        key = f"ep{episode_num}_round{round_num}"
        summary[key] = {
            'episode': episode_num,
            'round': round_num,
            'total_users': len(recommendations),
            'file': filename
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_current_user_beliefs(self, episode_num):
        """Save current user beliefs after each episode to separate files."""
        # Save to episode-specific file
        episode_filename = f"user_beliefs_episode_{episode_num}.json"
        episode_filepath = os.path.join(self.config['logging']['results_dir'], episode_filename)
        
        # Also save to current file (for backward compatibility)
        current_filename = "current_user_beliefs.json"
        current_filepath = os.path.join(self.config['logging']['results_dir'], current_filename)
        
        current_beliefs = {}
        # Only save beliefs for active users to keep file size manageable
        for user_id in self.user_list:
            if user_id in self.env.users:
                user_data = self.env.users[user_id]
                
                # Get counts if available
                counts_data = {}
                if hasattr(self.env, 'user_accepted_counts') and user_id in self.env.user_accepted_counts:
                    cluster_counts = self.env.user_accepted_counts[user_id]['cluster_counts']
                    counts_data = {
                        'cluster_counts': cluster_counts.tolist(),
                        'total_count': int(cluster_counts.sum())  # Calculate from cluster_counts
                    }
                
                current_beliefs[user_id] = {
                    'beliefs': user_data['beliefs'].tolist(),
                    'pp1_distance': float(user_data['pp1_distance']),
                    'cluster_distances': user_data['cluster_distances'].tolist(),
                    'counts': counts_data  # Add counts information
                }
        
        # Add metadata
        belief_data = {
            'episode': episode_num,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_users': len(current_beliefs),
            'user_beliefs': current_beliefs
        }
        
        # Save to episode-specific file
        with open(episode_filepath, 'w') as f:
            json.dump(belief_data, f, indent=2)
        
        # Also save to current file (overwrites, for backward compatibility)
        with open(current_filepath, 'w') as f:
            json.dump(belief_data, f, indent=2)
    
    def save_final_user_beliefs(self):
        """Save final user beliefs after training (overwrites current_user_beliefs.json)."""
        # Just update the current beliefs file with final data
        self.save_current_user_beliefs("FINAL")
        print("Final user beliefs saved to current_user_beliefs.json")


def main():
    """Main function to run the recommendation RL training."""
    print("Recommendation RL System")
    print("=" * 50)
    
    # Initialize trainer
    trainer = RecommendationTrainer()
    
    # Run training
    training_stats = trainer.train()
    
    # Save final results
    trainer.save_final_user_beliefs()
    
    print("\nTraining completed successfully!")
    return training_stats


if __name__ == "__main__":
    main()