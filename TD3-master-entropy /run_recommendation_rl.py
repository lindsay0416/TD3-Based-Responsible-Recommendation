#!/usr/bin/env python3
"""
Single entry point for running the Recommendation RL System.
This is the main file to execute the entire training process.
"""

import sys
import os
import argparse
import yaml
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_trainer import RecommendationTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def resolve_data_paths(config: dict) -> dict:
    """
    Resolve template placeholders in data_paths using experiment_params.
    Replaces {topk}, {rs_model}, {dataset} with actual values.
    """
    if 'experiment_params' not in config:
        print("Warning: No experiment_params found, using data_paths as-is")
        return config
    
    params = config['experiment_params']
    topk = str(params.get('topk', 5))
    rs_model = params.get('rs_model', 'LightGCN')
    dataset = params.get('dataset', 'news')
    
    print(f"\n{'='*60}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"  TopK: {topk}")
    print(f"  RS Model: {rs_model}")
    print(f"{'='*60}")
    
    # Resolve all paths in data_paths
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


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_sections = ['training', 'td3', 'environment', 'data_paths']
    
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section: {section}")
            return False
    
    # Validate batch size
    batch_size = config['training'].get('batch_size', 1000)
    users_per_round = config['training'].get('users_per_round', 1000)
    
    if batch_size <= 0:
        print(f"Error: batch_size must be positive, got {batch_size}")
        return False
    
    if batch_size > users_per_round:
        print(f"Warning: batch_size ({batch_size}) > users_per_round ({users_per_round})")
        print("This will be automatically adjusted to users_per_round")
    
    # Validate data paths exist
    for path_name, path_value in config['data_paths'].items():
        if not os.path.exists(path_value):
            print(f"Warning: Data file not found: {path_name} = {path_value}")
    
    return True


def print_config_summary(config: dict):
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Training Episodes: {config['training']['episodes']}")
    print(f"Rounds per Episode: {config['training']['rounds_per_episode']}")
    print(f"Users per Round: {config['training']['users_per_round']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Total Rounds: {config['training']['episodes'] * config['training']['rounds_per_episode']}")
    
    print(f"\nTD3 Parameters:")
    print(f"  Discount (γ): {config['td3']['discount']}")
    print(f"  Batch Size: {config['td3']['batch_size']}")
    print(f"  Learning Rate: {config['td3']['learning_rate']}")
    
    print(f"\nEnvironment:")
    print(f"  State Dimension: {config['environment']['state_dim']}")
    print(f"  Action Dimension: {config['environment']['action_dim']}")
    print(f"  User Acceptance Rate: {config['environment']['user_acceptance_rate']}")
    
    print(f"\nRecommendation:")
    print(f"  RL List Size: {config['recommendation']['rl_list_size']}")
    print(f"  Total Recommendations: {config['recommendation']['total_recommendations']}")
    
    print(f"\nReward (Step-based):")
    print(f"  Termination Threshold: {config['reward']['termination_threshold']}")
    print(f"  Termination Reward: {config['reward']['termination_reward']}")
    print(f"  Max Cluster Distance: {config['reward']['max_cluster_distance']}")
    print(f"  Use GPU Rewards: {config['reward']['use_gpu_rewards']}")
    
    print("="*60)


def main():
    """Main function - single entry point for the entire system."""
    parser = argparse.ArgumentParser(
        description="Recommendation RL System using TD3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_recommendation_rl.py                    # Use default config.yaml
  python run_recommendation_rl.py --config my_config.yaml  # Use custom config
  python run_recommendation_rl.py --validate-only    # Only validate configuration
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running training'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("RECOMMENDATION RL SYSTEM")
    print("Twin Delayed Deep Deterministic Policy Gradient (TD3)")
    print("for Recommendation System Filter Bubble Reduction")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Resolve template paths from experiment_params
    config = resolve_data_paths(config)
    
    # Override verbose setting if specified
    if args.verbose:
        config['debug']['verbose'] = True
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed!")
        sys.exit(1)
    
    # Print configuration summary
    print_config_summary(config)
    
    # If validate-only mode, exit here
    if args.validate_only:
        print("\nConfiguration validation completed successfully!")
        print("Use without --validate-only flag to start training.")
        return
    
    # Confirm before starting training
    total_rounds = config['training']['episodes'] * config['training']['rounds_per_episode']
    estimated_time = total_rounds * 0.1  # Rough estimate: 0.1 seconds per round
    
    print(f"\nEstimated training time: {estimated_time/3600:.1f} hours")
    print("This will train a TD3 agent to reduce filter bubbles in recommendations.")
    
    try:
        response = input("\nProceed with training? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training cancelled by user.")
            return
    except KeyboardInterrupt:
        print("\nTraining cancelled by user.")
        return
    
    # Initialize and run trainer
    try:
        print("\nInitializing trainer...")
        trainer = RecommendationTrainer(args.config)
        
        print("Starting training...")
        training_stats = trainer.train()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print final statistics
        final_episode = training_stats[-1]
        print(f"Final Episode Statistics:")
        print(f"  Total Transitions: {final_episode['total_transitions']}")
        print(f"  Average Acceptance Rate: {final_episode['avg_acceptance_rate']:.3f}")
        print(f"  Users with Belief Improvements: {final_episode['belief_improvements']}")
        print(f"  Average Step Reward: {final_episode['avg_step_reward']:.6f}")
        print(f"  Average Round Reward: {final_episode['avg_round_reward']:.2f}")
        print(f"  Total Reward: {final_episode['total_reward']:.2f}")
        
        print(f"\nResults saved in: {config['logging']['results_dir']}")
        print(f"Models saved in: {config['logging']['models_dir']}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results have been saved automatically.")
        print(f"Check {config['logging']['results_dir']} for:")
        print("   - training_all_episodes.json (partial progress)")
        print("   - current_user_beliefs.json (latest user beliefs)")
        print("   - interrupted_* model files")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Check the configuration and data files.")
        if config['debug']['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()