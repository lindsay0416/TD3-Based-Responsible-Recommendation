# Natural Belief Target Generation

This tool generates `natural_belief_target.json` from cluster manifests with full configuration control.

## Quick Start

1. **Edit the configuration file** `data_preprocessing/natural_target_config.yaml`:
   ```yaml
   input_files:
     cluster_manifest: "path/to/your/cluster_manifest.json"
   
   output_files:
     natural_belief_target_json: "processed_data/natural_belief_target.json"
   ```

2. **Run the generator**:
   ```bash
   python data_preprocessing/generate_natural_target.py
   ```

3. **Output files generated**:
   - `processed_data/natural_belief_target.json` (main target file)
   - `processed_data/natural_belief_target.pkl` (fast loading)
   - `processed_data/filter_bubble_utilities.py` (measurement functions)

## Configuration Options

### Input Files
- `cluster_manifest`: JSON file with cluster-to-items mapping

### Output Files
- `natural_belief_target_json`: Main target file with metadata
- `natural_belief_target_pkl`: Binary file for fast RL loading
- `filter_bubble_utilities`: Python utility functions
- `cluster_size_analysis`: Optional detailed analysis file

### Processing Methods
Choose how to calculate natural beliefs:

- **`proportional`** (default): Direct proportion of cluster sizes
  - Best for: Balanced clusters, natural distribution
  - Formula: `belief[i] = cluster_size[i] / total_items`

- **`sqrt_normalized`**: Reduces large cluster dominance
  - Best for: Very unbalanced clusters (ratio > 10:1)
  - Formula: `belief[i] = sqrt(cluster_size[i]) / sum(sqrt(sizes))`

- **`log_normalized`**: Further reduces dominance
  - Best for: Extremely unbalanced clusters
  - Formula: `belief[i] = log(cluster_size[i] + 1) / sum(log(sizes + 1))`

- **`balanced`**: Ensures minimum percentage per cluster
  - Best for: When you want guaranteed minimum exposure
  - Formula: Proportional + minimum threshold + renormalization

### Output Control
- `include_analysis`: Add detailed method comparison
- `generate_utilities`: Create utility functions file
- `include_examples`: Add example usage in utilities

## Advanced Usage

### Custom Configuration
```bash
python data_preprocessing/generate_natural_target.py --config my_config.yaml
```

### Different Belief Methods
```yaml
processing:
  belief_method: "sqrt_normalized"  # Reduce large cluster dominance
  min_belief_percentage: 10.0      # 10% minimum for balanced method
```

### Custom Output Paths
```yaml
output_files:
  natural_belief_target_json: "custom/path/target.json"
  natural_belief_target_pkl: "custom/path/target.pkl"
  filter_bubble_utilities: "custom/path/utilities.py"
```

### Minimal Output (No Utilities)
```yaml
output:
  include_analysis: false
  generate_utilities: false
  include_examples: false
```

## Output Format

### Main Target File (`natural_belief_target.json`)
```json
{
  "natural_belief_target": [0.699, 0.071, 0.074, 0.072, 0.084],
  "method_used": "proportional",
  "description": "Natural belief distribution based on cluster sizes",
  "cluster_sizes": {"0": 13383, "1": 1355, "2": 1413, "3": 1381, "4": 1598},
  "total_items": 19130,
  "usage": "Use as target belief in RL reward function"
}
```

### Utility Functions (`filter_bubble_utilities.py`)
Generated Python file with functions:
- `calculate_filter_bubble_score(user_belief)`: KL divergence from natural
- `calculate_diversity_score(user_belief)`: Cosine similarity to natural
- `calculate_rl_reward(before, after)`: Reward for moving toward natural
- `analyze_user_belief(user_belief)`: Comprehensive analysis

## Method Comparison

The tool automatically compares all methods and includes analysis:

```json
{
  "method_comparison": {
    "proportional": {
      "max_belief": 0.699,
      "min_belief": 0.071,
      "max_min_ratio": 9.85,
      "entropy": 1.234,
      "dominant_cluster": 0
    },
    "sqrt_normalized": {
      "max_belief": 0.456,
      "min_belief": 0.089,
      "max_min_ratio": 5.12,
      "entropy": 1.456,
      "dominant_cluster": 0
    }
  }
}
```

## Integration with Belief Processing

Use the generated target in the belief processing pipeline:

```yaml
# In belief_config.yaml
input_files:
  natural_belief_target: "processed_data/natural_belief_target.json"

processing:
  use_constrained_normalization: true
```

## Troubleshooting

- **File not found**: Check `cluster_manifest` path in config
- **Wrong cluster count**: Adjust `expected_clusters` in config
- **Method not found**: Use one of: proportional, sqrt_normalized, log_normalized, balanced
- **Permission errors**: Ensure output directories are writable

## Pipeline Integration

This tool integrates with the complete belief processing pipeline:

1. **Generate Natural Target**: `python data_preprocessing/generate_natural_target.py`
2. **Process User Beliefs**: `python data_preprocessing/process_user_beliefs.py`
3. **Use in RL Training**: Load target in recommendation environment

The natural target serves as the "ideal" distribution that guides users away from filter bubbles toward more diverse content consumption.