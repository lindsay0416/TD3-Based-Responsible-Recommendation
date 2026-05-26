#!/bin/bash
# ----------------------------------------------------------------------
# Table 3 retrain + eval pipeline, Path A K=5 canonical.
#
# For each of 15 (dataset, backbone) cells:
#   1. Generate a temp config from config_template.yaml
#   2. Retrain TD3 (50 episodes) and save final actor
#   3. Run rerun eval at k=10 in both --agent_mode ferl and --agent_mode base_only
#   4. Write metrics JSON
#
# Cells that already have a Path A trained actor under k_ablation_rl/ are
# skipped at the training step unless --force_retrain is set.
#
# Usage (run from TD3-master-entropy/):
#   bash run_table3_pipeline.sh                     # full 15-cell sweep
#   bash run_table3_pipeline.sh --eval_only         # skip retrain, eval whatever is in models/
#   bash run_table3_pipeline.sh --force_retrain     # retrain even cells we have actors for
#
# Knobs:
#   OUT_ROOT      where retrain checkpoints + results land (default: results_table3_retrain)
#   EVAL_OUT_ROOT where eval JSONs land (default: RL_evaluation_metrics_k10)
#   SIGMOID       sigmoid_scale (default: 3500, paper Table 3 canonical)
#   NUM_ROUNDS    eval rounds per user (default: 50)
#   DATASETS      space-separated list (default: "news books movies")
#   BACKBONES     space-separated list (default: "ENMF LightGCN NCL NGCF SGL")
#
# Runtime envelope on RTX 4090:
#   - Per-cell retrain: ~10-15 min (2.5M transitions)
#   - Per-cell eval (ferl + base_only): ~10-25 min (sequential, num_rounds=50)
#   - Full 15-cell sweep: ~6-10 hours. Plan an overnight run.
# ----------------------------------------------------------------------

set -e

# --- Args ---
EVAL_ONLY=0
FORCE_RETRAIN=0
for arg in "$@"; do
  case "$arg" in
    --eval_only)     EVAL_ONLY=1 ;;
    --force_retrain) FORCE_RETRAIN=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# --- Knobs (env-overridable) ---
OUT_ROOT="${OUT_ROOT:-results_table3_retrain}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-RL_evaluation_metrics_k10}"
SIGMOID="${SIGMOID:-3500}"
NUM_ROUNDS="${NUM_ROUNDS:-50}"
DATASETS="${DATASETS:-news books movies}"
BACKBONES="${BACKBONES:-ENMF LightGCN NCL NGCF SGL}"

TEMPLATE="config_template.yaml"
GEN="generate_config.py"
EVAL_PY="test_set/evaluate_rl_model_rerun.py"
TRAIN_PY="run_recommendation_rl.py"

if [ ! -f "$TEMPLATE" ] || [ ! -f "$GEN" ]; then
  echo "ERROR: $TEMPLATE or $GEN not in current dir."
  echo "  Run this script from TD3-master-entropy/ with the bundle files placed there."
  exit 1
fi

mkdir -p _configs_tmp
mkdir -p logs

# --- Per-cell loop ---
for dataset in $DATASETS; do
  for backbone in $BACKBONES; do
    CELL="${dataset}/${backbone}"
    CONFIG="_configs_tmp/config_${dataset}_${backbone}.yaml"
    CELL_OUT="${OUT_ROOT}/${dataset}/${backbone}"
    MODELS_DIR="${CELL_OUT}/models"
    EVAL_DIR="${EVAL_OUT_ROOT}/${dataset}/K_means/${backbone}"

    # Create output directories
    mkdir -p "$CELL_OUT"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$EVAL_DIR"

    echo ""
    echo "=========================================================="
    echo "  CELL: ${CELL}"
    echo "=========================================================="

    # 1. Generate config
    python "$GEN" \
      --template "$TEMPLATE" \
      --dataset "$dataset" \
      --rs_model "$backbone" \
      --base_path . \
      --out_root "$OUT_ROOT" \
      --sigmoid_scale "$SIGMOID" \
      --out "$CONFIG"

    # 2. Retrain (unless --eval_only)
    if [ "$EVAL_ONLY" -eq 0 ]; then
      # Skip if we already have a final actor and not forcing
      EXISTING=$(ls "$MODELS_DIR"/td3_recommendation_*_final_actor 2>/dev/null | head -1 || true)
      if [ -n "$EXISTING" ] && [ "$FORCE_RETRAIN" -eq 0 ]; then
        echo "  [SKIP retrain] existing actor: $EXISTING"
      else
        echo "  [RETRAIN] $(date '+%H:%M:%S')"
        # `yes` pipes y into the input() prompt of run_recommendation_rl.py
        yes | python "$TRAIN_PY" --config "$CONFIG" 2>&1 | tee "logs/train_${dataset}_${backbone}.log" \
          || { echo "  [FAIL retrain] ${CELL}; continuing"; continue; }
      fi
    fi

    # 3. Eval ferl + base_only
    mkdir -p "$EVAL_DIR"
    # Use the cell-specific config so eval reads the right experiment_params and paths
    # by symlinking it as config.yaml in cwd; restore at the end.
    if [ -f config.yaml ] && [ ! -L config.yaml ]; then
      cp config.yaml config.yaml.user_backup
    fi
    cp -f "$CONFIG" config.yaml

    MODEL_ARG=""
    LATEST=$(ls -t "$MODELS_DIR"/td3_recommendation_*_final_actor 2>/dev/null | head -1 || true)
    if [ -n "$LATEST" ]; then
      MODEL_ARG="--model_path ${LATEST%_actor}"
    fi

    echo "  [EVAL ferl] $(date '+%H:%M:%S')"
    python "$EVAL_PY" \
      --agent_mode ferl \
      --dataset "$dataset" --rs_model "$backbone" --topk 5 \
      --k_eval 10 --num_rounds "$NUM_ROUNDS" \
      $MODEL_ARG \
      --output_path "${EVAL_DIR}/ferl_results.json" \
      2>&1 | tee "logs/eval_ferl_${dataset}_${backbone}.log" \
      || echo "  [WARN ferl] ${CELL} failed; continuing"

    echo "  [EVAL base_only] $(date '+%H:%M:%S')"
    python "$EVAL_PY" \
      --agent_mode base_only \
      --dataset "$dataset" --rs_model "$backbone" --topk 5 \
      --k_eval 10 --num_rounds "$NUM_ROUNDS" \
      --output_path "${EVAL_DIR}/base_only_results.json" \
      2>&1 | tee "logs/eval_base_${dataset}_${backbone}.log" \
      || echo "  [WARN base_only] ${CELL} failed; continuing"

    # Restore user config.yaml if we shadowed it
    if [ -f config.yaml.user_backup ]; then
      mv config.yaml.user_backup config.yaml
    fi
  done
done

echo ""
echo "=========================================================="
echo "  Pipeline complete."
echo "  Retrain checkpoints: ${OUT_ROOT}/{dataset}/{backbone}/models/"
echo "  Eval JSONs:          ${EVAL_OUT_ROOT}/{dataset}/K_means/{backbone}/"
echo "  Training logs:       logs/train_*.log"
echo ""
echo "  When done, zip ${EVAL_OUT_ROOT}/ + logs/ + ${OUT_ROOT}/<each cell>/metrics.json"
echo "  and send back."
echo "=========================================================="
