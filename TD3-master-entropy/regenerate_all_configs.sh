#!/bin/bash

# Regenerate all config files with updated template
# 3 datasets: news, books, movies
# 5 models: ENMF, LightGCN, NCL, NGCF, SGL

BASE_PATH="/Users/lindsay/Desktop/Git/TD3-Based-Responsible-Recommendation/TD3-master-entropy"
OUT_ROOT="results_table3_retrain"
SIGMOID_SCALE=3500
OUT_DIR="_configs_tmp"

mkdir -p "$OUT_DIR"

for DATASET in news books movies; do
    for MODEL in ENMF LightGCN NCL NGCF SGL; do
        echo "Generating config for $DATASET $MODEL..."
        python generate_config.py \
            --template config_template.yaml \
            --dataset "$DATASET" \
            --rs_model "$MODEL" \
            --base_path "$BASE_PATH" \
            --out_root "$OUT_ROOT" \
            --sigmoid_scale "$SIGMOID_SCALE" \
            --out "$OUT_DIR/config_${DATASET}_${MODEL}.yaml"
    done
done

echo "All configs regenerated in $OUT_DIR/"