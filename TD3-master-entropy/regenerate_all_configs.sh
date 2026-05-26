#!/bin/bash
# 重新生成所有配置文件

DATASETS="news books movies"
BACKBONES="ENMF LightGCN NCL NGCF SGL"

for dataset in $DATASETS; do
  for backbone in $BACKBONES; do
    echo "生成 config_${dataset}_${backbone}.yaml"
    python generate_config.py \
      --template config_template.yaml \
      --dataset "$dataset" \
      --rs_model "$backbone" \
      --base_path . \
      --out_root results_table3_retrain \
      --sigmoid_scale 3500 \
      --out "_configs_tmp/config_${dataset}_${backbone}.yaml"
  done
done

echo "所有配置文件已重新生成"
