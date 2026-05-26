#!/usr/bin/env python3
"""
Render a per-cell training config from config_template.yaml.

Usage:
    python generate_config.py \
        --template config_template.yaml \
        --dataset news --rs_model LightGCN \
        --base_path . \
        --out_root results_table3_retrain \
        --sigmoid_scale 3500 \
        --out config_TMP.yaml
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--dataset", required=True, choices=["news", "books", "movies"])
    p.add_argument("--rs_model", required=True, choices=["ENMF", "LightGCN", "NCL", "NGCF", "SGL"])
    p.add_argument("--base_path", default=".",
                   help="Absolute or relative path to TD3-master-entropy/ root. "
                        "Resolved to absolute at render time.")
    p.add_argument("--out_root", default="results_table3_retrain",
                   help="Output dir for results/models (will hold per-cell subdirs).")
    p.add_argument("--sigmoid_scale", type=int, default=3500,
                   help="Path A Table 3 canonical = 3500 (default).")
    p.add_argument("--out", required=True, help="Destination config file path.")
    args = p.parse_args()

    # Use forward slashes everywhere so the rendered YAML stays valid even when
    # Python resolves paths to Windows-style backslashes (which YAML treats as
    # escape characters inside double-quoted strings).
    base_path = str(Path(args.base_path).resolve()).replace("\\", "/")
    out_root = args.out_root
    # Keep out_root as relative path for logging directories
    out_root = out_root.replace("\\", "/")

    with open(args.template, "r", encoding="utf-8") as f:
        txt = f.read()

    rendered = (txt.replace("{DATASET}", args.dataset)
                   .replace("{BACKBONE}", args.rs_model)
                   .replace("{BASE_PATH}", base_path)
                   .replace("{OUT_ROOT}", out_root)
                   .replace("{SIGMOID_SCALE}", str(args.sigmoid_scale)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Generated: {out_path}")
    print(f"  dataset={args.dataset}, rs_model={args.rs_model}, "
          f"sigmoid_scale={args.sigmoid_scale}")
    print(f"  base_path={base_path}")
    print(f"  out_root={out_root}")


if __name__ == "__main__":
    main()
