#!/usr/bin/env python3
"""
Convert node_labels.json {item_id: cluster_id} format
to cluster_matrix_manifest format {cluster_to_items: {cluster_id: [item_ids]}}
"""

import json
from collections import defaultdict


def convert(node_labels_path: str, output_path: str):
    with open(node_labels_path) as f:
        node_labels = json.load(f)

    cluster_to_items = defaultdict(list)
    for item_id, cluster_id in node_labels.items():
        cluster_to_items[str(cluster_id)].append(item_id)

    num_clusters = len(cluster_to_items)
    cluster_sizes = {k: len(v) for k, v in cluster_to_items.items()}

    manifest = {
        "num_clusters": num_clusters,
        "num_nodes": len(node_labels),
        "cluster_sizes": cluster_sizes,
        "cluster_to_items": dict(cluster_to_items),
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Converted {len(node_labels)} nodes into {num_clusters} clusters")
    for k, v in cluster_sizes.items():
        print(f"  Cluster {k}: {v} items")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    convert(
        node_labels_path="embeddings/books/K_means/node_labels.json",
        output_path="embeddings/books/K_means/cluster_matrix_manifest_K5.json",
    )
