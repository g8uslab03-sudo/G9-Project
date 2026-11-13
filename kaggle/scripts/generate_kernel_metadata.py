"""Kaggle kernel metadata generator."""

import json
import os

METADATA = {
    "id": "g9/kaggle-t4-worker",
    "title": "G9 Autonomous System Kaggle T4x2 Worker",
    "code_file": "worker.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}

if __name__ == "__main__":
    print(json.dumps(METADATA, indent=2))
    with open("kernel-metadata.json", "w") as f:
        json.dump(METADATA, f, indent=2)
