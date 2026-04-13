"""
phase2/utils/results_io.py
Save and load Phase 2 results as JSON.
"""

import json
import os


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert any non-serialisable types
    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        if hasattr(obj, 'item'):    # numpy / torch scalar
            return obj.item()
        return obj
    with open(path, 'w') as f:
        json.dump(_convert(results), f, indent=2)
    print(f"Results saved -> {path} ✓")


def load_results(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
