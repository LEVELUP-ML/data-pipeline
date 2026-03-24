#!/usr/bin/env python3
"""
Export trained sklearn model to the JSON format used by the app's
JavaScript inference engine.

The app expects model_weights.json with this schema:
{
    "type": "random_forest",
    "n_trees": N,
    "max_depth": D,
    "feature_names": [...],
    "target": "energy_score",
    "target_range": [0, 100],
    "training_samples": N,
    "in_sample_mae": X.XX,
    "in_sample_rmse": X.XX,
    "trees": [{ tree nodes... }, ...]
}

Each tree node is either:
    - Leaf:   {"leaf": value}
    - Split:  {"feature": idx, "threshold": val, "left": {...}, "right": {...}}

Usage:
    python export_model.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH, MODEL_WEIGHTS_JSON,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def tree_to_dict(tree, node_id=0):
    """
    Recursively convert a sklearn DecisionTree node to a dict.

    Args:
        tree: sklearn Tree object (from estimator.tree_)
        node_id: current node index

    Returns:
        dict ready for JSON serialization
    """
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    # Leaf node
    if left_child == right_child:  # both are TREE_LEAF (-1)
        value = float(tree.value[node_id].flatten()[0])
        return {"leaf": round(value, 2)}

    # Split node
    return {
        "feature": int(tree.feature[node_id]),
        "threshold": round(float(tree.threshold[node_id]), 2),
        "left": tree_to_dict(tree, left_child),
        "right": tree_to_dict(tree, right_child),
    }


def export_random_forest(model, feature_names, metadata):
    """Export a RandomForest model to JSON format."""
    trees_json = []
    for estimator in model.estimators_:
        trees_json.append(tree_to_dict(estimator.tree_))

    weights = {
        "type": "random_forest",
        "n_trees": len(model.estimators_),
        "max_depth": model.max_depth or max(e.tree_.max_depth for e in model.estimators_),
        "feature_names": feature_names,
        "target": "energy_score",
        "target_range": [0, 100],
        "energy_formula": "sleep_satisfaction*55 + avg_accuracy*35 + min(attempts/5,1)*10",
        "training_samples": metadata.get("train_samples", 0),
        "in_sample_mae": metadata.get("test_mae", 0),
        "in_sample_rmse": metadata.get("test_rmse", 0),
        "trees": trees_json,
    }

    return weights


def export_xgboost(model, feature_names, metadata):
    """Export an XGBoost model to JSON format using its booster."""
    try:
        booster = model.get_booster()
        dump = booster.get_dump(dump_format="json")

        trees_json = []
        for tree_str in dump:
            tree_dict = json.loads(tree_str)
            trees_json.append(_convert_xgb_node(tree_dict))

        params = model.get_params()

        # max_depth and learning_rate may be None when using XGBoost defaults
        max_depth = params.get("max_depth")
        if max_depth is None:
            try:
                config = json.loads(booster.save_config())
                max_depth = int(
                    config["learner"]["gradient_booster"]
                          ["tree_train_param"]["max_depth"]
                )
            except Exception:
                max_depth = 6

        learning_rate = params.get("learning_rate")
        if learning_rate is None:
            try:
                config = json.loads(booster.save_config())
                learning_rate = float(
                    config["learner"]["gradient_booster"]
                          ["tree_train_param"].get("eta", 0.3)
                )
            except Exception:
                learning_rate = 0.3

        # base_score may be a bracketed string e.g. '[5.47E1]' — always use 0.5 as fallback
        try:
            raw_base = params.get("base_score")
            base_score = float(str(raw_base).strip("[]")) if raw_base is not None else 0.5
        except (ValueError, TypeError):
            base_score = 0.5

        weights = {
            "type": "xgboost",
            "n_trees": len(dump),
            "max_depth": int(max_depth),
            "feature_names": feature_names,
            "target": "energy_score",
            "target_range": [0, 100],
            "training_samples": metadata.get("train_samples", 0),
            "in_sample_mae": metadata.get("test_mae", 0),
            "in_sample_rmse": metadata.get("test_rmse", 0),
            "learning_rate": float(learning_rate),
            "base_score": base_score,
            "trees": trees_json,
        }
        return weights
    except Exception as e:
        log.error("Failed to export XGBoost model: %s", e)
        return None


def _convert_xgb_node(node):
    """Convert XGBoost JSON tree node to our format."""
    if "leaf" in node:
        return {"leaf": round(node["leaf"], 2)}

    # Extract feature index from "fN" format
    split_feature = node.get("split", "f0")
    if isinstance(split_feature, str) and split_feature.startswith("f"):
        feature_idx = int(split_feature[1:])
    else:
        feature_idx = int(split_feature)

    result = {
        "feature": feature_idx,
        "threshold": round(node.get("split_condition", 0), 2),
    }

    children = node.get("children", [])
    if len(children) >= 2:
        # XGBoost: first child is "yes" (left), second is "no" (right)
        result["left"] = _convert_xgb_node(children[0])
        result["right"] = _convert_xgb_node(children[1])
    else:
        result["leaf"] = 0

    return result


def main():
    """Export the trained model to model_weights.json."""
    ensure_dirs()

    model = joblib.load(BEST_MODEL_PATH)
    log.info("Loaded model: %s", type(model).__name__)

    # Load metadata
    meta_path = BEST_MODEL_PATH.parent / "training_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    feature_names = metadata.get("feature_names", [])

    # Export based on model type
    model_type = type(model).__name__

    if "RandomForest" in model_type:
        weights = export_random_forest(model, feature_names, metadata)
    elif "XGB" in model_type or "Gradient" in model_type:
        weights = export_xgboost(model, feature_names, metadata)
    else:
        log.error("Unsupported model type for export: %s", model_type)
        sys.exit(1)

    if weights is None:
        log.error("Export failed")
        sys.exit(1)

    MODEL_WEIGHTS_JSON.write_text(json.dumps(weights))
    log.info("Exported model → %s (%d trees)", MODEL_WEIGHTS_JSON, weights["n_trees"])

    log.info("✅ Model export complete")
    return weights


if __name__ == "__main__":
    main()