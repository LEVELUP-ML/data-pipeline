#!/usr/bin/env python3
"""
ML Pipeline Orchestrator.

Runs the complete ML pipeline in sequence:
  1. Hyperparameter tuning
  2. Model training with best params
  3. Model validation (with gate)
  4. Bias detection
  5. Sensitivity analysis (SHAP)
  6. Model export to JSON
  7. Registry push

Usage:
    python run_pipeline.py [--skip-tuning] [--skip-push]
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_step(name, func, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    log.info("=" * 60)
    log.info("▶ STEP: %s", name)
    log.info("=" * 60)

    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        log.info("✅ %s completed (%.1fs)", name, elapsed)
        return result
    except SystemExit as e:
        elapsed = time.time() - start
        if e.code != 0:
            log.error("❌ %s FAILED (%.1fs)", name, elapsed)
            raise
        return None
    except Exception as e:
        elapsed = time.time() - start
        log.error("❌ %s FAILED (%.1fs): %s", name, elapsed, e)
        raise


def main():
    """Execute the full ML pipeline."""
    skip_tuning = "--skip-tuning" in sys.argv
    skip_push = "--skip-push" in sys.argv

    log.info("🚀 Level Up ML Pipeline Starting")
    log.info("   Skip tuning: %s", skip_tuning)
    log.info("   Skip push: %s", skip_push)

    pipeline_start = time.time()

    # Step 1: Hyperparameter Tuning
    params_file = None
    if not skip_tuning:
        from hyperparameter_tuning import main as tune_main
        run_step("Hyperparameter Tuning", tune_main)
        from config import MODELS_DIR
        params_file = str(MODELS_DIR / "best_params.json")
    else:
        log.info("⏭ Skipping hyperparameter tuning")

    # Step 2: Model Training
    from train_model import main as train_main
    run_step("Model Training", train_main, params_file)

    # Step 3: Validation
    from validate_model import main as validate_main
    run_step("Model Validation", validate_main)

    # Step 4: Bias Detection
    from model_bias_detection import main as bias_main
    run_step("Bias Detection", bias_main)

    # Step 5: Sensitivity Analysis
    from sensitivity_analysis import main as shap_main
    run_step("Sensitivity Analysis (SHAP)", shap_main)

    # Step 6: Model Export
    from export_model import main as export_main
    run_step("Model Export", export_main)

    # Step 7: Registry Push
    if not skip_push:
        from registry_push import main as push_main
        run_step("Registry Push", push_main)
    else:
        log.info("⏭ Skipping registry push")

    total_time = time.time() - pipeline_start
    log.info("=" * 60)
    log.info("🎉 ML Pipeline Complete! Total time: %.1fs", total_time)
    log.info("=" * 60)


if __name__ == "__main__":
    main()