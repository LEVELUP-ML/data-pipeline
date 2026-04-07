"""
scripts/model_train.py — Multi-step forecasting for flexibility and strength scores.

Trains three model architectures and selects the best by d7 RMSE:
  1. Ridge regression (linear baseline)
  2. Random Forest
  3. XGBoost with RandomizedSearchCV (20 iter, 3-fold)

All three results are logged to MLflow and written to metrics.json
under "model_comparison" so generate_plots.py can visualise them.

Outputs (data/models/{model_type}/):
  model.pkl          MultiOutputRegressor(best estimator)
  metrics.json       per-model + per-horizon metrics, SHAP, sensitivity
  bias_report.json   Fairlearn slices (sex, age_bucket) on d7 horizon
  shap_summary.png   SHAP beeswarm on d7 horizon of winning model
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost missing. Add to requirements.txt and rebuild.")

try:
    import shap
except ImportError:
    sys.exit("shap missing. Add to requirements.txt and rebuild.")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False
    print("WARNING: mlflow not installed — tracking disabled.")

try:
    from fairlearn.metrics import MetricFrame
    FAIRLEARN_OK = True
except ImportError:
    FAIRLEARN_OK = False
    print("WARNING: fairlearn not installed — bias analysis skipped.")


AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")


def get_model_config(model_type: str):
    """Get configuration for different model types."""
    configs = {
        "flexibility": {
            "features_path": Path(f"{AIRFLOW_HOME}/data/processed/flexibility_features.parquet"),
            "models_dir":    Path(f"{AIRFLOW_HOME}/data/models/flexibility"),
            "experiment_name": "flexibility_score_forecasting",
            "rmse_threshold": float(os.getenv("FLEXIBILITY_RMSE_THRESHOLD",
                                       os.getenv("MODEL_RMSE_THRESHOLD", "10.0"))),
            "non_feature_cols": {"user_id", "ref_date", "ref_score", "sex_raw", "age_raw"},
        },
        "strength": {
            "features_path": Path(f"{AIRFLOW_HOME}/data/processed/strength_features.parquet"),
            "models_dir":    Path(f"{AIRFLOW_HOME}/data/models/strength"),
            "experiment_name": "strength_score_forecasting",
            "rmse_threshold": float(os.getenv("STRENGTH_RMSE_THRESHOLD",
                                       os.getenv("MODEL_RMSE_THRESHOLD", "2000.0"))),

            "non_feature_cols": {"user_id", "ref_date", "ref_score", "sex_raw", "age_raw"},
        },
    }

    if model_type not in configs:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(configs.keys())}")

    return configs[model_type]


# Global config (set by train function)
FEATURES_PATH    = None
MODELS_DIR       = None
MODEL_PATH       = None
METRICS_PATH     = None
BIAS_PATH        = None
SHAP_PATH        = None
EXPERIMENT_NAME  = None
NON_FEATURE_COLS = {"user_id", "ref_date", "ref_score", "sex_raw", "age_raw"}

HORIZONS     = [1, 3, 7, 14]
TARGET_COLS  = [f"target_d{h}" for h in HORIZONS]
TEST_FRAC    = 0.20
GATE_HORIZON = "d7"


def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            f"Run {EXPERIMENT_NAME.split('_')[0]}_features DAG first."
        )
    df = pd.read_parquet(str(FEATURES_PATH))
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def time_split(df):
    date_col = "ref_date" if "ref_date" in df.columns else "reference_date"
    dates   = sorted(df[date_col].unique())
    cut_idx = int(len(dates) * (1 - TEST_FRAC))
    cutoff  = dates[cut_idx]
    train   = df[df[date_col] <  cutoff].copy()
    test    = df[df[date_col] >= cutoff].copy()
    print(f"Train: {len(train):,} | Test: {len(test):,} | cutoff={cutoff.date()}")
    return train, test


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS and not c.startswith("target_")]


def prepare(df, feature_cols):
    valid = df.dropna(subset=TARGET_COLS)
    X     = valid[feature_cols].fillna(valid[feature_cols].median(numeric_only=True))
    Y     = valid[TARGET_COLS]
    return X, Y


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_per_horizon(model, X, Y, label):
    Y_pred = pd.DataFrame(model.predict(X), columns=TARGET_COLS, index=Y.index)
    result = {}
    for col, h in zip(TARGET_COLS, HORIZONS):
        yt, yp = Y[col], Y_pred[col]
        r = {
            "rmse": round(rmse(yt, yp), 4),
            "mae":  round(float(mean_absolute_error(yt, yp)), 4),
            "r2":   round(float(r2_score(yt, yp)), 4),
        }
        result[f"d{h}"] = r
        print(f"  [{label}] +{h:2d}d  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R2={r['r2']:.4f}")
    return result


#  Three architectures 

def build_ridge(X_train, Y_train):
    print("\n--- Ridge regression (baseline) ---")
    model = MultiOutputRegressor(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    )
    model.fit(X_train, Y_train)
    return model


def build_random_forest(X_train, Y_train):
    print("\n--- Random Forest ---")
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, Y_train)
    return model


def build_xgboost(X_train, Y_train):
    print("\n--- XGBoost (RandomizedSearchCV) ---")
    param_dist = {
        "estimator__n_estimators":     [100, 200, 300, 400],
        "estimator__max_depth":        [3, 4, 5, 6],
        "estimator__learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
        "estimator__subsample":        [0.7, 0.8, 0.9, 1.0],
        "estimator__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "estimator__min_child_weight": [1, 3, 5],
        "estimator__reg_alpha":        [0, 0.05, 0.1, 0.5],
        "estimator__reg_lambda":       [1.0, 1.5, 2.0],
    }
    base = MultiOutputRegressor(
        xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42,
            n_jobs=-1, verbosity=0, tree_method="hist",
        ),
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=20, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, verbose=1, refit=True,
    )
    search.fit(X_train, Y_train)
    print(f"Best params: { {k.replace('estimator__',''):v for k,v in search.best_params_.items()} }")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.cv_results_


def hyperparam_sensitivity(cv_results):
    try:
        df   = pd.DataFrame(cv_results)
        sens = {}
        for col in [c for c in df.columns if c.startswith("param_")]:
            name    = col.replace("param_estimator__", "").replace("param_", "")
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.isna().all():
                continue
            corr = numeric.corr(df["mean_test_score"])
            sens[name] = round(float(corr), 4) if not np.isnan(corr) else 0.0
        return {
            "description": "Pearson correlation: hyperparameter value vs CV RMSE",
            "correlations": sens,
        }
    except Exception as e:
        print(f"WARNING: sensitivity failed: {e}")
        return {"available": False}


#  Bias analysis 

def run_bias(model, test_df, X_test, Y_test):
    if not FAIRLEARN_OK:
        return {"available": False}

    Y_pred    = pd.DataFrame(model.predict(X_test), columns=TARGET_COLS, index=Y_test.index)
    gate_col  = f"target_{GATE_HORIZON}"
    y_true_d7 = Y_test[gate_col]
    y_pred_d7 = Y_pred[gate_col]
    overall_r = rmse(y_true_d7, y_pred_d7)

    report = {
        "available": True,
        "slices":    {},
        "flagged":   [],
        "mitigation_notes": (
            "If bias is detected: (1) collect more data from underrepresented groups, "
            "(2) apply sample_weight inversely proportional to group frequency, "
            "(3) use stratified CV splits by demographic."
        ),
    }

    for slice_col, raw_col in [("sex", "sex_raw"), ("age_bucket", "age_raw")]:
        if raw_col not in test_df.columns:
            continue
        sensitive = test_df.loc[Y_test.index, raw_col]
        if slice_col == "age_bucket":
            def _bkt(a):
                try:
                    a = int(a)
                    return "<20" if a < 20 else "20-29" if a < 30 else "30-39" if a < 40 else "40+"
                except Exception:
                    return "unknown"
            sensitive = sensitive.apply(_bkt)
        sensitive = sensitive.fillna("unknown")

        mf       = MetricFrame(
            metrics={"rmse": lambda yt, yp: rmse(yt, yp)},
            y_true=y_true_d7, y_pred=y_pred_d7,
            sensitive_features=sensitive,
        )
        by_group = {str(k): round(float(v["rmse"]), 4) for k, v in mf.by_group.iterrows()}
        report["slices"][slice_col] = {"overall_rmse": round(overall_r, 4), "by_group": by_group}

        for g, gr in by_group.items():
            if overall_r > 0 and gr / overall_r > 1.5:
                flag = f"{slice_col}={g}: RMSE={gr:.2f} ({gr/overall_r:.1f}x overall)"
                report["flagged"].append(flag)
                print(f"  BIAS FLAG: {flag}")
        print(f"  [bias/{slice_col}] {by_group}")

    return report


#  SHAP 

def run_shap(model, X_train, horizon_idx=2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        estimator = model.estimators_[horizon_idx]
        sample    = X_train.sample(min(300, len(X_train)), random_state=42)

        try:
            explainer   = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(sample)
        except Exception:
            explainer   = shap.KernelExplainer(estimator.predict, sample.iloc[:50])
            shap_values = explainer.shap_values(sample, nsamples=50)

        mean_abs   = np.abs(shap_values).mean(axis=0)
        importance = {col: round(float(v), 5) for col, v in zip(X_train.columns, mean_abs)}
        top10      = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, sample, show=False)
        plt.title(f"SHAP — horizon +{HORIZONS[horizon_idx]}d")
        plt.tight_layout()
        plt.savefig(str(SHAP_PATH), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"SHAP plot -> {SHAP_PATH}")
        return top10
    except Exception as e:
        print(f"WARNING: SHAP failed: {e}")
        return {}


#  Main 

def train(run_id=None, model_type="flexibility"):
    global FEATURES_PATH, MODELS_DIR, MODEL_PATH, METRICS_PATH, BIAS_PATH, SHAP_PATH
    global EXPERIMENT_NAME, NON_FEATURE_COLS

    config           = get_model_config(model_type)
    FEATURES_PATH    = config["features_path"]
    MODELS_DIR       = config["models_dir"]
    MODEL_PATH       = MODELS_DIR / "model.pkl"
    METRICS_PATH     = MODELS_DIR / "metrics.json"
    BIAS_PATH        = MODELS_DIR / "bias_report.json"
    SHAP_PATH        = MODELS_DIR / "shap_summary.png"
    EXPERIMENT_NAME  = config["experiment_name"]
    NON_FEATURE_COLS = config["non_feature_cols"] | set(TARGET_COLS)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    threshold = float(os.getenv("MODEL_RMSE_THRESHOLD", str(config["rmse_threshold"])))

    df               = load_data()
    train_df, test_df = time_split(df)
    date_col         = "ref_date" if "ref_date" in df.columns else "reference_date"
    feature_cols     = get_feature_cols(df)
    X_train, Y_train = prepare(train_df, feature_cols)
    X_test,  Y_test  = prepare(test_df,  feature_cols)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train or test set empty after dropping null targets.")

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{AIRFLOW_HOME}/data/models/mlflow.db",
    )
    active_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            active_run = mlflow.start_run(
                run_name=run_id or f"train_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
            )
        except Exception as e:
            print(f"WARNING: MLflow unavailable ({e}) — training will continue without tracking.")
            active_run = None

    try:
        #  Train all three architectures 
        models = {}

        ridge_model                    = build_ridge(X_train, Y_train)
        models["Ridge"]                = ridge_model

        rf_model                       = build_random_forest(X_train, Y_train)
        models["Random Forest"]        = rf_model

        xgb_model, best_params, cv_res = build_xgboost(X_train, Y_train)
        models["XGBoost"]              = xgb_model

        #  Evaluate each 
        comparison = {}
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            comparison[name] = eval_per_horizon(model, X_test, Y_test, name)

        #  Select winner by d7 RMSE 
        winner_name  = min(comparison, key=lambda n: comparison[n]["d7"]["rmse"])
        winner_model = models[winner_name]
        print(f"\nWinner: {winner_name} (d7 RMSE={comparison[winner_name]['d7']['rmse']:.4f})")

        train_metrics = eval_per_horizon(winner_model, X_train, Y_train, f"{winner_name}/train")
        test_metrics  = eval_per_horizon(winner_model, X_test,  Y_test,  f"{winner_name}/test")
        gate_rmse     = test_metrics[GATE_HORIZON]["rmse"]

        print("\nRunning SHAP...")
        shap_top10 = run_shap(winner_model, X_train, horizon_idx=2)

        sensitivity = hyperparam_sensitivity(cv_res)

        print("\nRunning bias analysis...")
        bias_report = run_bias(winner_model, test_df, X_test, Y_test)

        #  Log to MLflow 
        if MLFLOW_OK and active_run:
            try:
                mlflow.log_param("winner_model", winner_name)
                mlflow.log_params({k.replace("estimator__", ""): v for k, v in best_params.items()})
                for h_key, h_m in test_metrics.items():
                    for metric_name, val in h_m.items():
                        mlflow.log_metric(f"test_{h_key}_{metric_name}", val)
                for arch_name, arch_metrics in comparison.items():
                    safe = arch_name.lower().replace(" ", "_")
                    mlflow.log_metric(f"{safe}_d7_rmse", arch_metrics["d7"]["rmse"])
                mlflow.log_metric(f"gate_{GATE_HORIZON}_rmse", gate_rmse)
                mlflow.log_dict(bias_report, "bias_report.json")
                mlflow.log_dict(sensitivity, "hyperparam_sensitivity.json")
                if SHAP_PATH.exists():
                    mlflow.log_artifact(str(SHAP_PATH))
                mlflow.sklearn.log_model(
                    winner_model,
                    artifact_path=f"{model_type}_forecaster",
                    registered_model_name=f"{model_type}_score_forecaster",
                )
            except Exception as e:
                print(f"WARNING: MLflow logging failed ({e}) — artifacts saved locally regardless.")

        #  Save artifacts 
        with MODEL_PATH.open("wb") as f:
            pickle.dump(winner_model, f)

        metrics_out = {
            "trained_at":             datetime.utcnow().isoformat() + "Z",
            "run_id":                 run_id,
            "winner_model":           winner_name,
            "features":               feature_cols,
            "train_rows":             len(X_train),
            "test_rows":              len(X_test),
            "users_train":            int(train_df["user_id"].nunique()),
            "users_test":             int(test_df["user_id"].nunique()),
            "train_date_range":       [
                str(train_df[date_col].min().date()),
                str(train_df[date_col].max().date()),
            ],
            "test_date_range":        [
                str(test_df[date_col].min().date()),
                str(test_df[date_col].max().date()),
            ],
            "model_comparison":       comparison,
            "best_params":            {
                k.replace("estimator__", ""): v for k, v in best_params.items()
            },
            "best_cv_rmse":           round(-float(min(cv_res["mean_test_score"])), 4)
                                      if cv_res else None,
            "train_metrics":          train_metrics,
            "test_metrics":           test_metrics,
            "gate_horizon":           GATE_HORIZON,
            "gate_rmse":              gate_rmse,
            "rmse_threshold":         threshold,
            "passed_gate":            gate_rmse <= threshold,
            "shap_top10":             shap_top10,
            "hyperparam_sensitivity": sensitivity,
            "bias_flagged":           bias_report.get("flagged", []),
        }

        METRICS_PATH.write_text(json.dumps(metrics_out, indent=2, default=str))
        BIAS_PATH.write_text(json.dumps(bias_report, indent=2, default=str))
        print(f"\nModel   -> {MODEL_PATH}")
        print(f"Metrics -> {METRICS_PATH}")
        print(f"Bias    -> {BIAS_PATH}")

        # Auto-generate plots right after training
        try:
            import importlib.util as _ilu
            plots_path = Path(__file__).parent / "generate_plots.py"
            if plots_path.exists():
                spec = _ilu.spec_from_file_location("generate_plots", str(plots_path))
                mod  = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.generate_all(
                    metrics_path=str(METRICS_PATH),
                    bias_path=str(BIAS_PATH),
                    plots_dir=str(MODELS_DIR / "plots"),
                    features_path=str(FEATURES_PATH),
                )
        except Exception as e:
            print(f"WARNING: plot generation failed (non-fatal): {e}")

        if gate_rmse > threshold:
            raise ValueError(
                f"Model gate FAILED: {GATE_HORIZON} RMSE {gate_rmse:.4f} > {threshold}. "
                "Inspect model_comparison in metrics.json and tune accordingly."
            )

        print(f"\nModel gate PASSED ({GATE_HORIZON} RMSE {gate_rmse:.4f} <= {threshold})")
        return metrics_out

    finally:
        if MLFLOW_OK and active_run:
            mlflow.end_run()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-id",         default=None)
    p.add_argument("--model-type",     default="flexibility", choices=["flexibility", "strength"])
    p.add_argument("--rmse-threshold", type=float, default=None)
    args = p.parse_args()
    if args.rmse_threshold:
        os.environ["MODEL_RMSE_THRESHOLD"] = str(args.rmse_threshold)
    result = train(run_id=args.run_id, model_type=args.model_type)
    print(f"\n=== Winner model per-horizon test metrics for {args.model_type} ===")
    for h_key, h_m in result["test_metrics"].items():
        print(f"  +{h_key}  RMSE={h_m['rmse']:.4f}  MAE={h_m['mae']:.4f}  R2={h_m['r2']:.4f}")