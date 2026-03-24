import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: file not found: {p}")
        return None
    return pd.read_csv(p)


def mifflin_st_jeor_bmr(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    """
    BMR estimate using Mifflin-St Jeor.
    sex: 'male' or 'female'
    """
    sex = sex.lower()
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    if sex == "male":
        return base + 5
    return base - 161


def categorize_meal_gap(hours: float | None) -> str:
    if hours is None or pd.isna(hours):
        return "unknown"
    if hours < 1:
        return "too_close"
    if hours <= 3:
        return "optimal"
    if hours <= 4:
        return "acceptable"
    return "too_long"


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    if len(a.dropna()) < 2 or len(b.dropna()) < 2:
        return None
    try:
        val = a.corr(b)
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze nutrition, sleep, and workout behavior.")
    parser.add_argument("--food_csv", required=True, help="CSV with datetime, calories, protein_g, carbs_g, fat_g")
    parser.add_argument("--sleep_csv", default=None, help="Optional CSV with date,sleep_hours or sleep_start,sleep_end")
    parser.add_argument("--workout_csv", default=None, help="Optional CSV with workout_datetime,duration_min,calories_burned,stamina_score")
    parser.add_argument("--weight_kg", type=float, required=True)
    parser.add_argument("--height_cm", type=float, required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--protein_goal_g_per_kg", type=float, default=1.6)
    parser.add_argument("--pre_workout_window_hr", type=float, default=3.0)
    parser.add_argument("--out_dir", default="Data-Pipeline/data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load food logs
    # -----------------------------
    food = pd.read_csv(args.food_csv)
    food["datetime"] = pd.to_datetime(food["datetime"])
    food["date"] = food["datetime"].dt.date

    required_food_cols = {"datetime", "calories", "protein_g", "carbs_g", "fat_g"}
    missing_food = required_food_cols - set(food.columns)
    if missing_food:
        raise ValueError(f"Food CSV missing columns: {missing_food}")

    # Daily food totals
    daily_food = (
        food.groupby("date", as_index=False)[["calories", "protein_g", "carbs_g", "fat_g"]]
        .sum()
        .sort_values("date")
    )

    # -----------------------------
    # Load sleep logs
    # -----------------------------
    sleep = load_csv(args.sleep_csv)
    daily_sleep = None
    if sleep is not None:
        if "date" in sleep.columns and "sleep_hours" in sleep.columns:
            sleep["date"] = pd.to_datetime(sleep["date"]).dt.date
            daily_sleep = sleep[["date", "sleep_hours"]].copy()
        elif {"sleep_start", "sleep_end"}.issubset(sleep.columns):
            sleep["sleep_start"] = pd.to_datetime(sleep["sleep_start"])
            sleep["sleep_end"] = pd.to_datetime(sleep["sleep_end"])
            sleep["sleep_hours"] = (sleep["sleep_end"] - sleep["sleep_start"]).dt.total_seconds() / 3600.0
            sleep["date"] = sleep["sleep_end"].dt.date
            daily_sleep = sleep[["date", "sleep_hours"]].copy()
        else:
            print("Warning: sleep CSV format not recognized. Expected date+sleep_hours or sleep_start+sleep_end.")

    # -----------------------------
    # Load workouts
    # -----------------------------
    workouts = load_csv(args.workout_csv)
    daily_workouts = None
    workout_analysis = None

    if workouts is not None:
        required_workout_base = {"workout_datetime"}
        if not required_workout_base.issubset(workouts.columns):
            raise ValueError("Workout CSV must include workout_datetime")

        workouts["workout_datetime"] = pd.to_datetime(workouts["workout_datetime"])
        workouts["date"] = workouts["workout_datetime"].dt.date

        if "duration_min" not in workouts.columns:
            workouts["duration_min"] = np.nan
        if "calories_burned" not in workouts.columns:
            workouts["calories_burned"] = 0.0
        if "stamina_score" not in workouts.columns:
            workouts["stamina_score"] = np.nan

        daily_workouts = (
            workouts.groupby("date", as_index=False)[["duration_min", "calories_burned", "stamina_score"]]
            .agg({
                "duration_min": "sum",
                "calories_burned": "sum",
                "stamina_score": "mean"
            })
        )

        # -----------------------------
        # Workout fueling analytics
        # -----------------------------
        workout_rows = []
        for _, w in workouts.iterrows():
            w_time = w["workout_datetime"]
            same_day_meals = food[food["datetime"] <= w_time].copy()

            if same_day_meals.empty:
                last_meal_time = None
                gap_hours = None
                pre_window_carbs = 0.0
                pre_window_calories = 0.0
                pre_window_protein = 0.0
            else:
                last_meal_time = same_day_meals["datetime"].max()
                gap_hours = (w_time - last_meal_time).total_seconds() / 3600.0

                window_start = w_time - pd.Timedelta(hours=args.pre_workout_window_hr)
                pre_window_meals = food[(food["datetime"] >= window_start) & (food["datetime"] <= w_time)].copy()

                pre_window_carbs = float(pre_window_meals["carbs_g"].sum()) if not pre_window_meals.empty else 0.0
                pre_window_calories = float(pre_window_meals["calories"].sum()) if not pre_window_meals.empty else 0.0
                pre_window_protein = float(pre_window_meals["protein_g"].sum()) if not pre_window_meals.empty else 0.0

            workout_rows.append({
                "workout_datetime": w_time,
                "date": w["date"],
                "duration_min": w["duration_min"],
                "calories_burned": w["calories_burned"],
                "stamina_score": w["stamina_score"],
                "last_meal_time": last_meal_time,
                "meal_gap_hours": gap_hours,
                "meal_gap_category": categorize_meal_gap(gap_hours),
                "pre_window_carbs_g": pre_window_carbs,
                "pre_window_calories": pre_window_calories,
                "pre_window_protein_g": pre_window_protein
            })

        workout_analysis = pd.DataFrame(workout_rows)

    # -----------------------------
    # Merge daily analytics
    # -----------------------------
    daily = daily_food.copy()

    if daily_sleep is not None:
        daily = daily.merge(daily_sleep, on="date", how="left")
    else:
        daily["sleep_hours"] = np.nan

    if daily_workouts is not None:
        daily = daily.merge(daily_workouts, on="date", how="left")
    else:
        daily["duration_min"] = np.nan
        daily["calories_burned"] = 0.0
        daily["stamina_score"] = np.nan

    daily["calories_burned"] = daily["calories_burned"].fillna(0.0)

    # BMR / energy balance
    bmr = mifflin_st_jeor_bmr(args.weight_kg, args.height_cm, args.age, args.sex)
    protein_goal = args.weight_kg * args.protein_goal_g_per_kg

    daily["estimated_bmr"] = bmr
    daily["protein_goal_g"] = protein_goal
    daily["protein_adequacy_ratio"] = daily["protein_g"] / protein_goal
    daily["net_energy_balance"] = daily["calories"] - daily["estimated_bmr"] - daily["calories_burned"]

    # Macro percentages
    daily["protein_kcal"] = daily["protein_g"] * 4
    daily["carbs_kcal"] = daily["carbs_g"] * 4
    daily["fat_kcal"] = daily["fat_g"] * 9
    total_macro_kcal = (daily["protein_kcal"] + daily["carbs_kcal"] + daily["fat_kcal"]).replace(0, 1)
    daily["protein_pct"] = daily["protein_kcal"] / total_macro_kcal * 100
    daily["carbs_pct"] = daily["carbs_kcal"] / total_macro_kcal * 100
    daily["fat_pct"] = daily["fat_kcal"] / total_macro_kcal * 100

    # Recovery score
    # Simple feature-engineered score using protein + sleep
    daily["sleep_hours"] = daily["sleep_hours"].fillna(0)
    daily["recovery_score"] = (daily["protein_g"] * 0.6) + (daily["sleep_hours"] * 10)

    # Rolling trends
    daily = daily.sort_values("date")
    daily["calories_7d_avg"] = daily["calories"].rolling(7, min_periods=1).mean()
    daily["protein_7d_avg"] = daily["protein_g"].rolling(7, min_periods=1).mean()
    daily["sleep_7d_avg"] = daily["sleep_hours"].rolling(7, min_periods=1).mean()
    daily["recovery_7d_avg"] = daily["recovery_score"].rolling(7, min_periods=1).mean()

    # -----------------------------
    # Insights
    # -----------------------------
    insights = {}

    # General nutrition / recovery insights
    avg_energy_balance = float(daily["net_energy_balance"].mean()) if len(daily) else None
    avg_protein_ratio = float(daily["protein_adequacy_ratio"].mean()) if len(daily) else None
    avg_sleep = float(daily["sleep_hours"].mean()) if len(daily) else None

    if avg_energy_balance is not None:
        if avg_energy_balance < -300:
            insights["energy_balance"] = "Average energy balance suggests a meaningful calorie deficit."
        elif avg_energy_balance > 300:
            insights["energy_balance"] = "Average energy balance suggests a calorie surplus."
        else:
            insights["energy_balance"] = "Average energy balance is near maintenance."

    if avg_protein_ratio is not None:
        if avg_protein_ratio < 0.9:
            insights["protein"] = "Protein intake is below the target needed to support recovery."
        else:
            insights["protein"] = "Protein intake is generally meeting recovery targets."

    if avg_sleep is not None:
        if avg_sleep < 6:
            insights["sleep"] = "Average sleep is low and may reduce recovery and next-day performance."
        elif avg_sleep < 7:
            insights["sleep"] = "Average sleep is moderate but may still limit optimal recovery."
        else:
            insights["sleep"] = "Average sleep appears supportive of recovery."

    # Workout timing / fueling insights
    if workout_analysis is not None and len(workout_analysis) > 0:
        avg_gap = workout_analysis["meal_gap_hours"].dropna().mean()
        avg_pre_carbs = workout_analysis["pre_window_carbs_g"].mean()

        if pd.notna(avg_gap):
            if avg_gap > 4:
                insights["meal_timing"] = "Workouts often happen more than 4 hours after eating, which may reduce available energy."
            elif avg_gap < 1:
                insights["meal_timing"] = "Workouts often happen less than 1 hour after eating, which may feel too close for comfort."
            else:
                insights["meal_timing"] = "Meal timing before workouts is generally in a reasonable range."

        if avg_pre_carbs < 25:
            insights["pre_workout_carbs"] = "Carbohydrate intake before workouts appears low and may be limiting performance."
        else:
            insights["pre_workout_carbs"] = "Pre-workout carbohydrate intake looks adequate on average."

        if "stamina_score" in workout_analysis.columns and workout_analysis["stamina_score"].notna().any():
            corr_sleep_perf = None
            corr_carb_perf = None

            # Merge sleep to workout day
            wa = workout_analysis.copy()
            if daily_sleep is not None:
                wa = wa.merge(daily_sleep, on="date", how="left")
                corr_sleep_perf = safe_corr(wa["sleep_hours"], wa["stamina_score"])

            corr_carb_perf = safe_corr(wa["pre_window_carbs_g"], wa["stamina_score"])

            insights["correlations"] = {
                "sleep_vs_stamina_score": corr_sleep_perf,
                "pre_workout_carbs_vs_stamina_score": corr_carb_perf
            }

    # Recommendations
    recommendations = []

    if avg_protein_ratio is not None and avg_protein_ratio < 0.9:
        recommendations.append("Increase daily protein intake to better support recovery.")
    if avg_sleep is not None and avg_sleep < 7:
        recommendations.append("Improve sleep duration to support recovery and next-day performance.")
    if workout_analysis is not None and len(workout_analysis) > 0:
        if workout_analysis["pre_window_carbs_g"].mean() < 25:
            recommendations.append("Try consuming more carbohydrates within 1–3 hours before workouts.")
        if workout_analysis["meal_gap_hours"].dropna().mean() > 4:
            recommendations.append("Try eating closer to workouts to avoid long gaps without fuel.")

    insights["recommendations"] = recommendations

    # -----------------------------
    # Save outputs
    # -----------------------------
    daily_out = out_dir / "analytics_daily_summary.csv"
    daily.to_csv(daily_out, index=False)

    if workout_analysis is not None:
        workout_out = out_dir / "analytics_workout_fueling.csv"
        workout_analysis.to_csv(workout_out, index=False)

    insights_out = out_dir / "analytics_insights.json"
    with open(insights_out, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, default=str)

    print(f"Saved daily summary to: {daily_out}")
    if workout_analysis is not None:
        print(f"Saved workout fueling analysis to: {workout_out}")
    print(f"Saved insights to: {insights_out}")
    print(json.dumps(insights, indent=2, default=str))


if __name__ == "__main__":
    main()