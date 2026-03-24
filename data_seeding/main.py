#!/usr/bin/env python3
"""
data_seeding/main.py — Seed Firestore with Level Up synthetic data.

Firestore layout:
  users/{uid}                          user profile + display name
  users/{uid}/metrics/current          latest aggregate scores for all metrics
  users/{uid}/metric_events/{id}       daily score update events (all metrics)
  users/{uid}/daily_rollups/{date}     optional daily summary
  users/{uid}/sleep_logs/{date}        sleep data
  users/{uid}/quiz_attempts/{id}       quiz results
  users/{uid}/flexibility_workouts/{date}   ← NEW: per-session workout data

Flexibility workout fields (NEW):
  date                  YYYY-MM-DD
  session_duration_min  int  20-90
  exercise_type         str  one of EXERCISE_TYPES
  effort_level          int  1-5  (1=easy, 5=max effort)
  sit_and_reach_cm      float  derived from score + noise
  shoulder_mobility     int  0-5  derived from score
  score_before          float  score at start of session (from previous event)
  score_after           float  new score after session
  score_delta           float  score_after - score_before
  streak_days           int  consecutive days with a workout logged
  rest_days_before      int  days since last workout
  notes                 str  ""

Score dynamics (realistic for a lifestyle gamification app):
  - Base progression: slow upward trend (harder to improve at higher scores)
  - Session gain: effort × duration effect, capped by diminishing returns
  - Consistency bonus: streak multiplier
  - Decay: small score drop per rest day (muscle memory fades)
  - Noise: ±random biological variation

Run:
  python data_seeding/main.py \
    --service-account secrets/firebase-admin.json \
    --num-users 20 --days 90 --write-rollups --seed 42
"""

from __future__ import annotations

import argparse
import random
import string
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore

#  Constants 

METRICS = ["strength", "stamina", "speed", "flexibility", "intelligence"]
TOPICS  = ["Biology", "Chemistry", "Physics", "Math", "History", "CS", "English"]
SEXES   = ["Male", "Female"]

EXERCISE_TYPES = ["stretching", "yoga", "pilates", "mobility", "foam_rolling"]

# Probability of doing a flexibility workout on any given day (realistic ~4×/week)
WORKOUT_DAY_PROB = 0.55

# Score gain per session — bounded by effort and duration, with diminishing returns
MAX_GAIN_PER_SESSION = 1.8   # hard cap
MIN_GAIN_PER_SESSION = 0.05  # always gain something if you show up

# Score decay per rest day (muscle memory fades slowly)
DECAY_PER_REST_DAY = 0.10

# Consistency bonus: extra gain per streak day (up to 5-day streak)
STREAK_BONUS_PER_DAY = 0.08
MAX_STREAK_BONUS    = 0.40


#  Utility 

def rand_uid(n: int = 28) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def score_to_level(score: float) -> int:
    return int(clamp(score) // 10)


#  Score dynamics 

def session_score_gain(
    current_score: float,
    effort_level: int,
    duration_min: int,
    streak_days: int,
) -> float:
    """
    Compute the score gain from a single flexibility session.

    Gain decreases as score approaches 100 (diminishing returns).
    Higher effort and longer sessions yield more gain.
    Consistency streak adds a small bonus.
    """
    # Diminishing returns: full gain below 50, halved above 75, minimal above 90
    dr_factor = 1.0 - clamp((current_score - 50) / 60, lo=0.0, hi=0.85)

    # Effort contribution: effort 3 = baseline, 1 = 40%, 5 = 160%
    effort_factor = 0.4 + (effort_level - 1) * 0.3   # 0.4, 0.7, 1.0, 1.3, 1.6

    # Duration contribution: 45 min = baseline 1.0, scales log-ish
    duration_factor = clamp(duration_min / 45.0, lo=0.3, hi=1.5)

    streak_bonus = min(streak_days * STREAK_BONUS_PER_DAY, MAX_STREAK_BONUS)

    raw_gain = MAX_GAIN_PER_SESSION * effort_factor * duration_factor * dr_factor
    gain = max(MIN_GAIN_PER_SESSION, raw_gain) + streak_bonus

    # Biological noise ±0.3
    gain += random.gauss(0, 0.3)

    return clamp(gain, lo=0.0, hi=MAX_GAIN_PER_SESSION + MAX_STREAK_BONUS + 0.5)


def apply_rest_decay(score: float, rest_days: int) -> float:
    """Score decays slowly during rest (max 3 days worth)."""
    effective_rest = min(rest_days, 3)
    return clamp(score - effective_rest * DECAY_PER_REST_DAY)


#  Component derivations 

def flexibility_components(score: float, duration_min: int, effort_level: int) -> dict:
    """Derive measurable flexibility sub-scores from the composite score."""
    noise_reach = random.gauss(0, 0.8)
    sit_and_reach = round(clamp(10.0 + score * 0.42 + noise_reach, lo=2.0, hi=55.0), 1)
    # shoulder mobility: 0-5 scale, tied to score level + effort
    shoulder_raw = score_to_level(score) * 0.55 + effort_level * 0.1
    shoulder_mobility = int(clamp(round(shoulder_raw), lo=0, hi=5))
    return {
        "sit_and_reach_cm": sit_and_reach,
        "shoulder_mobility": shoulder_mobility,
        "session_duration_min": duration_min,
        "effort_level": effort_level,
    }


#  Metric components (non-flexibility) 

def metric_components(metric: str, score: float) -> dict:
    if metric == "strength":
        return {
            "bench_1rm": int(95 + score * 2.0),
            "squat_1rm": int(135 + score * 2.5),
            "deadlift_1rm": int(155 + score * 2.8),
        }
    if metric == "stamina":
        return {
            "vo2max": round(30 + score * 0.25, 1),
            "resting_hr": int(75 - score * 0.15),
        }
    if metric == "speed":
        return {"sprint_100m_sec": clamp(round(18.5 - score * 0.05, 2), lo=9.5, hi=25.0)}
    if metric == "flexibility":
        return flexibility_components(score, duration_min=45, effort_level=3)
    if metric == "intelligence":
        return {"memory_task": round(score / 100.0, 2), "reaction_time_ms": int(420 - score * 2.3)}
    return {}


def metric_sources(metric: str) -> list:
    if metric == "stamina":
        return ["wearable"]
    if metric == "intelligence":
        return ["app_tasks"]
    if metric in ("speed", "strength"):
        return ["workout_log"]
    return ["manual"]


#  Firestore helpers 

def init_firestore(service_account_path: str):
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()


def seed_user(db, uid: str, display_name: str, tz: str, profile: dict):
    db.collection("users").document(uid).set({
        "createdAt": firestore.SERVER_TIMESTAMP,
        "displayName": display_name,
        "photoUrl": None,
        "timezone": tz,
        "privacy": {"shareAggregate": False, "shareLeaderboard": False},
        "active": True,
        "profile": profile,
    })


def seed_current_metrics(db, uid: str, base_scores: dict):
    doc = {"updatedAt": firestore.SERVER_TIMESTAMP, "version": 1}
    for m, s in base_scores.items():
        doc[m] = {
            "score": round(s, 1),
            "level": score_to_level(s),
            "confidence": round(random.uniform(0.45, 0.85), 2),
            "sources": metric_sources(m),
            "components": metric_components(m, s),
        }
    db.collection("users").document(uid).collection("metrics").document("current").set(doc)


#  Core seeding: metric_events (all metrics, daily) 

def seed_history(
    db, uid: str, start_day: datetime, days: int, base_scores: dict, write_rollups: bool
):
    """
    Seeds daily metric_events for all metrics.
    Flexibility scores here are the *aggregated* model score (derived from workout sessions).
    The detailed per-session data lives in flexibility_workouts subcollection.
    """
    events_col = db.collection("users").document(uid).collection("metric_events")
    rollups_col = db.collection("users").document(uid).collection("daily_rollups")
    scores = dict(base_scores)
    batch = db.batch()
    ops = 0

    for i in range(days):
        day_dt = start_day + timedelta(days=i)
        day_str = day_dt.strftime("%Y-%m-%d")
        daily_metrics = {}

        for m in METRICS:
            if m == "flexibility":
                # Flexibility score is updated by workout sessions (seeded separately).
                # Here we just record the current running score as a daily snapshot.
                drift = random.gauss(0, 0.2)
            else:
                drift = random.uniform(-1.2, 1.4) * 0.15 + random.uniform(-0.8, 0.8) * 0.35

            scores[m] = clamp(scores[m] + drift)
            confidence = round(random.uniform(0.45, 0.85), 2)
            new_score = round(scores[m], 1)

            doc_ref = events_col.document()
            batch.set(doc_ref, {
                "ts":         firestore.SERVER_TIMESTAMP,
                "day":        day_str,
                "type":       "model_update",
                "source":     metric_sources(m)[0],
                "metric":     m,
                "score":      new_score,
                "delta":      round(new_score - round(base_scores[m], 1), 1),
                "confidence": confidence,
                "payload":    metric_components(m, new_score),
            })
            ops += 1
            daily_metrics[m] = {"score": new_score, "confidence": confidence}

            if ops >= 450:
                batch.commit()
                batch = db.batch()
                ops = 0

        if write_rollups:
            rollups_col.document(day_str).set({
                "day":       day_str,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "metrics":   daily_metrics,
            })
            ops += 1
            if ops >= 450:
                batch.commit()
                batch = db.batch()
                ops = 0

    if ops > 0:
        batch.commit()


#  NEW: flexibility_workouts subcollection 

def seed_flexibility_workouts(
    db,
    uid: str,
    start_day: datetime,
    days: int,
    base_flexibility_score: float,
):
    """
    Seeds per-session flexibility workout data into:
      users/{uid}/flexibility_workouts/{YYYY-MM-DD}

    Each day has at most one session. Sessions occur ~WORKOUT_DAY_PROB of days.
    Score dynamics are realistic:
      - Gain depends on effort × duration with diminishing returns
      - Consistency streak adds bonus
      - Rest days cause small decay
    """
    col = db.collection("users").document(uid).collection("flexibility_workouts")
    batch = db.batch()
    ops = 0
    now = datetime.now(timezone.utc)

    score = base_flexibility_score
    streak_days = 0
    last_workout_day: int | None = None  # index of last workout day

    for i in range(days):
        day_dt = start_day + timedelta(days=i)
        day_str = day_dt.strftime("%Y-%m-%d")

        did_workout = random.random() < WORKOUT_DAY_PROB

        if not did_workout:
            streak_days = 0
            # Small decay on rest days
            rest_so_far = (i - last_workout_day) if last_workout_day is not None else 0
            if rest_so_far > 0:
                score = apply_rest_decay(score, rest_days=1)
            continue

        # Workout happened
        rest_days_before = (i - last_workout_day) if last_workout_day is not None else 0
        streak_days = streak_days + 1 if rest_days_before <= 1 else 1

        effort_level    = random.choices([1, 2, 3, 4, 5], weights=[5, 15, 40, 30, 10])[0]
        duration_min    = int(random.gauss(45, 15))
        duration_min    = max(15, min(90, duration_min))
        exercise_type   = random.choice(EXERCISE_TYPES)

        score_before = round(score, 2)
        gain = session_score_gain(score, effort_level, duration_min, streak_days)
        score = clamp(score + gain)
        score_after  = round(score, 2)

        components = flexibility_components(score_after, duration_min, effort_level)

        # Workout logged at a realistic time during the day
        workout_hour = random.choices(
            [6, 7, 8, 12, 17, 18, 19, 20],
            weights=[10, 15, 10, 10, 20, 20, 10, 5],
        )[0]
        workout_ts = day_dt.replace(
            hour=workout_hour,
            minute=random.randint(0, 59),
            tzinfo=timezone.utc,
        )

        doc = {
            "user_id":             uid,
            "date":                day_str,
            "timestamp":           workout_ts,
            "exercise_type":       exercise_type,
            "session_duration_min": duration_min,
            "effort_level":        effort_level,
            "sit_and_reach_cm":    components["sit_and_reach_cm"],
            "shoulder_mobility":   components["shoulder_mobility"],
            "score_before":        score_before,
            "score_after":         score_after,
            "score_delta":         round(score_after - score_before, 2),
            "streak_days":         streak_days,
            "rest_days_before":    rest_days_before,
            "notes":               "",
            "seededAt":            now,
        }

        ref = col.document(day_str)
        batch.set(ref, doc)
        ops += 1
        last_workout_day = i

        if ops >= 450:
            batch.commit()
            batch = db.batch()
            ops = 0

    if ops > 0:
        batch.commit()


#  Sleep logs 

def seed_sleep_logs(db, uid: str, start_day: datetime, days: int):
    col = db.collection("users").document(uid).collection("sleep_logs")
    batch = db.batch()
    ops = 0
    now = datetime.now(timezone.utc)

    for i in range(days):
        day_dt = start_day + timedelta(days=i)
        day_str = day_dt.strftime("%Y-%m-%d")

        if random.random() < 0.15:
            continue

        bed_hour  = random.choice([22, 23, 0, 1])
        bed_min   = random.randint(0, 59)
        sleep_hrs = round(max(2.0, min(14.0, random.gauss(7.0, 1.2))), 2)
        wake_hour = (bed_hour + int(sleep_hrs)) % 24
        wake_min  = random.randint(0, 59)
        quality   = random.randint(1, 5)

        wake_date = day_dt.date()
        if bed_hour in (22, 23) and wake_hour in range(0, 12):
            wake_date = (day_dt + timedelta(days=1)).date()

        created_at = datetime(
            wake_date.year, wake_date.month, wake_date.day,
            wake_hour, wake_min, 0, tzinfo=timezone.utc,
        )

        batch.set(col.document(day_str), {
            "id":          f"{uid}_{day_str}",
            "user_id":     uid,
            "date":        day_str,
            "bedTime":     f"{bed_hour:02d}:{bed_min:02d}",
            "wakeTime":    f"{wake_hour:02d}:{wake_min:02d}",
            "sleepHours":  float(sleep_hrs),
            "quality":     quality,
            "note":        "",
            "createdAt":   created_at,
            "seededAt":    now,
        })
        ops += 1
        if ops >= 450:
            batch.commit()
            batch = db.batch()
            ops = 0

    if ops > 0:
        batch.commit()


#  Quiz attempts 

def seed_quiz_attempts(db, uid: str, start_day: datetime, days: int):
    col = db.collection("users").document(uid).collection("quiz_attempts")
    batch = db.batch()
    ops = 0
    now = datetime.now(timezone.utc)

    for i in range(days):
        day_dt = start_day + timedelta(days=i)
        n_quizzes = random.choices([0, 1, 2, 3], weights=[40, 35, 15, 10])[0]

        for q in range(n_quizzes):
            total      = random.choice([5, 10, 15, 20])
            correct    = random.randint(0, total)
            time_taken = random.randint(30, 600)
            ts = (day_dt + timedelta(
                hours=random.randint(8, 22), minutes=random.randint(0, 59)
            )).astimezone(timezone.utc)
            attempt_id = f"{uid}_{day_dt.strftime('%Y%m%d')}_{q}"

            batch.set(col.document(attempt_id), {
                "id":                           attempt_id,
                "user_id":                      uid,
                "timestamp":                    ts,
                "quiz_id":                      f"quiz_{random.randint(1, 100)}",
                "topic":                        random.choice(TOPICS),
                "num_questions":                int(total),
                "num_correct":                  int(correct),
                "total_time_seconds":           int(time_taken),
                "avg_time_per_question_seconds": round(time_taken / total, 2) if total else 0.0,
                "difficulty":                   int(random.randint(1, 5)),
                "percent":                      int(round(correct / total * 100)) if total else 0,
                "seededAt":                     now,
            })
            ops += 1
            if ops >= 450:
                batch.commit()
                batch = db.batch()
                ops = 0

    if ops > 0:
        batch.commit()


#  Profile generation 

def generate_profile() -> dict:
    return {
        "age":       random.randint(14, 60),
        "sex":       random.choice(SEXES),
        "height_cm": random.randint(150, 195),
        "weight_kg": random.randint(45, 120),
    }


#  Entry point 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--num-users",    type=int,  default=20)
    parser.add_argument("--days",         type=int,  default=90)
    parser.add_argument("--timezone",     default="America/New_York")
    parser.add_argument("--write-rollups", action="store_true")
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--no-sleep",     action="store_true")
    parser.add_argument("--no-quizzes",   action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    db = init_firestore(args.service_account)

    start_day = (
        datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        - timedelta(days=args.days)
    )

    for idx in range(args.num_users):
        uid          = rand_uid()
        display_name = f"User{idx + 1}"
        profile      = generate_profile()

        base_scores = {
            "strength":     random.uniform(30, 80),
            "stamina":      random.uniform(25, 75),
            "speed":        random.uniform(20, 70),
            "flexibility":  random.uniform(20, 65),   # lower start — room to grow
            "intelligence": random.uniform(35, 75),
        }

        seed_user(db, uid, display_name, args.timezone, profile)
        seed_current_metrics(db, uid, base_scores)
        seed_history(db, uid, start_day, args.days, base_scores, args.write_rollups)

        # NEW: seed detailed per-session flexibility workout data
        seed_flexibility_workouts(
            db, uid, start_day, args.days, base_scores["flexibility"]
        )

        if not args.no_sleep:
            seed_sleep_logs(db, uid, start_day, args.days)
        if not args.no_quizzes:
            seed_quiz_attempts(db, uid, start_day, args.days)

        print(f"Seeded uid={uid}  name={display_name}  flex_base={base_scores['flexibility']:.1f}")

    print("Done.")


if __name__ == "__main__":
    main()