#!/usr/bin/env python3
"""
Seed Firestore with synthetic Level Up data.

Keeps existing schema:
  users/{uid}
  users/{uid}/metrics/current
  users/{uid}/metric_events/*
  users/{uid}/daily_rollups/{YYYY-MM-DD}  (optional)

Adds app-like raw data into Firestore:
  users/{uid}.profile: {age, sex, height_cm, weight_kg}
  users/{uid}/sleep_logs/{YYYY-MM-DD}
  users/{uid}/quiz_attempts/{attemptId}

Run:
  python seed_firestore.py --service-account path/to/serviceAccount.json --num-users 5 --days 30 --write-rollups --seed 42
"""

import argparse
import random
import string
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore

METRICS = ["strength", "stamina", "speed", "flexibility", "intelligence"]
TOPICS = ["Biology", "Chemistry", "Physics", "Math", "History", "CS", "English"]
SEXES = ["Male", "Female"]


def rand_uid(n=28) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def score_to_level(score: float) -> int:
    return int(clamp(score) // 10)


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
        sprint = round(18.5 - score * 0.05, 2)
        return {"sprint_100m_sec": clamp(sprint, 9.5, 25.0)}
    if metric == "flexibility":
        return {
            "sit_and_reach_cm": round(10 + score * 0.4, 1),
            "shoulder_mobility": int(clamp(score_to_level(score), 0, 5)),
        }
    if metric == "intelligence":
        return {
            "memory_task": round(score / 100.0, 2),
            "reaction_time_ms": int(420 - score * 2.3),
        }
    return {}


def metric_sources(metric: str):
    if metric in ("stamina",):
        return ["wearable"]
    if metric in ("intelligence",):
        return ["app_tasks"]
    if metric in ("speed", "strength"):
        return ["workout_log"]
    return ["manual"]


def init_firestore(service_account_path: str):
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()


def seed_user(db, uid: str, display_name: str, tz: str, profile: dict):
    user_ref = db.collection("users").document(uid)
    user_ref.set(
        {
            "createdAt": firestore.SERVER_TIMESTAMP,
            "displayName": display_name,
            "photoUrl": None,
            "timezone": tz,
            "privacy": {"shareAggregate": False, "shareLeaderboard": False},
            "active": True,
            "profile": profile,
        }
    )


def seed_current_metrics(db, uid: str, base_scores: dict):
    current_ref = (
        db.collection("users").document(uid).collection("metrics").document("current")
    )
    current_doc = {"updatedAt": firestore.SERVER_TIMESTAMP, "version": 1}
    for m, s in base_scores.items():
        confidence = round(random.uniform(0.45, 0.85), 2)
        current_doc[m] = {
            "score": round(s, 1),
            "level": score_to_level(s),
            "confidence": confidence,
            "sources": metric_sources(m),
            "components": metric_components(m, s),
        }
    current_ref.set(current_doc)


def seed_history(
    db, uid: str, start_day: datetime, days: int, base_scores: dict, write_rollups: bool
):
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
            drift = random.uniform(-1.2, 1.4)
            noise = random.uniform(-0.8, 0.8)
            scores[m] = clamp(scores[m] + 0.15 * drift + 0.35 * noise)
            confidence = round(random.uniform(0.45, 0.85), 2)
            new_score = round(scores[m], 1)

            doc_ref = events_col.document()
            event_doc = {
                "ts": firestore.SERVER_TIMESTAMP,
                "day": day_str,
                "type": "model_update",
                "source": metric_sources(m)[0],
                "metric": m,
                "score": new_score,
                "delta": round(new_score - round(base_scores[m], 1), 1),
                "confidence": confidence,
                "payload": metric_components(m, new_score),
            }
            batch.set(doc_ref, event_doc)
            ops += 1
            daily_metrics[m] = {"score": new_score, "confidence": confidence}

            if ops >= 450:
                batch.commit()
                batch = db.batch()
                ops = 0

        if write_rollups:
            rollup_ref = rollups_col.document(day_str)
            rollup_doc = {
                "day": day_str,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "metrics": daily_metrics,
            }
            batch.set(rollup_ref, rollup_doc)
            ops += 1
            if ops >= 450:
                batch.commit()
                batch = db.batch()
                ops = 0

    if ops > 0:
        batch.commit()


def generate_profile():
    return {
        "age": random.randint(14, 60),
        "sex": random.choice(SEXES),
        "height_cm": random.randint(150, 195),
        "weight_kg": random.randint(45, 120),
    }


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

        bed_hour = random.choice([22, 23, 0, 1])
        bed_min = random.randint(0, 59)

        sleep_hours = round(random.gauss(7.0, 1.2), 2)
        sleep_hours = max(2.0, min(14.0, sleep_hours))

        wake_hour = (bed_hour + int(sleep_hours)) % 24
        wake_min = random.randint(0, 59)

        quality = random.randint(1, 5)

        wake_date = day_dt.date()
        if bed_hour in (22, 23) and wake_hour in range(0, 12):
            wake_date = (day_dt + timedelta(days=1)).date()

        created_at = datetime(
            wake_date.year,
            wake_date.month,
            wake_date.day,
            wake_hour,
            wake_min,
            0,
            tzinfo=timezone.utc,
        )

        doc = {
            "id": f"{uid}_{day_str}",
            "user_id": uid,
            "date": day_str,
            "bedTime": f"{bed_hour:02d}:{bed_min:02d}",
            "wakeTime": f"{wake_hour:02d}:{wake_min:02d}",
            "sleepHours": float(sleep_hours),
            "quality": quality,
            "note": "",
            "createdAt": created_at,
            "seededAt": now,
        }

        ref = col.document(day_str)
        batch.set(ref, doc)
        ops += 1

        if ops >= 450:
            batch.commit()
            batch = db.batch()
            ops = 0

    if ops > 0:
        batch.commit()


def seed_quiz_attempts(db, uid: str, start_day: datetime, days: int):
    col = db.collection("users").document(uid).collection("quiz_attempts")
    batch = db.batch()
    ops = 0
    now = datetime.now(timezone.utc)

    for i in range(days):
        day_dt = start_day + timedelta(days=i)
        n_quizzes = random.choices([0, 1, 2, 3], weights=[40, 35, 15, 10])[0]

        for q in range(n_quizzes):
            total = random.choice([5, 10, 15, 20])
            correct = random.randint(0, total)
            time_taken = random.randint(30, 600)

            ts = (
                day_dt
                + timedelta(hours=random.randint(8, 22), minutes=random.randint(0, 59))
            ).astimezone(timezone.utc)

            attempt_id = f"{uid}_{day_dt.strftime('%Y%m%d')}_{q}"

            doc = {
                "id": attempt_id,
                "user_id": uid,
                "timestamp": ts,
                "quiz_id": f"quiz_{random.randint(1, 100)}",
                "topic": random.choice(TOPICS),
                "num_questions": int(total),
                "num_correct": int(correct),
                "total_time_seconds": int(time_taken),
                "avg_time_per_question_seconds": (
                    round(time_taken / total, 2) if total > 0 else 0.0
                ),
                "difficulty": int(random.randint(1, 5)),
                "percent": int(round(correct / total * 100)) if total > 0 else 0,
                "seededAt": now,
            }

            ref = col.document(attempt_id)
            batch.set(ref, doc)
            ops += 1

            if ops >= 450:
                batch.commit()
                batch = db.batch()
                ops = 0

    if ops > 0:
        batch.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-account", required=True, help="Path to Firebase service account JSON"
    )
    parser.add_argument("--num-users", type=int, default=5)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--write-rollups", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-sleep", action="store_true", help="Do not seed sleep logs"
    )
    parser.add_argument(
        "--no-quizzes", action="store_true", help="Do not seed quiz attempts"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    db = init_firestore(args.service_account)

    start_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=args.days)

    for idx in range(args.num_users):
        uid = rand_uid()
        display_name = f"User{idx+1}"
        profile = generate_profile()

        base_scores = {
            "strength": random.uniform(30, 80),
            "stamina": random.uniform(25, 75),
            "speed": random.uniform(20, 70),
            "flexibility": random.uniform(30, 85),
            "intelligence": random.uniform(35, 75),
        }

        seed_user(db, uid, display_name, args.timezone, profile)
        seed_current_metrics(db, uid, base_scores)
        seed_history(db, uid, start_day, args.days, base_scores, args.write_rollups)

        if not args.no_sleep:
            seed_sleep_logs(db, uid, start_day, args.days)
        if not args.no_quizzes:
            seed_quiz_attempts(db, uid, start_day, args.days)

        print(f"Seeded uid={uid} name={display_name}")

    print("Done.")


if __name__ == "__main__":
    main()