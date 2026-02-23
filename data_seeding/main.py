import argparse
import random
import string
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore


METRICS = ["strength", "stamina", "speed", "flexibility", "intelligence"]


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


def seed_user(db, uid: str, display_name: str, tz: str):
    user_ref = db.collection("users").document(uid)
    user_ref.set(
        {
            "createdAt": firestore.SERVER_TIMESTAMP,
            "displayName": display_name,
            "photoUrl": None,
            "timezone": tz,
            "privacy": {"shareAggregate": False, "shareLeaderboard": False},
            "active": True,
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


def seed_history(db, uid: str, start_day: datetime, days: int, base_scores: dict, write_rollups: bool):
    events_col = db.collection("users").document(uid).collection("metric_events")
    rollups_col = db.collection("users").document(uid).collection("daily_rollups")

    scores = dict(base_scores)

    # Batch writes (Firestore max ~500 ops per batch)
    batch = db.batch()
    ops = 0

    for i in range(days):
        day_dt = start_day + timedelta(days=i)
        day_str = day_dt.strftime("%Y-%m-%d")

        daily_metrics = {}

        for m in METRICS:
            # random walk
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-account", required=True, help="Path to Firebase service account JSON")
    parser.add_argument("--num-users", type=int, default=5)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--write-rollups", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    db = init_firestore(args.service_account)

    # Seed starting from today
    start_day = datetime.now(timezone.utc) - timedelta(days=args.days)

    for idx in range(args.num_users):
        uid = rand_uid()
        display_name = f"User{idx+1}"

        # Base scores
        base_scores = {
            "strength": random.uniform(30, 80),
            "stamina": random.uniform(25, 75),
            "speed": random.uniform(20, 70),
            "flexibility": random.uniform(30, 85),
            "intelligence": random.uniform(35, 75),
        }

        seed_user(db, uid, display_name, args.timezone)
        seed_current_metrics(db, uid, base_scores)
        seed_history(db, uid, start_day, args.days, base_scores, args.write_rollups)

        print(f"Seeded uid={uid} name={display_name}")

    print("Done.")


if __name__ == "__main__":
    main()