"""Tests for Firestore schema validation logic."""

import re

import pytest

#  Inline validation logic

VALID_METRICS = {"strength", "stamina", "speed", "flexibility", "intelligence"}
VALID_SOURCES = {"wearable", "app_tasks", "workout_log", "manual"}
VALID_TOPICS = {"Biology", "Chemistry", "Physics", "Math", "History", "CS", "English"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
EXPECTED_COMPONENTS = {
    "strength": {"bench_1rm", "squat_1rm", "deadlift_1rm"},
    "stamina": {"vo2max", "resting_hr"},
    "speed": {"sprint_100m_sec"},
    "flexibility": {"sit_and_reach_cm", "shoulder_mobility"},
    "intelligence": {"memory_task", "reaction_time_ms"},
}


class SimpleReport:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def add_error(self, doc_id, msg):
        self.errors.append(f"[{doc_id}] {msg}")

    def add_warning(self, doc_id, msg):
        self.warnings.append(f"[{doc_id}] {msg}")


def validate_metric_event(d, doc_id, rpt, target_day):
    day = d.get("day")
    if day is None:
        rpt.add_error(doc_id, "missing 'day'")
    elif not DATE_RE.match(day):
        rpt.add_error(doc_id, f"'day' bad format: {day}")

    metric = d.get("metric")
    if metric is None:
        rpt.add_error(doc_id, "missing 'metric'")
    elif metric not in VALID_METRICS:
        rpt.add_error(doc_id, f"unknown metric '{metric}'")

    score = d.get("score")
    if score is None:
        rpt.add_error(doc_id, "missing 'score'")
    elif not isinstance(score, (int, float)):
        rpt.add_error(doc_id, "score not numeric")
    elif not (0 <= score <= 100):
        rpt.add_error(doc_id, f"score {score} out of range")

    conf = d.get("confidence")
    if conf is not None:
        if not isinstance(conf, (int, float)):
            rpt.add_error(doc_id, "confidence not numeric")
        elif not (0 <= conf <= 1):
            rpt.add_error(doc_id, f"confidence {conf} out of range")

    payload = d.get("payload")
    if payload and metric and metric in EXPECTED_COMPONENTS:
        if not isinstance(payload, dict):
            rpt.add_error(doc_id, "payload not dict")
        else:
            missing = EXPECTED_COMPONENTS[metric] - set(payload.keys())
            if missing:
                rpt.add_error(doc_id, f"payload missing keys: {missing}")


def validate_sleep_log(d, doc_id, rpt):
    hrs = d.get("sleepHours")
    if hrs is None:
        rpt.add_error(doc_id, "missing sleepHours")
    elif not isinstance(hrs, (int, float)):
        rpt.add_error(doc_id, "sleepHours not numeric")
    elif not (0 <= hrs <= 24):
        rpt.add_error(doc_id, f"sleepHours {hrs} out of range")

    quality = d.get("quality")
    if quality is not None:
        if not isinstance(quality, int):
            rpt.add_error(doc_id, "quality not int")
        elif not (1 <= quality <= 5):
            rpt.add_error(doc_id, f"quality {quality} out of range")

    bed = d.get("bedTime")
    if bed and not TIME_RE.match(bed):
        rpt.add_error(doc_id, f"bedTime bad format: {bed}")

    wake = d.get("wakeTime")
    if wake and not TIME_RE.match(wake):
        rpt.add_error(doc_id, f"wakeTime bad format: {wake}")


def validate_quiz_attempt(d, doc_id, rpt):
    topic = d.get("topic")
    if topic and topic not in VALID_TOPICS:
        rpt.add_error(doc_id, f"unknown topic: {topic}")

    n_q = d.get("num_questions")
    n_c = d.get("num_correct")
    if n_c is not None and isinstance(n_c, int) and n_c < 0:
        rpt.add_error(doc_id, f"num_correct negative: {n_c}")
    if (
        n_c is not None
        and n_q is not None
        and isinstance(n_c, int)
        and isinstance(n_q, int)
        and n_c > n_q
    ):
        rpt.add_error(doc_id, "num_correct > num_questions")

    diff = d.get("difficulty")
    if diff is not None and isinstance(diff, int) and not (1 <= diff <= 5):
        rpt.add_error(doc_id, f"difficulty {diff} out of range")


#  Metric Event Tests


class TestMetricEventValidation:

    def test_valid_event_no_errors(self):
        rpt = SimpleReport()
        doc = {
            "day": "2026-02-22",
            "metric": "strength",
            "type": "model_update",
            "source": "workout_log",
            "score": 75.5,
            "confidence": 0.82,
            "delta": 1.2,
            "payload": {"bench_1rm": 200, "squat_1rm": 300, "deadlift_1rm": 350},
        }
        validate_metric_event(doc, "test1", rpt, "2026-02-22")
        assert rpt.errors == []

    def test_missing_day(self):
        rpt = SimpleReport()
        validate_metric_event({"metric": "speed", "score": 50}, "t", rpt, "2026-02-22")
        assert any("day" in e for e in rpt.errors)

    def test_bad_day_format(self):
        rpt = SimpleReport()
        validate_metric_event(
            {"day": "02-22-2026", "metric": "speed", "score": 50},
            "t",
            rpt,
            "2026-02-22",
        )
        assert any("bad format" in e for e in rpt.errors)

    def test_unknown_metric(self):
        rpt = SimpleReport()
        validate_metric_event(
            {"day": "2026-02-22", "metric": "charisma", "score": 50},
            "t",
            rpt,
            "2026-02-22",
        )
        assert any("unknown metric" in e for e in rpt.errors)

    def test_missing_score(self):
        rpt = SimpleReport()
        validate_metric_event(
            {"day": "2026-02-22", "metric": "speed"}, "t", rpt, "2026-02-22"
        )
        assert any("score" in e for e in rpt.errors)

    def test_score_out_of_range(self):
        rpt = SimpleReport()
        validate_metric_event(
            {"day": "2026-02-22", "metric": "speed", "score": 150},
            "t",
            rpt,
            "2026-02-22",
        )
        assert any("out of range" in e for e in rpt.errors)

    def test_confidence_out_of_range(self):
        rpt = SimpleReport()
        validate_metric_event(
            {"day": "2026-02-22", "metric": "speed", "score": 50, "confidence": 1.5},
            "t",
            rpt,
            "2026-02-22",
        )
        assert any("confidence" in e for e in rpt.errors)

    def test_payload_missing_keys(self):
        rpt = SimpleReport()
        validate_metric_event(
            {
                "day": "2026-02-22",
                "metric": "strength",
                "score": 50,
                "payload": {"bench_1rm": 200},
            },
            "t",
            rpt,
            "2026-02-22",
        )
        assert any("payload missing" in e for e in rpt.errors)

    @pytest.mark.parametrize(
        "metric,keys",
        [
            ("strength", {"bench_1rm", "squat_1rm", "deadlift_1rm"}),
            ("stamina", {"vo2max", "resting_hr"}),
            ("speed", {"sprint_100m_sec"}),
            ("flexibility", {"sit_and_reach_cm", "shoulder_mobility"}),
            ("intelligence", {"memory_task", "reaction_time_ms"}),
        ],
    )
    def test_all_metrics_complete_payload(self, metric, keys):
        rpt = SimpleReport()
        payload = {k: 1 for k in keys}
        validate_metric_event(
            {"day": "2026-02-22", "metric": metric, "score": 50, "payload": payload},
            "t",
            rpt,
            "2026-02-22",
        )
        assert not any("payload" in e for e in rpt.errors)


#  Sleep Log Tests


class TestSleepLogValidation:

    def test_valid_sleep_log(self):
        rpt = SimpleReport()
        validate_sleep_log(
            {"sleepHours": 7.5, "quality": 4, "bedTime": "23:30", "wakeTime": "07:00"},
            "t",
            rpt,
        )
        assert rpt.errors == []

    def test_missing_sleep_hours(self):
        rpt = SimpleReport()
        validate_sleep_log({"quality": 3}, "t", rpt)
        assert any("sleepHours" in e for e in rpt.errors)

    def test_sleep_hours_out_of_range(self):
        rpt = SimpleReport()
        validate_sleep_log({"sleepHours": 25.0, "quality": 3}, "t", rpt)
        assert any("out of range" in e for e in rpt.errors)

    def test_quality_out_of_range(self):
        rpt = SimpleReport()
        validate_sleep_log({"sleepHours": 7.0, "quality": 0}, "t", rpt)
        assert any("quality" in e for e in rpt.errors)

    def test_bad_time_format(self):
        rpt = SimpleReport()
        validate_sleep_log(
            {"sleepHours": 7.0, "bedTime": "11:30 PM", "wakeTime": "7am"},
            "t",
            rpt,
        )
        assert any("bedTime" in e for e in rpt.errors)
        assert any("wakeTime" in e for e in rpt.errors)


#  Quiz Attempt Tests


class TestQuizAttemptValidation:

    def test_valid_quiz(self):
        rpt = SimpleReport()
        validate_quiz_attempt(
            {"topic": "Math", "num_questions": 10, "num_correct": 8, "difficulty": 3},
            "t",
            rpt,
        )
        assert rpt.errors == []

    def test_invalid_topic(self):
        rpt = SimpleReport()
        validate_quiz_attempt({"topic": "Astrology"}, "t", rpt)
        assert any("topic" in e for e in rpt.errors)

    def test_negative_num_correct(self):
        rpt = SimpleReport()
        validate_quiz_attempt(
            {"topic": "Math", "num_questions": 10, "num_correct": -1},
            "t",
            rpt,
        )
        assert any("negative" in e for e in rpt.errors)

    def test_correct_exceeds_questions(self):
        rpt = SimpleReport()
        validate_quiz_attempt(
            {"topic": "Math", "num_questions": 10, "num_correct": 15},
            "t",
            rpt,
        )
        assert any("num_correct > num_questions" in e for e in rpt.errors)

    def test_difficulty_out_of_range(self):
        rpt = SimpleReport()
        validate_quiz_attempt({"topic": "Math", "difficulty": 6}, "t", rpt)
        assert any("difficulty" in e for e in rpt.errors)

    @pytest.mark.parametrize("topic", VALID_TOPICS)
    def test_all_valid_topics_accepted(self, topic):
        rpt = SimpleReport()
        validate_quiz_attempt({"topic": topic}, "t", rpt)
        assert not any("topic" in e for e in rpt.errors)
