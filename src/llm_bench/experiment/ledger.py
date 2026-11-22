from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from llm_bench.experiment.types import ExperimentSpec, TrialResult, canonical_json

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS specs (
    spec_id TEXT PRIMARY KEY,
    spec_hash TEXT NOT NULL UNIQUE,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    experiment_hash TEXT NOT NULL UNIQUE,
    spec_id TEXT NOT NULL REFERENCES specs(spec_id),
    strategy TEXT NOT NULL,
    sequence_no INTEGER NOT NULL CHECK (sequence_no >= 0),
    repeat_index INTEGER NOT NULL CHECK (repeat_index >= 0),
    candidate_id TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_type TEXT,
    failure_detail TEXT,
    quality_passed INTEGER NOT NULL CHECK (quality_passed IN (0, 1)),
    quality_json TEXT NOT NULL,
    performance_json TEXT NOT NULL,
    baseline_delta_json TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    runtime_fingerprint TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    load_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reflections (
    reflection_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL UNIQUE REFERENCES trials(trial_id),
    spec_id TEXT NOT NULL REFERENCES specs(spec_id),
    strategy TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    positive_experience INTEGER NOT NULL CHECK (positive_experience IN (0, 1)),
    negative_experience INTEGER NOT NULL CHECK (negative_experience IN (0, 1)),
    evidence_json TEXT NOT NULL,
    recommendations_json TEXT NOT NULL,
    memory_summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_strategy_sequence
    ON trials(spec_id, strategy, sequence_no);
CREATE INDEX IF NOT EXISTS idx_trials_candidate
    ON trials(spec_id, candidate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_trials_quality
    ON trials(spec_id, quality_passed, status);
CREATE TRIGGER IF NOT EXISTS specs_no_update
BEFORE UPDATE ON specs
BEGIN
    SELECT RAISE(ABORT, 'specs is append-only');
END;
CREATE TRIGGER IF NOT EXISTS specs_no_delete
BEFORE DELETE ON specs
BEGIN
    SELECT RAISE(ABORT, 'specs is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trials_no_update
BEFORE UPDATE ON trials
BEGIN
    SELECT RAISE(ABORT, 'trials is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trials_no_delete
BEFORE DELETE ON trials
BEGIN
    SELECT RAISE(ABORT, 'trials is append-only');
END;
CREATE TRIGGER IF NOT EXISTS reflections_no_update
BEFORE UPDATE ON reflections
BEGIN
    SELECT RAISE(ABORT, 'reflections is append-only');
END;
CREATE TRIGGER IF NOT EXISTS reflections_no_delete
BEFORE DELETE ON reflections
BEGIN
    SELECT RAISE(ABORT, 'reflections is append-only');
END;
"""


class ExperimentLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register_spec(self, spec: ExperimentSpec, created_at: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO specs(spec_id, spec_hash, spec_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (spec.experiment_id, spec.spec_hash, canonical_json(spec.as_dict()), created_at),
            )
            return cursor.rowcount == 1

    def append_trial(self, result: TrialResult) -> bool:
        try:
            self.append_many([result])
        except sqlite3.IntegrityError:
            if self.find_by_hash(result.experiment_hash) is not None:
                return False
            raise
        return True

    def append_many(self, results: Iterable[TrialResult]) -> int:
        rows = list(results)
        if not rows:
            return 0
        with self.transaction() as connection:
            for result in rows:
                payload = result.ledger_payload()
                columns = list(payload)
                placeholders = ", ".join("?" for _ in columns)
                connection.execute(
                    f"INSERT INTO trials ({', '.join(columns)}) VALUES ({placeholders})",
                    [payload[column] for column in columns],
                )
        return len(rows)

    def find_by_hash(self, experiment_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trials WHERE experiment_hash = ?",
                (experiment_hash,),
            ).fetchone()
        return dict(row) if row else None

    def all_trials(self, spec_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM trials"
        arguments: tuple[Any, ...] = ()
        if spec_id is not None:
            query += " WHERE spec_id = ?"
            arguments = (spec_id,)
        query += " ORDER BY strategy, sequence_no, created_at"
        with self._connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return [self._decode_trial(row) for row in rows]

    def strategy_history(
        self,
        spec_id: str,
        strategy: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM trials
                WHERE spec_id = ? AND strategy = ?
                ORDER BY sequence_no, created_at
                """,
                (spec_id, strategy),
            ).fetchall()
        return [self._decode_trial(row) for row in rows]

    def append_reflection(self, reflection: dict[str, Any]) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO reflections(
                    reflection_id, trial_id, spec_id, strategy, candidate_id,
                    categories_json, positive_experience, negative_experience, evidence_json,
                    recommendations_json, memory_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reflection["reflection_id"],
                    reflection["trial_id"],
                    reflection["spec_id"],
                    reflection["strategy"],
                    reflection["candidate_id"],
                    canonical_json(reflection["bottleneck_categories"]),
                    int(reflection["positive_experience"]),
                    int(reflection["negative_experience"]),
                    canonical_json(reflection["evidence"]),
                    canonical_json(reflection["recommended_candidate_ids"]),
                    reflection["memory_summary"],
                    reflection["created_at"],
                ),
            )
            return cursor.rowcount == 1

    def strategy_memory(
        self,
        spec_id: str,
        strategy: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM reflections
                WHERE spec_id = ? AND strategy = ?
                ORDER BY created_at
                """,
                (spec_id, strategy),
            ).fetchall()
        output = []
        for row in rows:
            payload = dict(row)
            payload["bottleneck_categories"] = json.loads(
                payload.pop("categories_json")
            )
            payload["evidence"] = json.loads(payload.pop("evidence_json"))
            payload["recommended_candidate_ids"] = json.loads(
                payload.pop("recommendations_json")
            )
            payload["negative_experience"] = bool(payload["negative_experience"])
            payload["positive_experience"] = bool(
                payload.get("positive_experience", False)
            )
            output.append(payload)
        return output

    def all_reflections(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reflections ORDER BY created_at"
            ).fetchall()
        output = []
        for row in rows:
            payload = dict(row)
            payload["bottleneck_categories"] = json.loads(
                payload.pop("categories_json")
            )
            payload["evidence"] = json.loads(payload.pop("evidence_json"))
            payload["recommended_candidate_ids"] = json.loads(
                payload.pop("recommendations_json")
            )
            payload["negative_experience"] = bool(payload["negative_experience"])
            payload["positive_experience"] = bool(
                payload.get("positive_experience", False)
            )
            output.append(payload)
        return output

    def export_jsonl(self, destination: Path, spec_id: str | None = None) -> int:
        rows = self.all_trials(spec_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    @staticmethod
    def _decode_trial(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        for key in (
            "candidate_json",
            "quality_json",
            "performance_json",
            "baseline_delta_json",
        ):
            payload[key.removesuffix("_json")] = json.loads(payload.pop(key))
        payload["quality_passed"] = bool(payload["quality_passed"])
        for key in ("runtime_fingerprint", "hardware_fingerprint"):
            payload[key] = json.loads(payload[key])
        return payload
