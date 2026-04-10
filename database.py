import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parent / "financial_agent.db"


def _create_tables(conn: sqlite3.Connection):
    """
    DB 테이블 스키마 정의.
    setup.py 와 database.py 양쪽에서 공유하는 단일 정의입니다.
    setup.py 에서 init_database()를 import 없이 직접 이 함수를 호출합니다.
    """
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
            pdf_path     TEXT    NOT NULL,
            compare_mode TEXT    NOT NULL,
            top_k        INTEGER NOT NULL DEFAULT 5,
            use_reasoning INTEGER NOT NULL DEFAULT 1,
            status       TEXT    NOT NULL DEFAULT 'pending',
            error_msg    TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_bundles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id   INTEGER NOT NULL REFERENCES reports(id),
            pdf_path    TEXT    NOT NULL,
            bundle_json TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS metric_selections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id   INTEGER NOT NULL REFERENCES reports(id),
            metrics_json TEXT   NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_extractions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id       INTEGER NOT NULL REFERENCES reports(id),
            pdf_path        TEXT    NOT NULL,
            unit_raw        TEXT,
            now_period      TEXT,
            ref_period      TEXT,
            now_period_key  TEXT,
            ref_period_key  TEXT,
            items_json      TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id         INTEGER NOT NULL REFERENCES reports(id),
            now_period        TEXT,
            ref_period        TEXT,
            compare_mode      TEXT,
            unit              TEXT,
            top_moves_json    TEXT,
            growth_trajectory TEXT,
            key_changes_json  TEXT,
            caveats_json      TEXT,
            summary_table     TEXT,
            report_text       TEXT,
            created_at        TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS warnings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id  INTEGER NOT NULL REFERENCES reports(id),
            stage      TEXT,
            message    TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.commit()


class DBClient:
    """모든 DB 작업을 담당하는 단일 클라이언트."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or _DEFAULT_DB)
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_db(self):
        """
        DB 파일과 테이블이 없으면 자동 생성합니다.
        setup.py 를 import하지 않고 _create_tables()를 직접 호출합니다.
        (setup.py import 시 logging.basicConfig 등 모듈레벨 코드 재실행 방지)
        """
        if not Path(self.db_path).exists():
            log.warning(
                f"DB 파일이 없습니다. 자동 생성합니다: {self.db_path}\n"
                "  정식 환경 구성은 'python setup.py' 를 실행하세요."
            )
        with self._connect() as conn:
            _create_tables(conn)

    def create_report(self, pdf_path: str, compare_mode: str, top_k: int = 5, use_reasoning: bool = True) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reports (pdf_path, compare_mode, top_k, use_reasoning, status) VALUES (?, ?, ?, ?, 'running')",
                (pdf_path, compare_mode, top_k, int(use_reasoning)),
            )
            rid = cur.lastrowid
        log.info(f"[DB] report 생성: id={rid}, pdf={pdf_path}")
        return rid

    def finish_report(self, report_id: int, status: str = "success", error_msg: str = ""):
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET status=?, error_msg=? WHERE id=?",
                (status, error_msg or None, report_id),
            )
        log.info(f"[DB] report 완료: id={report_id}, status={status}")

    def save_doc_bundle(self, report_id: int, pdf_path: str, bundle: Dict[str, Any]):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO doc_bundles (report_id, pdf_path, bundle_json) VALUES (?, ?, ?)",
                (report_id, pdf_path, json.dumps(bundle, ensure_ascii=False)),
            )
        log.info(f"[DB] doc_bundle 저장: report_id={report_id}")

    def load_doc_bundle(self, report_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pdf_path, bundle_json FROM doc_bundles WHERE report_id=?", (report_id,)
            ).fetchall()
        return [{"pdf_path": r["pdf_path"], **json.loads(r["bundle_json"])} for r in rows]

    def save_metric_selection(self, report_id: int, metrics: List[str]):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metric_selections (report_id, metrics_json) VALUES (?, ?)",
                (report_id, json.dumps(metrics, ensure_ascii=False)),
            )
        log.info(f"[DB] metric_selection 저장: {metrics}")

    def save_llm_extraction(self, report_id: int, pdf_path: str, extraction):
        d = extraction.model_dump() if hasattr(extraction, "model_dump") else dict(extraction)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO llm_extractions
                    (report_id, pdf_path, unit_raw, now_period, ref_period,
                     now_period_key, ref_period_key, items_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_id, pdf_path,
                    d.get("unit_raw"), d.get("now_period"), d.get("ref_period"),
                    d.get("now_period_key"), d.get("ref_period_key"),
                    json.dumps(d.get("items", {}), ensure_ascii=False),
                ),
            )
        log.info(f"[DB] llm_extraction 저장: report_id={report_id}")

    def save_analysis_result(self, report_id: int, final_state: Dict[str, Any], report_text: str):
        top_moves = final_state.get("top_moves", [])
        top_moves_s = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in top_moves]

        reasoning = final_state.get("llm_reasoning")
        growth_trajectory = key_changes = caveats = summary_table = None
        if reasoning:
            rd = reasoning.model_dump() if hasattr(reasoning, "model_dump") else dict(reasoning)
            growth_trajectory = rd.get("growth_trajectory")
            key_changes       = json.dumps(rd.get("key_changes", []), ensure_ascii=False)
            caveats           = json.dumps(rd.get("caveats", []),      ensure_ascii=False)
            summary_table     = rd.get("summary_table")

        fin  = final_state.get("fin")
        unit = fin.unit if fin else None

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO analysis_results
                    (report_id, now_period, ref_period, compare_mode, unit,
                     top_moves_json, growth_trajectory, key_changes_json,
                     caveats_json, summary_table, report_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_id,
                    final_state.get("now_period"), final_state.get("ref_period"),
                    final_state.get("compare"), unit,
                    json.dumps(top_moves_s, ensure_ascii=False),
                    growth_trajectory, key_changes, caveats, summary_table, report_text,
                ),
            )
        log.info(f"[DB] analysis_result 저장: report_id={report_id}")

    def save_warnings(self, report_id: int, warnings: List[str], stage: str = "pipeline"):
        if not warnings:
            return
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO warnings (report_id, stage, message) VALUES (?, ?, ?)",
                [(report_id, stage, w) for w in warnings],
            )
        log.info(f"[DB] warnings 저장: {len(warnings)}건")

    def get_report(self, report_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return dict(row) if row else None

    def list_reports(self, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_analysis_result(self, report_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_results WHERE report_id=? ORDER BY id DESC LIMIT 1",
                (report_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("key_changes_json", "caveats_json"):
            if d.get(field):
                d[field.replace("_json", "")] = json.loads(d[field])
        return d
