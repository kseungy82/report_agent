import os
import sys
import subprocess
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "financial_agent.db"
LOG_DIR  = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"
REQ_FILE = BASE_DIR / "requirements.txt"


def install_packages():
    if not REQ_FILE.exists():
        log.warning(f"requirements.txt 를 찾을 수 없습니다: {REQ_FILE}")
        return
    log.info("=" * 50)
    log.info("Step 1/5  패키지 설치 시작")
    log.info("=" * 50)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)],
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("패키지 설치 실패. 위 오류 메시지를 확인하세요.")
        sys.exit(1)
    log.info("패키지 설치 완료 ✓")


def check_env():
    log.info("=" * 50)
    log.info("Step 2/5  API 키 / 환경 변수 검사")
    log.info("=" * 50)
    required = {
        "UPSTAGE_API_KEY":          "Upstage Document Parse (OCR)",
        "HUGGINGFACEHUB_API_TOKEN": "Hugging Face 모델 다운로드",
    }
    # ── BUGFIX: config.py 를 import하지 않고 환경변수를 직접 읽습니다.
    # config.py import 시 placeholder 값으로 os.environ 을 덮어쓰는 부작용 방지.
    missing = []
    for key, desc in required.items():
        val = os.environ.get(key, "")
        if not val or val.startswith("여기에"):
            log.warning(f"  ✗  {key}  ({desc})  — 설정되지 않았습니다.")
            missing.append(key)
        else:
            log.info(f"  ✓  {key}  ({desc})")

    if missing:
        log.warning(
            "\n[주의] config.py 의 API 키를 실제 값으로 교체하거나 "
            "환경 변수로 직접 설정한 뒤 다시 실행하세요.\n"
            "  예) export UPSTAGE_API_KEY='sk-...'\n"
        )
    else:
        log.info("모든 API 키 확인 완료 ✓")


def create_directories():
    log.info("=" * 50)
    log.info("Step 3/5  작업 디렉토리 생성")
    log.info("=" * 50)
    for d in [LOG_DIR, DATA_DIR, TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        log.info(f"  ✓  {d}")
    log.info("디렉토리 생성 완료 ✓")


def init_database():
    """
    SQLite DB 및 테이블 초기화.
    database.py 를 import하지 않고 직접 스키마를 생성합니다.
    (순환 import 및 모듈레벨 부작용 방지)
    """
    log.info("=" * 50)
    log.info("Step 4/5  SQLite DB 초기화")
    log.info("=" * 50)
    log.info(f"  DB 경로: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))

    # database.py 의 _create_tables 와 완전히 동일한 스키마
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
            pdf_path     TEXT    NOT NULL,
            compare_mode TEXT    NOT NULL,
            top_k        INTEGER NOT NULL DEFAULT 5,
            use_reasoning INTEGER NOT NULL DEFAULT 1,
            status       TEXT    NOT NULL DEFAULT 'pending',
            error_msg    TEXT
        );

        CREATE TABLE IF NOT EXISTS doc_bundles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id   INTEGER NOT NULL REFERENCES reports(id),
            pdf_path    TEXT    NOT NULL,
            bundle_json TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS metric_selections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id   INTEGER NOT NULL REFERENCES reports(id),
            metrics_json TEXT   NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        );

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
        );

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
        );

        CREATE TABLE IF NOT EXISTS warnings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id  INTEGER NOT NULL REFERENCES reports(id),
            stage      TEXT,
            message    TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        );
    """)

    conn.commit()
    conn.close()

    for t in ["reports", "doc_bundles", "metric_selections", "llm_extractions", "analysis_results", "warnings"]:
        log.info(f"  ✓  {t}")
    log.info("DB 초기화 완료 ✓")


def prefetch_model():
    log.info("=" * 50)
    log.info("Step 5/5  Hugging Face 모델 사전 다운로드 (선택)")
    log.info("=" * 50)
    answer = input(
        "  모델을 지금 미리 다운로드하시겠습니까? (첫 실행 시 수 분 소요)\n"
        "  [y/N] > "
    ).strip().lower()

    if answer != "y":
        log.info("  → 건너뜀. 첫 API 호출 시 자동으로 다운로드됩니다.")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        sys.path.insert(0, str(BASE_DIR))
        from config import MODEL_ID
        log.info(f"  다운로드 중: {MODEL_ID}")
        AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
        log.info("  모델 다운로드 완료 ✓")
    except Exception as e:
        log.warning(f"  모델 다운로드 실패 (나중에 수동으로 가능): {e}")


def main():
    log.info("")
    log.info(f"  Base directory : {BASE_DIR}")
    log.info(f"  Python         : {sys.version}")
    log.info("")

    install_packages()
    check_env()
    create_directories()
    init_database()
    prefetch_model()

    log.info("")
    log.info(f"  DB 경로   : {DB_PATH}")
    log.info(f"  데이터    : {DATA_DIR}  ← PDF 파일을 여기에 넣으세요")
    log.info(f"  로그      : {LOG_DIR}")
    log.info("")


if __name__ == "__main__":
    main()
