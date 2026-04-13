from __future__ import annotations

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def check_and_download_model():
    model_path = os.getenv("KANANA_MODEL_PATH")
    model_id = os.getenv("KANANA_MODEL_ID", "kakaocorp/kanana-1.5-2.1b-instruct-2505")

    if not model_path:
        logging.info(".env에 KANANA_MODEL_PATH가 설정되지 않았습니다.")
        return False

    path = Path(model_path)

    if path.exists() and any(path.iterdir()):
        logging.info(f"모델이 이미 존재합니다: {model_path}")
        return True

    print(f"모델 다운로드 시작: {model_id} → {model_path}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path.mkdir(parents=True, exist_ok=True)
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(model_path)
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).save_pretrained(model_path)
        logging.info(f"모델 다운로드 완료: {model_path}")
        return True
    except Exception as e:
        logging.info(f"모델 다운로드 실패: {e}")
        return False


def check_upstage_key():
    key = os.getenv("UPSTAGE_API_KEY")
    if not key:
        logging.info("UPSTAGE_API_KEY가 설정되지 않았습니다.")
        return False
    logging.info("UPSTAGE_API_KEY 확인 완료")
    return True


if __name__ == "__main__":
    logging.info("=== 사전 준비 체크 시작 ===\n")
    ok1 = check_upstage_key()
    ok2 = check_and_download_model()

    if ok1 and ok2:
        logging.info("\n모든 준비 완료. 서버를 시작할 수 있습니다.")
    else:
        logging.info("\n준비가 완료되지 않았습니다. .env 파일을 확인하세요.")
