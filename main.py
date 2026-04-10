from __future__ import annotations

from datetime import datetime
import logging
import os
import sys
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from report_agent import analyze_pdf
from router import RouterAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CompareMode = Literal["QoQ", "YoY"]

app = FastAPI(title="Kanana Report Agent API", version="0.1.0")
router_agent = RouterAgent()


def _configure_request_file_logging(base_dir: str) -> str:
    """
    노트북과 동일한 방식으로 요청별 로그 파일을 생성합니다.
    """
    log_filename = os.path.join(base_dir, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("로그 파일이 다음 경로에 생성됩니다: %s", log_filename)
    return log_filename


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    pdf: UploadFile = File(...),
    compare: CompareMode = Form("YoY"),
    top_k: int = Form(5),
    use_reasoning: bool = Form(True),
    slice_financial_statement: bool = Form(True),
):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    try:
        # 노트북과 동일하게 프로젝트 루트에 요청별 로그 파일 생성
        log_path = _configure_request_file_logging(os.getcwd())
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, pdf.filename)
            content = await pdf.read()
            with open(input_path, "wb") as f:
                f.write(content)

            state, text, effective_pdf_path = analyze_pdf(
                pdf_path=input_path,
                compare=compare,
                top_k=int(top_k),
                use_reasoning=bool(use_reasoning),
                slice_financial_statement=bool(slice_financial_statement),
                work_dir=td,
            )
            logging.info("\n" + "=" * 60)
            logging.info("\n[최종 생성된 분석 보고서]")
            logging.info(text)

            return JSONResponse(
                {
                    "report_text": text,
                    "effective_pdf": os.path.basename(effective_pdf_path),
                    "state": state,
                    "log_file": log_path,
                }
            )
    except Exception as e:
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route")
async def route(
    task: str = Form(...),
    pdf: UploadFile = File(...),
    compare: CompareMode = Form("YoY"),
    top_k: int = Form(5),
    use_reasoning: bool = Form(True),
    slice_financial_statement: bool = Form(True),
):
    """
    상위 에이전트 라우팅 엔드포인트.
    현재는 라우팅 결과가 report이면, 하위 report agent를 실행합니다.
    """
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    try:
        # 노트북과 동일하게 프로젝트 루트에 요청별 로그 파일 생성
        log_path = _configure_request_file_logging(os.getcwd())
        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, pdf.filename)
            content = await pdf.read()
            with open(input_path, "wb") as f:
                f.write(content)

            routed = router_agent.route(
                task=task,
                pdf_path=input_path,
                compare=compare,
                top_k=int(top_k),
                use_reasoning=bool(use_reasoning),
                slice_financial_statement=bool(slice_financial_statement),
                work_dir=td,
            )
            logging.info("\n" + "=" * 60)
            logging.info("\n[최종 생성된 분석 보고서]")
            logging.info(routed["report_text"])
            return JSONResponse(
                {
                    "route": routed["route"],
                    "report_text": routed["report_text"],
                    "effective_pdf": os.path.basename(routed["effective_pdf"]),
                    "state": routed["state"],
                    "log_file": log_path,
                }
            )
    except Exception as e:
        logger.exception("route failed")
        raise HTTPException(status_code=500, detail=str(e))

