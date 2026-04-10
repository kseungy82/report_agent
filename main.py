from __future__ import annotations

import logging
import os
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agents.report_agent import analyze_pdf
from agents.router import RouterAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CompareMode = Literal["QoQ", "YoY"]

app = FastAPI(title="Kanana Report Agent API", version="0.1.0")
router_agent = RouterAgent()


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

            return JSONResponse(
                {
                    "report_text": text,
                    "effective_pdf": os.path.basename(effective_pdf_path),
                    "state": state,
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
            return JSONResponse(
                {
                    "route": routed["route"],
                    "report_text": routed["report_text"],
                    "effective_pdf": os.path.basename(routed["effective_pdf"]),
                    "state": routed["state"],
                }
            )
    except Exception as e:
        logger.exception("route failed")
        raise HTTPException(status_code=500, detail=str(e))
