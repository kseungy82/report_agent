import os
import sys
import json
import logging
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

from config import WORKSPACE_DIR, MODEL_ID, CompareMode, API_HOST, API_PORT
from utils import auto_slice_financials
from classes import UpstageDocumentParseClient, LLMClient, MetricSelector, LLMTableExtractor, GrowthReasoner
from nodes import build_graph, render_report
from database import DBClient

# ────────────────────────────────────────────
# 로그 설정
# ────────────────────────────────────────────
LOG_DIR = os.path.join(WORKSPACE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True,
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


# ────────────────────────────────────────────
# 파이프라인 실행 함수
# ────────────────────────────────────────────
def run_pipeline(
    pdf_paths: List[str],
    llm: LLMClient,
    upstage: UpstageDocumentParseClient,
    compare: CompareMode = "YoY",
    top_k: int = 5,
    use_reasoning: bool = False,
    db: DBClient = None,
    report_id: int = None,
):
    app = build_graph(
        MetricSelector(llm),
        LLMTableExtractor(llm),
        upstage,
        GrowthReasoner(llm) if use_reasoning else None
    )

    final_state = app.invoke({
        "pdf_paths": pdf_paths,
        "compare": compare,
        "top_k": top_k,
        "warnings": []
    })

    report_text = render_report(final_state)

    # ── DB 저장 ──────────────────────────────
    if db and report_id:
        try:
            for path, bundle in final_state.get("doc_bundle_by_pdf", {}).items():
                db.save_doc_bundle(report_id, path, bundle)

            if final_state.get("selected_metrics"):
                db.save_metric_selection(report_id, final_state["selected_metrics"])

            for path, extraction in final_state.get("llm_extraction_by_pdf", {}).items():
                db.save_llm_extraction(report_id, path, extraction)

            db.save_analysis_result(report_id, final_state, report_text)
            db.save_warnings(report_id, final_state.get("warnings", []))

        except Exception as e:
            logging.error(f"[DB] 저장 중 오류 발생 (분석 결과에는 영향 없음): {e}")

    return final_state, report_text


# ────────────────────────────────────────────
# FastAPI 앱
# ────────────────────────────────────────────
app = FastAPI(
    title="재무 분석 에이전트 API",
    description="PDF를 분석하여 재무 추이 보고서를 반환하는 하위 에이전트"
)

upstage_client = None
llm_client     = None
db_client      = None


@app.on_event("startup")
def startup_event():
    global upstage_client, llm_client, db_client
    logging.info("서버 시작 중: API 클라이언트와 LLM 모델을 메모리에 로드합니다...")
    upstage_client = UpstageDocumentParseClient()
    llm_client     = LLMClient(model_id=MODEL_ID)
    db_client      = DBClient()
    logging.info("모델 로드 완료! 요청 대기 중...")


# ────────────────────────────────────────────
# 요청 스키마
# ────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    pdf_path:      str
    compare:       CompareMode = "YoY"
    top_k:         int         = 5
    use_reasoning: bool        = True


# ────────────────────────────────────────────
# 분석 엔드포인트
# ────────────────────────────────────────────
@app.post("/analyze")
def analyze_report(req: AnalyzeRequest):
    # ── BUGFIX: report_id를 None으로 먼저 초기화하여 except 블록에서 UnboundLocalError 방지
    report_id = None

    try:
        if not os.path.exists(req.pdf_path):
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {req.pdf_path}")

        report_id = db_client.create_report(
            pdf_path      = req.pdf_path,
            compare_mode  = req.compare,
            top_k         = req.top_k,
            use_reasoning = req.use_reasoning,
        )

        temp_dir   = os.path.join(WORKSPACE_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        sliced_pdf = os.path.join(temp_dir, "temp_sliced_finance.pdf")

        logging.info(f"상위 에이전트로부터 분석 요청 수신: {req.pdf_path}")
        auto_slice_financials(req.pdf_path, sliced_pdf)

        report_state, text = run_pipeline(
            pdf_paths     = [sliced_pdf],
            llm           = llm_client,
            upstage       = upstage_client,
            compare       = req.compare,
            top_k         = req.top_k,
            use_reasoning = req.use_reasoning,
            db            = db_client,
            report_id     = report_id,
        )

        db_client.finish_report(report_id, status="success")
        logging.info(f"분석 완료 (report_id={report_id}). 상위 에이전트로 결과를 반환합니다.")

        return {
            "status":      "success",
            "report_id":   report_id,
            "report_text": text,
            "warnings":    report_state.get("warnings", [])
        }

    except HTTPException:
        raise

    except Exception as e:
        logging.error(f"분석 중 에러 발생: {e}")
        # ── BUGFIX: report_id가 None이 아닐 때만 DB 업데이트 (locals()/dir() 방식 제거)
        if report_id is not None:
            try:
                db_client.finish_report(report_id, status="error", error_msg=str(e))
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────
# 조회 엔드포인트 (상위 에이전트용)
# ────────────────────────────────────────────
@app.get("/reports")
def list_reports(limit: int = 20):
    """최근 분석 목록 반환"""
    return db_client.list_reports(limit=limit)


@app.get("/reports/{report_id}")
def get_report(report_id: int):
    """특정 분석 결과 반환"""
    result = db_client.get_analysis_result(report_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"report_id={report_id} 를 찾을 수 없습니다.")
    return result


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
if __name__ == "__main__":
    logging.info(f"FastAPI 서버를 {API_HOST}:{API_PORT} 에서 시작합니다.")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False)
