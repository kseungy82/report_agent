import os
import sys
import json
import logging
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
# 내가 만든 모듈들 불러오기
from config import WORKSPACE_DIR, MODEL_ID, CompareMode, API_HOST, API_PORT
from utils import auto_slice_financials
from classes import UpstageDocumentParseClient, LLMClient, MetricSelector, LLMTableExtractor, GrowthReasoner
from nodes import build_graph, render_report

#로그 파일 설정
os.chdir(WORKSPACE_DIR)
current_path = os.getcwd()
log_filename = os.path.join(current_path, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True, # 보고서 별로 로그 파일 따로 기록
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # 파일 저장
        logging.StreamHandler(sys.stdout)                  # 터미널 동시 출력
    ]
)


def run_pipeline(pdf_paths: List[str], llm: LLMClient, upstage: UpstageDocumentParseClient, compare: CompareMode = "YoY", top_k: int = 5, use_reasoning: bool = False):
    app = build_graph(MetricSelector(llm), LLMTableExtractor(llm), upstage, GrowthReasoner(llm) if use_reasoning else None)
    final_state = app.invoke({"pdf_paths": pdf_paths, "compare": compare, "top_k": top_k, "warnings": []})
    return final_state, render_report(final_state)

# FASTAPI 백엔드 서버 설정
app = FastAPI(title="재무 분석 에이전트 API", description="PDF를 분석하여 재무 추이 보고서를 반환하는 하위 에이전트")
  
upstage_client = None
llm_client = None

@app.on_event("startup")
def startup_event():
    global upstage_client, llm_client
    logging.info("서버 시작 중: API 클라이언트와 LLM 모델을 메모리에 로드합니다...")
    upstage_client = UpstageDocumentParseClient()
    llm_client = LLMClient(model_id=MODEL_ID)
    logging.info("모델 로드 완료! 요청 대기 중...")

# 상위 에이전트가 보낼 요청(Request) 데이터 양식
class AnalyzeRequest(BaseModel):
    pdf_path: str
    compare: CompareMode = "YoY"
    top_k: int = 5
    use_reasoning: bool = True

# API 엔드포인트 (상위 에이전트가 이 주소로 접근함)
@app.post("/analyze")
def analyze_report(req: AnalyzeRequest):
    try:
        if not os.path.exists(req.pdf_path):
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {req.pdf_path}")

        sliced_pdf = "./temp_sliced_finance.pdf" 
        
        logging.info(f"상위 에이전트로부터 분석 요청 수신: {req.pdf_path}")
        auto_slice_financials(req.pdf_path, sliced_pdf)

        report_state, text = run_pipeline(
            pdf_paths=[sliced_pdf], 
            llm=llm_client,
            upstage=upstage_client,
            compare=req.compare,
            top_k=req.top_k,
            use_reasoning=req.use_reasoning
        )

        logging.info("분석 완료. 상위 에이전트로 결과를 반환합니다.")
        
        # 상위 에이전트에게 전달될 JSON 응답 (최종 텍스트 + 필요시 raw 데이터 포함)
        return {
            "status": "success",
            "report_text": text,
            "warnings": report_state.get("warnings", [])
        }

    except Exception as e:
        logging.error(f"분석 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # config.py에 설정된 포트 번호로 서버를 실행합니다.
    logging.info(f"FastAPI 서버를 {API_HOST}:{API_PORT} 에서 시작합니다.")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False)
