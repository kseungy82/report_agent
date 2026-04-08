import os
import sys
import json
import logging
from datetime import datetime

# 내가 만든 모듈들 불러오기
from config import WORKSPACE_DIR, MODEL_ID
from utils import auto_slice_financials
from classes import UpstageDocumentParseClient, LLMClient, MetricSelector, LLMTableExtractor, GrowthReasoner
from nodes import build_graph, render_report

#로그 파일 설정
current_path = os.getcwd()
log_filename = os.path.join(current_path, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
print(f"로그 파일이 다음 경로에 생성됩니다: {log_filename}")
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
  
if __name__ == "__main__":
    upstage = UpstageDocumentParseClient()
    llm = LLMClient(model_id="kakaocorp/kanana-1.5-2.1b-instruct-2505")
    # 파일 경로 설정 
    original_pdf = "./[폴라리스오피스]분기보고서(2025.11.14).pdf"
    sliced_pdf = "./temp_sliced_finance.pdf" # 잘라낸 파일 임시 저장

    # 자동 스캔 및 자르기 실행
    logging.info(f"{original_pdf}분석\n")
    logging.info("PDF 자동 분리를 시작합니다...")
    auto_slice_financials(original_pdf, sliced_pdf)

    # 파이프라인 실행 
    report_state, text = run_pipeline(
        pdf_paths=[sliced_pdf], # 자른 파일을 투입
        llm=llm,
        upstage=upstage,
        compare="YoY",
        top_k=5,
        use_reasoning=True
    )

    # 결과 출력
    logging.info("\n[LLM 추출 Raw Data]")
    logging.info(json.dumps(report_state["llm_extraction_by_pdf"], indent=2, default=lambda x: x.__dict__, ensure_ascii=False))

    logging.info("\n" + "="*60)
    logging.info("\n[최종 생성된 분석 보고서]")
    logging.info(text)
