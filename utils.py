import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter
from config import Unit, CompareMode, PeriodKey
import logging

#LLM이 기업 분석에 적합하다고 판단한 기업의 핵심 지표들
class MetricSelectionOutput(BaseModel):
    selected_metrics: List[str] = Field(description="제공된 보기 중 이 기업 분석에 가장 적합한 핵심 지표 5~10개")

#LLM이 추출한 보고서 기본 정보
class LLMExtractionOutput(BaseModel):
    unit_raw: str = Field(description="문서에 기록된 원본 단위")
    now_period: str = Field(description="당기 종료일 문자열")
    ref_period: str = Field(description="전기 종료일 문자열")
    now_period_key: str = Field(description="당기 PeriodKey (예: 2025Q1)")
    ref_period_key: str = Field(description="전기 PeriodKey (예: 2024Q1)")

    items: Dict[str, Dict[str, Union[str, int, float, None]]] = Field(
        description="PeriodKey별 지표 데이터"
    )
#단위 '백만원'으로 통일 후 시계열로 정렬한 데이터
class NormalizedFinancials(BaseModel):
    unit: Unit
    periods: List[PeriodKey] 
    pl: Dict[str, List[float]]  

#지표 변화 요약 (전년 또는 전분기 대비 증감과 증감률 계산 결과 저장)
class TopMove(BaseModel):
    metric: str
    now_period: PeriodKey
    ref_period: PeriodKey
    now: float   #이번
    ref: float   #비교
    delta: float #변화량
    delta_pct: Optional[float] = None  #변화율
    flip_flag: bool = False  #흑자,적자 전환 여부

#GrowthReasoningOutput에 보낼 내용 정리
class GrowthReasoningInput(BaseModel):
    unit: Unit   #화폐 단위
    periods: List[PeriodKey]
    metrics_timeseries: Dict[str, List[float]]  
    top_moves: List[TopMove]
    compare: CompareMode   #분기별/전년 동기 (QoQ/YoY)
    now_period: PeriodKey
    ref_period: Optional[PeriodKey]

class GrowthReasoningOutput(BaseModel):
    growth_trajectory: str
    key_changes: List[str]
    caveats: List[str]  
    summary_table: Optional[str]

# 시간순 정렬
def sort_period_keys(keys: List[PeriodKey]) -> List[PeriodKey]:
    def kf(k: str) -> Tuple[int, int]:
        m = re.match(r"(\d{4})Q([1-4])", k)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0) #정규화된 형식(YYYYQn) 아닌 경우 (0,0)반환
    return sorted(keys, key=kf)

#직전 분기를 reference period로 추출
def pick_reference_index(periods: List[PeriodKey], idx_now: int, compare: CompareMode) -> Optional[int]:
    if idx_now < 0 or idx_now >= len(periods):
        return None
    m = re.match(r"(\d{4})Q([1-4])", periods[idx_now])
    if not m:
        return None
    y, q = int(m.group(1)), int(m.group(2))
    if compare == "YoY":
        target = f"{y-1}Q{q}"
    else:
        target = f"{y-1}Q4" if q == 1 else f"{y}Q{q-1}"
    try:
        idx = periods.index(target)
        return idx if idx < idx_now else None
    except ValueError:
        return None


#단위 통일 (백만원 기준)
def unit_multiplier_to_million(u: Unit) -> float:
    mapping = {"백만원": 1.0, "천원": 1e-3, "원": 1e-6, "천만원": 10.0, "억원": 100.0}
    return mapping.get(u, 1.0)

#날짜 추출
def get_quarter(month):
    month = int(month)
    if 1 <= month <= 3: return "Q1"
    elif 4 <= month <= 6: return "Q2"
    elif 7 <= month <= 9: return "Q3"
    elif 10 <= month <= 12: return "Q4"
    return ""

# 날짜 추출 (최신 Upstage 통합 HTML 완벽 대응)
def extract_financial_periods(doc_text_or_bundle):
    if isinstance(doc_text_or_bundle, dict):
        html_list = [t.get("html", "") for t in doc_text_or_bundle.get("tables", [])]
        file_name = doc_text_or_bundle.get("file", "")
        clean_text = " ".join([file_name] + html_list)
    else:
        clean_text = str(doc_text_or_bundle)

    clean_text = clean_text.replace("\\n", " ").replace("\\t", " ").replace("&nbsp;", " ")
    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)  
    clean_text = re.sub(r'\s+', ' ', clean_text)      

    pattern_full = r"(제\s*\d+\s*기\s*(?:[1-4]분기|반기|사업연도)?\s*)?(20\d{2})\s*[\.년\-\/]\s*(\d{1,2})\s*[\.월\-\/]\s*(\d{1,2})\s*[일\.]?\s*(?:부터|~|-|∼|—| )?\s*(20\d{2})\s*[\.년\-\/]\s*(\d{1,2})\s*[\.월\-\/]\s*(\d{1,2})\s*[일\.]?(?:\s*까지)?"
    matches_full = re.findall(pattern_full, clean_text)

    if matches_full:
        extracted_dates = []
        for match in matches_full:
            term_str, start_y, start_m, start_d, end_y, end_m, end_d = match
            clean_term = term_str.strip() if term_str else ""
            end_date_str = f"{end_y}.{int(end_m):02d}.{int(end_d):02d}"
            quarter = get_quarter(end_m)
            period_key = f"{end_y}{quarter}"

            # 중복 방지
            if not any(d[0] == end_date_str for d in extracted_dates):
                extracted_dates.append((end_date_str, period_key, clean_term))

        # 내림차순 정렬 (최신 날짜가 맨 앞으로 오도록)
        extracted_dates.sort(reverse=True, key=lambda x: x[0])

        if len(extracted_dates) >= 2:
            return extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2], extracted_dates[1][0], extracted_dates[1][1], extracted_dates[1][2]
        elif len(extracted_dates) == 1:
            return extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2], "추출불가", f"{int(extracted_dates[0][1][:4])-1}{extracted_dates[0][1][4:]}", "전기"
#전체 보고서 중 포괄손익계산서 부분만 추출
def auto_slice_financials(input_path, output_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    target_pages = []
    logging.info(f"문서 스캔 중... (총 {len(reader.pages)}페이지)")
    metric_keywords = [
        "매출", "매출액", "영업수익",
        "영업이익", "영업이익(손실)",
        "당기순이익", "당기순이익(손실)",
        "이자수익", "기타비용", "법인세비용", "법인세차감전순이익",
        "기타포괄손익", "배당수익", "보험손익", "보험비용", "투자손익"
    ]
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        clean_text = text.replace(" ", "").replace("\n", "")

        if "포괄손익계산서" in clean_text and "연결" not in clean_text:
            hit_count = sum(1 for k in metric_keywords if k in clean_text)
            if hit_count >= 2:
                logging.info(f"(별도)포괄손익계산서 페이지 발견: {i}")
                target_pages.append(i)
  
    if not target_pages:
        logging.info("전체 페이지 사용")
        for i in range(len(reader.pages)):
            writer.add_page(reader.pages[i])

        with open(output_path, "wb") as f:
            writer.write(f)
        return output_path

    # 첫 페이지 + 다음 페이지 추출 (표 최대 두 페이지에 작성된다고 생각)
    first = target_pages[0]
    pages = [first]
    if first + 1 < len(reader.pages):
        pages.append(first + 1)

    for p in pages:
        writer.add_page(reader.pages[p])

    with open(output_path, "wb") as f:
        writer.write(f)

    logging.info(f"필터링 완료: {pages}")
    return output_path
