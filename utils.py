import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter
from config import Unit, CompareMode, PeriodKey
import logging

class MetricSelectionOutput(BaseModel):
    selected_metrics: List[str] = Field(description="제공된 보기 중 이 기업 분석에 가장 적합한 핵심 지표 5~10개")

class LLMExtractionOutput(BaseModel):
    unit_raw: str = Field(description="문서에 기록된 원본 단위")
    now_period: str = Field(description="당기 종료일 문자열")
    ref_period: str = Field(description="전기 종료일 문자열")
    now_period_key: str = Field(description="당기 PeriodKey (예: 2025Q1)")
    ref_period_key: str = Field(description="전기 PeriodKey (예: 2024Q1)")
    items: Dict[str, Dict[str, Union[str, int, float, None]]] = Field(
        description="PeriodKey별 지표 데이터"
    )

class NormalizedFinancials(BaseModel):
    unit: Unit
    periods: List[PeriodKey]
    pl: Dict[str, List[float]]

class TopMove(BaseModel):
    metric: str
    now_period: PeriodKey
    ref_period: PeriodKey
    now: float
    ref: float
    delta: float
    delta_pct: Optional[float] = None
    flip_flag: bool = False

class GrowthReasoningInput(BaseModel):
    unit: Unit
    periods: List[PeriodKey]
    metrics_timeseries: Dict[str, List[float]]
    top_moves: List[TopMove]
    compare: CompareMode
    now_period: PeriodKey
    ref_period: Optional[PeriodKey]

class GrowthReasoningOutput(BaseModel):
    growth_trajectory: str
    key_changes: List[str]
    caveats: List[str]
    summary_table: Optional[str]


def sort_period_keys(keys: List[PeriodKey]) -> List[PeriodKey]:
    def kf(k: str) -> Tuple[int, int]:
        m = re.match(r"(\d{4})Q([1-4])", k)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return sorted(keys, key=kf)


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


def unit_multiplier_to_million(u: Unit) -> float:
    mapping = {"백만원": 1.0, "천원": 1e-3, "원": 1e-6, "천만원": 10.0, "억원": 100.0}
    return mapping.get(u, 1.0)


def get_quarter(month):
    month = int(month)
    if 1 <= month <= 3:  return "Q1"
    elif 4 <= month <= 6:  return "Q2"
    elif 7 <= month <= 9:  return "Q3"
    elif 10 <= month <= 12: return "Q4"
    return ""


def extract_financial_periods(doc_text_or_bundle):
    """
    문서에서 당기/전기 기간 정보를 추출합니다.
    항상 6-tuple (now_date, now_key, now_term, ref_date, ref_key, ref_term)을 반환합니다.

    BUGFIX: 기존 코드는 날짜 패턴 미발견 시 암묵적으로 None을 반환하여
    classes.py L132 의 6-tuple 언패킹에서 TypeError 크래시가 발생했습니다.
    이제 fallback 튜플을 반환하여 파이프라인이 계속 진행됩니다.
    """
    _fallback = ("추출불가", "9999Q1", "당기", "추출불가", "9998Q1", "전기")

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

    if not matches_full:
        logging.warning("[extract_financial_periods] 날짜 패턴을 찾지 못했습니다. fallback 값을 사용합니다.")
        return _fallback

    extracted_dates = []
    for match in matches_full:
        term_str, start_y, start_m, start_d, end_y, end_m, end_d = match
        clean_term    = term_str.strip() if term_str else ""
        end_date_str  = f"{end_y}.{int(end_m):02d}.{int(end_d):02d}"
        quarter       = get_quarter(end_m)
        period_key    = f"{end_y}{quarter}"
        if not any(d[0] == end_date_str for d in extracted_dates):
            extracted_dates.append((end_date_str, period_key, clean_term))

    extracted_dates.sort(reverse=True, key=lambda x: x[0])

    if len(extracted_dates) >= 2:
        return (
            extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2],
            extracted_dates[1][0], extracted_dates[1][1], extracted_dates[1][2],
        )
    elif len(extracted_dates) == 1:
        prev_year = str(int(extracted_dates[0][1][:4]) - 1)
        prev_key  = f"{prev_year}{extracted_dates[0][1][4:]}"
        return (
            extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2],
            "추출불가", prev_key, "전기",
        )
    else:
        logging.warning("[extract_financial_periods] 유효 날짜 없음. fallback 반환.")
        return _fallback


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
