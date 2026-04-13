from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from classes import CompareMode, PeriodKey, Unit
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

import math
from pydantic import BaseModel

def serialize_state(obj):
    """FGState 내 Pydantic 객체 및 직렬화 불가 타입을 재귀적으로 변환"""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: serialize_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_state(i) for i in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None  # JSON은 NaN을 지원하지 않으므로 None으로 변환
    return obj

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
        target = f"{y - 1}Q{q}"
    else:
        target = f"{y - 1}Q4" if q == 1 else f"{y}Q{q - 1}"
    try:
        idx = periods.index(target)
        return idx if idx < idx_now else None
    except ValueError:
        return None


def unit_multiplier_to_million(u: Unit) -> float:
    mapping = {"백만원": 1.0, "천원": 1e-3, "원": 1e-6, "천만원": 10.0, "억원": 100.0}
    return mapping.get(u, 1.0)


def get_quarter(month: int | str) -> str:
    month_i = int(month)
    if 1 <= month_i <= 3:
        return "Q1"
    if 4 <= month_i <= 6:
        return "Q2"
    if 7 <= month_i <= 9:
        return "Q3"
    if 10 <= month_i <= 12:
        return "Q4"
    return ""


def extract_financial_periods(doc_text_or_bundle: Any):
    if isinstance(doc_text_or_bundle, dict):
        html_list = [t.get("html", "") for t in doc_text_or_bundle.get("tables", [])]
        file_name = doc_text_or_bundle.get("file", "")
        clean_text = " ".join([file_name] + html_list)
    else:
        clean_text = str(doc_text_or_bundle)

    clean_text = clean_text.replace("\\n", " ").replace("\\t", " ").replace("&nbsp;", " ")
    clean_text = re.sub(r"<[^>]+>", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    pattern_full = (
        r"(제\s*\d+\s*기\s*(?:[1-4]분기|반기|사업연도)?\s*)?"
        r"(20\d{2})\s*[\.년\-\/]\s*(\d{1,2})\s*[\.월\-\/]\s*(\d{1,2})\s*[일\.]?\s*"
        r"(?:부터|~|-|∼|—| )?\s*"
        r"(20\d{2})\s*[\.년\-\/]\s*(\d{1,2})\s*[\.월\-\/]\s*(\d{1,2})\s*[일\.]?(?:\s*까지)?"
    )
    matches_full = re.findall(pattern_full, clean_text)

    if matches_full:
        extracted_dates: list[tuple[str, str, str]] = []
        for match in matches_full:
            term_str, _start_y, _start_m, _start_d, end_y, end_m, end_d = match
            clean_term = term_str.strip() if term_str else ""
            end_date_str = f"{end_y}.{int(end_m):02d}.{int(end_d):02d}"
            quarter = get_quarter(end_m)
            period_key = f"{end_y}{quarter}"
            if not any(d[0] == end_date_str for d in extracted_dates):
                extracted_dates.append((end_date_str, period_key, clean_term))

        extracted_dates.sort(reverse=True, key=lambda x: x[0])

        if len(extracted_dates) >= 2:
            return (
                extracted_dates[0][0],
                extracted_dates[0][1],
                extracted_dates[0][2],
                extracted_dates[1][0],
                extracted_dates[1][1],
                extracted_dates[1][2],
            )
        if len(extracted_dates) == 1:
            y = int(extracted_dates[0][1][:4]) - 1
            return extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2], "추출불가", f"{y}{extracted_dates[0][1][4:]}", "전기"

    return "추출불가", "UNKNOWN", "당기", "추출불가", "UNKNOWN", "전기"


def auto_slice_financials(input_path: str, output_path: str) -> str:
    reader = PdfReader(input_path)
    writer = PdfWriter()
    target_pages: list[int] = []
    logger.info("문서 스캔 중... (총 %s페이지)", len(reader.pages))

    metric_keywords = [
        "매출",
        "매출액",
        "영업수익",
        "영업이익",
        "영업이익(손실)",
        "당기순이익",
        "당기순이익(손실)",
        "이자수익",
        "기타비용",
        "법인세비용",
        "법인세차감전순이익",
        "기타포괄손익",
        "배당수익",
        "보험손익",
        "보험비용",
        "투자손익",
    ]

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        clean_text = text.replace(" ", "").replace("\n", "")
        if "포괄손익계산서" in clean_text and "연결" not in clean_text:
            hit_count = sum(1 for k in metric_keywords if k in clean_text)
            if hit_count >= 2:
                logger.info("(별도)포괄손익계산서 페이지 발견: %s", i)
                target_pages.append(i)

    if not target_pages:
        logger.info("전체 페이지 사용")
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

    logger.info("필터링 완료: %s", pages)
    return output_path

__all__ = [
    "sort_period_keys",
    "pick_reference_index",
    "unit_multiplier_to_million",
    "get_quarter",
    "extract_financial_periods",
    "auto_slice_financials",
]
