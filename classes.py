from __future__ import annotations

"""
공통 스키마/타입 모음.
"""

from typing import Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, Field

Unit = Literal["원", "천원", "백만원", "천만원", "억원", "UNKNOWN"]
CompareMode = Literal["QoQ", "YoY", "UNKNOWN"]
PeriodKey = str  # "YYYYQn"


CANDIDATE_METRICS = [
    "영업이익(손실)",
    "법인세비용차감전순이익(손실)",
    "법인세비용(수익)",
    "당기순이익(손실)",
    "영업수익",
    "매출액",
    "순이자이익",
    "수수료수익",
    "매출원가",
    "매출총이익",
    "판매비와관리비",
    "영업이익",
    "법인세비용차감전순이익",
    "법인세비용",
    "당기순이익",
    "총포괄손익",
    "지배기업소유주지분순이익",
    "비지배지분순이익",
    "자산총계",
    "유동자산",
    "비유동자산",
    "유형자산",
    "무형자산",
    "부채총계",
    "유동부채",
    "비유동부채",
    "차입금",
    "자본총계",
    "지배기업소유주지분",
    "이익잉여금",
    "영업활동현금흐름",
    "투자활동현금흐름",
    "재무활동현금흐름",
    "수익(매출액)",
    "수익",
    "영업비용",
    "금융수익",
    "금융비용",
    "기타수익",
    "기타비용",
    "기타포괄손익",
]


class MetricSelectionOutput(BaseModel):
    selected_metrics: List[str] = Field(description="제공된 보기 중 핵심 지표 5~10개")


class LLMExtractionOutput(BaseModel):
    unit_raw: str
    now_period: str
    ref_period: str
    now_period_key: str
    ref_period_key: str
    items: Dict[str, Dict[str, Union[str, int, float, None]]]


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
    summary_table: Optional[str] = None


class FGState(TypedDict, total=False):
    pdf_paths: List[str]
    compare: CompareMode
    top_k: int
    doc_bundle_by_pdf: Dict[str, Dict[str, object]]
    selected_metrics: List[str]
    llm_extraction_by_pdf: Dict[str, LLMExtractionOutput]
    fin: NormalizedFinancials
    now_period: PeriodKey
    ref_period: Optional[PeriodKey]
    ref_found: bool
    top_moves: List[TopMove]
    llm_reasoning: Optional[GrowthReasoningOutput]
    warnings: List[str]

__all__ = [
    "Unit",
    "CompareMode",
    "PeriodKey",
    "CANDIDATE_METRICS",
    "MetricSelectionOutput",
    "LLMExtractionOutput",
    "NormalizedFinancials",
    "TopMove",
    "GrowthReasoningInput",
    "GrowthReasoningOutput",
    "FGState",
]

